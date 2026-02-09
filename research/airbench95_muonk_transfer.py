from __future__ import annotations

import argparse
import json
import os
from math import ceil
from statistics import mean, median
from typing import Dict, List

import torch
from torch import nn

from airbench.lib_airbench95 import hyp, make_net95
from airbench.utils import CifarLoader, LookaheadState, evaluate, init_whitening_conv
from research.lionk.config import KConfig
from research.lionk.optimizer import MuonK

torch.backends.cudnn.benchmark = True


def _fidelity_epochs_and_tta(base_epochs: float, base_tta: int, fidelity: str) -> tuple[float, int]:
    if fidelity == "F1":
        return max(2.0, min(base_epochs, 4.0)), 0
    if fidelity == "F2":
        return base_epochs, 0
    return base_epochs, base_tta


def _run_once(args: argparse.Namespace, run_idx: int) -> Dict[str, float | None]:
    if args.seed is not None:
        torch.manual_seed(args.seed + run_idx)

    opt = hyp["opt"]
    net_hyp = hyp["net"]

    batch_size = opt["batch_size"]
    momentum = opt["momentum"]
    epochs, tta_level = _fidelity_epochs_and_tta(float(opt["train_epochs"]), int(net_hyp["tta_level"]), args.fidelity)

    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = opt["lr"] / kilostep_scale
    wd = opt["weight_decay"] * batch_size / kilostep_scale
    lr_biases = lr * opt["bias_scaler"]

    train_loader = CifarLoader("cifar10", train=True, batch_size=batch_size, aug=hyp["aug"], altflip=True)
    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    total_train_steps = ceil(len(train_loader) * epochs)

    model = make_net95()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=opt["label_smoothing"], reduction="none")

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    norm_biases = [p for n, p in named if "norm" in n]
    matrix_params = [p for n, p in named if "norm" not in n and p.ndim >= 2]
    vector_scalar_params = [p for n, p in named if "norm" not in n and p.ndim < 2]

    sgd_groups = [
        dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
        dict(params=vector_scalar_params, lr=lr, weight_decay=wd / lr),
    ]
    optimizer_sgd = torch.optim.SGD(sgd_groups, momentum=momentum, nesterov=True)

    k_config = KConfig(
        mode=args.matrix_opt,
        delta=args.k_delta,
        delta_final=args.k_delta_final,
        schedule=args.k_schedule,
        schedule_frac=args.k_schedule_frac,
        ns_steps=args.k_ns_steps,
        eps=args.k_eps,
        max_update_norm=args.k_max_update_norm,
        alpha=args.k_alpha,
    ).validate()
    optimizer_muonk = MuonK(matrix_params, lr=lr, momentum=momentum, nesterov=True, k_config=k_config)
    for group in optimizer_muonk.param_groups:
        group["total_steps"] = total_train_steps
        group["initial_lr"] = group["lr"]

    def get_lr(step: int) -> float:
        warmup_steps = int(total_train_steps * 0.23)
        warmdown_steps = max(1, total_train_steps - warmup_steps)
        if step < warmup_steps:
            frac = step / max(1, warmup_steps)
            return 0.2 * (1 - frac) + 1.0 * frac
        frac = (step - warmup_steps) / warmdown_steps
        return 1.0 * (1 - frac) + 0.07 * frac

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_sgd, get_lr)

    alpha_schedule = 0.95**5 * (torch.arange(total_train_steps + 1) / max(1, total_train_steps)) ** 3
    lookahead_state = LookaheadState(model)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    total_time_seconds = 0.0

    starter.record()
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model[0], train_images)
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    current_steps = 0
    for epoch in range(ceil(epochs)):
        model[0].bias.requires_grad = epoch < opt["whiten_bias_epochs"]

        starter.record()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels).sum()

            optimizer_sgd.zero_grad(set_to_none=True)
            optimizer_muonk.zero_grad(set_to_none=True)

            loss.backward()
            optimizer_sgd.step()

            lr_scale = get_lr(current_steps)
            for group in optimizer_muonk.param_groups:
                group["lr"] = group["initial_lr"] * lr_scale
            optimizer_muonk.step()

            scheduler.step()
            current_steps += 1

            if current_steps % 5 == 0:
                lookahead_state.update(model, decay=alpha_schedule[current_steps].item())

            if current_steps >= total_train_steps:
                lookahead_state.update(model, decay=1.0)
                break

        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)

        if current_steps >= total_train_steps:
            break

    starter.record()
    tta_val_acc = evaluate(model, test_loader, tta_level=tta_level)
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    time_to_target = total_time_seconds if tta_val_acc >= args.target_acc else None
    return {
        "tta_val_acc": float(tta_val_acc),
        "time_seconds": float(total_time_seconds),
        "time_to_target": time_to_target,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Airbench95 transfer runner with MuonK matrix optimizer")
    p.add_argument("--runs", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fidelity", choices=["F1", "F2", "F3"], default="F3")
    p.add_argument("--target-acc", type=float, default=0.95)
    p.add_argument("--matrix-opt", choices=["muon", "soft_huber_k", "power_k"], default="muon")
    p.add_argument("--k-delta", type=float, default=0.1)
    p.add_argument("--k-delta-final", type=float, default=0.1)
    p.add_argument("--k-schedule", choices=["static", "linear", "cosine"], default="static")
    p.add_argument("--k-schedule-frac", type=float, default=1.0)
    p.add_argument("--k-ns-steps", type=int, default=3)
    p.add_argument("--k-eps", type=float, default=1e-7)
    p.add_argument("--k-max-update-norm", type=float, default=0.0)
    p.add_argument("--k-alpha", type=float, default=0.5, help="Power-compression exponent for power_k mode")
    p.add_argument("--json-output", type=str, default="")
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    run_metrics: List[Dict[str, float | None]] = []
    for run_idx in range(args.runs):
        metrics = _run_once(args, run_idx)
        run_metrics.append(metrics)
        if not args.quiet:
            print(
                f"run={run_idx:03d} tta_val_acc={metrics['tta_val_acc']:.4f} "
                f"time_seconds={metrics['time_seconds']:.4f} time_to_target={metrics['time_to_target']}"
            )

    accs = [float(m["tta_val_acc"]) for m in run_metrics]
    times = [float(m["time_seconds"]) for m in run_metrics]
    times_to_target = [float(m["time_to_target"]) for m in run_metrics if m["time_to_target"] is not None]

    payload = {
        "runs": args.runs,
        "fidelity": args.fidelity,
        "matrix_opt": args.matrix_opt,
        "target_acc": args.target_acc,
        "mean_tta_val_acc": mean(accs),
        "median_tta_val_acc": median(accs),
        "mean_time_seconds": mean(times),
        "median_time_seconds": median(times),
        "time_to_target": (median(times_to_target) if times_to_target else None),
        "run_metrics": run_metrics,
    }

    if args.json_output:
        output_dir = os.path.dirname(args.json_output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

    if not args.quiet:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
