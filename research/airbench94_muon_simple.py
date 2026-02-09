from __future__ import annotations

import argparse
import json
import os
from math import ceil
from statistics import mean, median
from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F

import airbench
from research.lionk.config import KConfig
from research.lionk.optimizer import MuonK

torch.backends.cudnn.benchmark = True


# note the use of low BatchNorm stats momentum
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths["block1"]),
            ConvGroup(widths["block1"], widths["block2"]),
            ConvGroup(widths["block2"], widths["block3"]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths["block3"], 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1, c, h, w) / torch.sqrt(eigenvalues.view(-1, 1, 1, 1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)


def _fidelity_epochs_and_tta(fidelity: str) -> tuple[float, int]:
    if fidelity == "F1":
        return 2.0, 0
    if fidelity == "F2":
        return 8.0, 0
    return 8.0, 2


def _timer_enabled() -> bool:
    return torch.cuda.is_available()


def _run_once(args: argparse.Namespace, run_idx: int) -> Dict[str, float | None]:
    if args.seed is not None:
        torch.manual_seed(args.seed + run_idx)

    model = CifarNet().cuda().to(memory_format=torch.channels_last)

    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size

    epochs, tta_level = _fidelity_epochs_and_tta(args.fidelity)

    test_loader = airbench.CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = airbench.CifarLoader(
        "cifar10",
        train=True,
        batch_size=batch_size,
        aug=dict(flip=True, translate=2),
        altflip=True,
    )

    total_train_steps = ceil(epochs * len(train_loader))
    whiten_bias_train_steps = ceil(min(3.0, epochs) * len(train_loader))

    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True, fused=True)

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
    optimizer2 = MuonK(filter_params, lr=0.24, momentum=0.6, nesterov=True, k_config=k_config)
    for group in optimizer2.param_groups:
        group["total_steps"] = total_train_steps

    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    use_cuda_timers = _timer_enabled()
    time_seconds = 0.0
    if use_cuda_timers:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        def start_timer() -> None:
            starter.record()

        def stop_timer() -> None:
            nonlocal time_seconds
            ender.record()
            torch.cuda.synchronize()
            time_seconds += 1e-3 * starter.elapsed_time(ender)

    else:

        def start_timer() -> None:
            return None

        def stop_timer() -> None:
            return None

    model.reset()
    step = 0

    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    for _ in range(ceil(total_train_steps / len(train_loader))):
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum").backward()

            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / max(1, whiten_bias_train_steps))
            for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / max(1, total_train_steps))

            for opt in optimizers:
                opt.step()

            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break
        stop_timer()

        if step >= total_train_steps:
            break

    start_timer()
    tta_val_acc = airbench.evaluate(model, test_loader, tta_level=tta_level)
    stop_timer()

    time_to_target = time_seconds if tta_val_acc >= args.target_acc else None
    return {
        "tta_val_acc": float(tta_val_acc),
        "time_seconds": float(time_seconds),
        "time_to_target": time_to_target,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Airbench94 MuonK research runner")
    p.add_argument("--runs", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fidelity", choices=["F1", "F2", "F3"], default="F3")
    p.add_argument("--target-acc", type=float, default=0.94)
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
        metrics = _run_once(args, run_idx=run_idx)
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
        "k_config": {
            "mode": args.matrix_opt,
            "delta": args.k_delta,
            "delta_final": args.k_delta_final,
            "schedule": args.k_schedule,
            "schedule_frac": args.k_schedule_frac,
            "ns_steps": args.k_ns_steps,
            "eps": args.k_eps,
            "max_update_norm": args.k_max_update_norm,
            "alpha": args.k_alpha,
        },
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
