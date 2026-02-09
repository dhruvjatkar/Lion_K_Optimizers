"""
airbench94_compiled.py

Fully-compiled CIFAR-10 94% benchmark with configurable Lion-K spectral function.
Based on airbench94_muon.py (2.59s on A100-SXM4-40GB with torch==2.4.1).

All original optimizations preserved:
  - model.compile(mode="reduce-overhead") for fused forward+backward
  - Custom GPU-resident CifarLoader (no CPU-GPU transfers)
  - cudnn.benchmark for conv autotuning
  - Half-precision model with float32 batchnorm
  - Warmup run to amortize compilation cost
  - CUDA event timing (excludes compilation overhead)

Usage:
  # Muon baseline (should match ~2.6s on A100-40GB)
  python research/airbench94_compiled.py --matrix-opt muon --runs 10

  # Soft-Huber search candidate
  python research/airbench94_compiled.py --matrix-opt soft_huber_k --k-delta 0.3

  # Power-K search candidate
  python research/airbench94_compiled.py --matrix-opt power_k --k-alpha 0.5 --k-delta 0.2

  # Search pipeline invocation
  python research/airbench94_compiled.py --runs 1 --seed 42 --fidelity F1 \\
      --matrix-opt soft_huber_k --k-delta 0.3 --json-output metrics.json --quiet
"""

#############################################
#                  Setup                    #
#############################################

import argparse
import json
import os
import sys
from math import ceil, cos, pi
from statistics import mean, median

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True

#############################################
#    Compiled Spectral Kernels (Lion-K)     #
#############################################
#
# All kernels are @torch.compile'd for fair wall-clock comparison.
# Delta is passed as a 0-d tensor (not Python float) so the compiler
# traces it dynamically -- no recompilation when delta changes per step.


@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """
    Newton-Schulz iteration for the polar factor (Muon kernel).
    h(sigma) = 1 for all sigma > 0.
    K(X) = nuclear norm, nabla K = polar factor.
    Copied unchanged from airbench94_muon.py.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


@torch.compile
def soft_huber_update(G, delta_t, steps=3, eps=1e-7):
    """
    Soft-Huber spectral update: (G G^T + delta^2 I)^{-1/2} G.
    h_delta(sigma) = sigma / sqrt(sigma^2 + delta^2).
    delta_t must be a 0-d tensor (for dynamic tracing, no recompilation).
    """
    assert len(G.shape) == 2
    transposed = G.size(0) > G.size(1)
    X = G.T.float() if transposed else G.float()
    X = X / (X.norm() + eps)

    # Build covariance + regularization
    cov = X @ X.T
    n = cov.size(0)
    eye = torch.eye(n, device=cov.device, dtype=cov.dtype)
    cov = cov + (delta_t * delta_t) * eye

    # Newton-Schulz iteration for cov^{-1/2}
    scale = torch.clamp(torch.trace(cov) / float(n), min=eps)
    Y = cov / scale
    Z = eye.clone()
    for _ in range(steps):
        T = 0.5 * (3.0 * eye - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    inv_sqrt = Z / torch.sqrt(scale)

    out = (inv_sqrt @ X).to(G.dtype)
    return out.T if transposed else out


@torch.compile
def power_update(G, alpha, delta_t, eps=1e-7):
    """
    Power-compressed spectral update via SVD.
    h_{alpha,delta}(sigma) = sigma^alpha / (sigma^{2*alpha} + delta^{2*alpha})^{1/2}.
    alpha is a Python float (constant per process, compiler specializes on it).
    delta_t must be a 0-d tensor (dynamic).
    """
    assert len(G.shape) == 2
    transposed = G.size(0) > G.size(1)
    X = G.T.float() if transposed else G.float()

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    S_clamped = S.clamp(min=eps)
    S_alpha = S_clamped.pow(alpha)
    delta_alpha = delta_t.pow(alpha)
    h = S_alpha / torch.sqrt(S_alpha.square() + delta_alpha.square())

    out = ((U * h.unsqueeze(0)) @ Vh).to(G.dtype)
    return out.T if transposed else out


#############################################
#     MuonK Optimizer (inline, minimal)     #
#############################################


class MuonK(torch.optim.Optimizer):
    """
    Muon optimizer with configurable Lion-K spectral function.
    Inlined for maximum performance -- no import overhead, no dispatch layers.
    """

    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False,
                 mode="muon", delta=0.1, delta_final=None, schedule="static",
                 ns_steps=3, eps=1e-7, alpha=0.5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.mode = mode
        self.delta = delta
        self.delta_final = delta_final if delta_final is not None else delta
        self.schedule = schedule
        self.ns_steps = ns_steps
        self.eps = eps
        self.alpha = alpha

    def _current_delta(self, step_num, total_steps):
        if self.schedule == "static":
            return self.delta
        progress = min(1.0, step_num / max(1, total_steps))
        if self.schedule == "linear":
            return self.delta + (self.delta_final - self.delta) * progress
        elif self.schedule == "cosine":
            return self.delta + (self.delta_final - self.delta) * 0.5 * (1 - cos(pi * progress))
        return self.delta

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            step_num = group.get("step", 0)
            total_steps = group.get("total_steps", 1)
            delta_val = self._current_delta(step_num, total_steps)

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                p.data.mul_(len(p.data)**0.5 / p.data.norm())  # normalize the weight
                g_flat = g.reshape(len(g), -1)

                if self.mode == "muon":
                    update = zeropower_via_newtonschulz5(g_flat, steps=self.ns_steps, eps=self.eps)
                elif self.mode == "soft_huber_k":
                    delta_t = torch.tensor(delta_val, device=g.device, dtype=torch.float32)
                    update = soft_huber_update(g_flat, delta_t, steps=self.ns_steps, eps=self.eps)
                elif self.mode == "power_k":
                    delta_t = torch.tensor(delta_val, device=g.device, dtype=torch.float32)
                    update = power_update(g_flat, self.alpha, delta_t, eps=self.eps)
                else:
                    update = zeropower_via_newtonschulz5(g_flat, steps=self.ns_steps, eps=self.eps)

                p.data.add_(update.view(g.shape), alpha=-lr)

            group["step"] = step_num + 1

#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class CifarLoader:

    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device("cuda"))
        self.images, self.labels, self.classes = data["images"], data["labels"], data["classes"]
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ["flip", "translate"], "Unrecognized key: %s" % k

        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,)*4, "reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        if self.aug.get("flip", False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

#############################################
#            Network Definition             #
#############################################

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = False

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
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
            ConvGroup(whiten_width,     widths["block1"]),
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
        patches = train_images.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1,c,h,w) / torch.sqrt(eigenvalues.view(-1,1,1,1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)

############################################
#               Evaluation                 #
############################################

def infer(model, loader, tta_level=0):
    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, "reflect")
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

############################################
#                Training                  #
############################################

def _fidelity_epochs_and_tta(fidelity):
    """F1=short screening, F2=medium, F3=full evaluation with TTA."""
    if fidelity == "F1":
        return 2.0, 0
    if fidelity == "F2":
        return 8.0, 0
    return 8.0, 2


def main(run, model, args, train_loader, test_loader):

    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size

    epochs, tta_level = _fidelity_epochs_and_tta(args.fidelity)

    if run == "warmup":
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)

    total_train_steps = ceil(epochs * len(train_loader))
    whiten_bias_train_steps = ceil(min(3.0, epochs) * len(train_loader))

    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    param_configs = [dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=norm_biases,         lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)]
    optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True, fused=True)
    optimizer2 = MuonK(filter_params, lr=0.24, momentum=0.6, nesterov=True,
                       mode=args.matrix_opt, delta=args.k_delta,
                       delta_final=args.k_delta_final, schedule=args.k_schedule,
                       ns_steps=args.k_ns_steps, eps=args.k_eps, alpha=args.k_alpha)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    for group in optimizer2.param_groups:
        group["total_steps"] = total_train_steps

    # CUDA event timing (accurate, excludes compilation overhead)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0
    def start_timer():
        starter.record()
    def stop_timer():
        ender.record()
        torch.cuda.synchronize()
        nonlocal time_seconds
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()
    step = 0

    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    for epoch in range(ceil(total_train_steps / len(train_loader))):
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum").backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / max(1, whiten_bias_train_steps))
            for group in optimizer1.param_groups[1:]+optimizer2.param_groups:
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
    tta_val_acc = evaluate(model, test_loader, tta_level=tta_level)
    stop_timer()

    time_to_target = time_seconds if tta_val_acc >= args.target_acc else None
    return {
        "tta_val_acc": float(tta_val_acc),
        "time_seconds": float(time_seconds),
        "time_to_target": time_to_target,
    }

############################################
#              CLI + Main                  #
############################################

def _build_arg_parser():
    p = argparse.ArgumentParser(description="Compiled airbench94 with Lion-K spectral search")
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
    p.add_argument("--k-alpha", type=float, default=0.5)
    p.add_argument("--json-output", type=str, default="")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--compile-mode", type=str, default="reduce-overhead",
                   help="torch.compile mode (reduce-overhead, max-autotune, default)")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    # Create model once and compile (expensive, but amortized over all runs)
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.compile(mode=args.compile_mode)

    # Create data loaders once (GPU-resident, reused across runs)
    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader("cifar10", train=True, batch_size=2000,
                               aug=dict(flip=True, translate=2))

    # Warmup run: compiles model + kernels, NOT timed in results
    if not args.quiet:
        print("Warmup (compiling model + kernels)...")
    main("warmup", model, args, train_loader, test_loader)
    if not args.quiet:
        print("Warmup complete. Starting timed runs.\n")

    # Timed runs
    run_metrics = []
    for run_idx in range(args.runs):
        if args.seed is not None:
            torch.manual_seed(args.seed + run_idx)
        result = main(run_idx, model, args, train_loader, test_loader)
        run_metrics.append(result)
        if not args.quiet:
            print(f"run={run_idx:03d}  tta_val_acc={result['tta_val_acc']:.4f}  "
                  f"time={result['time_seconds']:.3f}s  "
                  f"time_to_target={result['time_to_target']}")

    # Aggregate
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
        print(f"\nMean acc: {mean(accs):.4f}  Mean time: {mean(times):.3f}s")
        if times_to_target:
            print(f"Median time-to-target: {median(times_to_target):.3f}s")
        else:
            print("No runs reached target accuracy.")
