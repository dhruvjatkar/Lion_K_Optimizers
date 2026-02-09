from __future__ import annotations

import math
from typing import Optional

import torch

from .config import KConfig
from .kernels import (
    power_spectral_update,
    soft_huber_spectral_update,
    zeropower_via_newtonschulz5,
)


class MuonK(torch.optim.Optimizer):
    """
    Muon-compatible optimizer with pluggable Lion-K spectral maps.

    Each spectral map h(sigma) applied to the momentum's singular values
    implicitly determines the convex regulariser K via K(X) = sum g(sigma_i(X))
    where g' = h.  Decoupled weight decay then minimises
        F_hat(X) = F(X) + (1/lambda) K^*(lambda X).

    Supported modes (set via k_config.mode):
      - "muon":         h = 1, polar factor, nuclear-norm constraint.
      - "soft_huber_k": h = s/sqrt(s^2+d^2), smooth Huber / controlled equalization.
      - "power_k":      h = s^a/(s^{2a}+d^{2a})^{1/2}, power-compressed family.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        nesterov: bool = False,
        k_config: Optional[KConfig] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")

        self.k_config = (k_config or KConfig()).validate()
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, step=0)
        super().__init__(params, defaults)

    def _delta_for_step(self, group_step: int, group: dict) -> float:
        """
        Compute the current delta value based on the schedule.

        Scheduling delta implements "equalize progressively over training":
        e.g. start with large delta (gentle, gradient-like) and anneal to
        small delta (aggressive, Muon-like).  This is the delta(t) schedule
        from the Lion-K Phase 1 recipe.
        """
        cfg = self.k_config
        if cfg.schedule == "static":
            return cfg.delta

        total_steps = int(group.get("total_steps", 0))
        if total_steps <= 0:
            total_steps = 1

        horizon = max(1, int(total_steps * cfg.schedule_frac))
        progress = min(max(group_step, 0) / float(horizon), 1.0)

        if cfg.schedule == "linear":
            delta = cfg.delta + progress * (cfg.delta_final - cfg.delta)
        elif cfg.schedule == "cosine":
            cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
            delta = cfg.delta + cosine * (cfg.delta_final - cfg.delta)
        else:
            raise ValueError(f"Unsupported schedule: {cfg.schedule}")

        lo = min(cfg.delta, cfg.delta_final)
        hi = max(cfg.delta, cfg.delta_final)
        return min(max(delta, lo), hi)

    def _compute_update(self, g2d: torch.Tensor, delta: float) -> torch.Tensor:
        cfg = self.k_config
        if cfg.mode == "muon":
            return zeropower_via_newtonschulz5(g2d, steps=cfg.ns_steps, eps=cfg.eps)
        if cfg.mode == "soft_huber_k":
            return soft_huber_spectral_update(
                g2d, delta=delta, steps=cfg.ns_steps, eps=cfg.eps
            )
        if cfg.mode == "power_k":
            return power_spectral_update(
                g2d, alpha=cfg.alpha, delta=delta, eps=cfg.eps
            )
        raise ValueError(f"Unsupported mode: {cfg.mode}")

    @torch.no_grad()
    def step(self):
        cfg = self.k_config

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            group_step = int(group.get("step", 0))
            delta = self._delta_for_step(group_step, group)

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue

                if not torch.isfinite(g).all():
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g_eff = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                # --- Parameter re-normalization (pre-update) ---
                # Normalize weight matrix to have Frobenius norm = sqrt(numel).
                # This enforces a "unit spectral scale" convention: after
                # normalization, a random orthogonal matrix would have
                # ||W||_F = sqrt(d) where d = numel.  This keeps the
                # effective learning rate consistent across layers of
                # different sizes and prevents gradual norm drift.
                # Equivalent to: p.data /= (||p||_F / sqrt(numel))
                p_norm = p.data.norm()
                if torch.isfinite(p_norm) and p_norm > 0:
                    p.data.mul_(len(p.data) ** 0.5 / p_norm)

                g2d = g_eff.reshape(len(g_eff), -1)
                update = self._compute_update(g2d, delta=delta).view_as(g_eff)

                if cfg.max_update_norm > 0:
                    update_norm = update.norm()
                    if (
                        torch.isfinite(update_norm)
                        and update_norm > cfg.max_update_norm
                    ):
                        update.mul_(cfg.max_update_norm / (update_norm + cfg.eps))

                if torch.isfinite(update).all():
                    p.data.add_(update, alpha=-lr)

            group["step"] = group_step + 1

        return None
