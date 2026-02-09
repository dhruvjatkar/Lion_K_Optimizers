from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal

# ---------------------------------------------------------------------------
# The three kernel modes correspond to specific Lion-K spectral families:
#   muon:         h(s) = 1            (polar factor / matrix sign)
#   soft_huber_k: h(s) = s/sqrt(s^2+d^2)  (smooth Huber, family A)
#   power_k:      h(s) = s^a/(s^{2a}+d^{2a})^{1/2}  (power-compressed, family B)
# ---------------------------------------------------------------------------
KMode = Literal["muon", "soft_huber_k", "power_k"]
KSchedule = Literal["static", "linear", "cosine"]
Fidelity = Literal["F0", "F1", "F2", "F3"]


@dataclass(frozen=True)
class KConfig:
    mode: KMode = "muon"
    delta: float = 0.1
    delta_final: float = 0.1
    schedule: KSchedule = "static"
    schedule_frac: float = 1.0
    ns_steps: int = 3
    eps: float = 1e-7
    max_update_norm: float = 0.0
    # alpha controls the compression exponent for power_k (family B).
    # alpha=1 reduces to soft_huber_k.  alpha->0 approaches Muon.
    alpha: float = 0.5

    def validate(self) -> "KConfig":
        if self.mode not in ("muon", "soft_huber_k", "power_k"):
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.mode in ("soft_huber_k", "power_k"):
            if self.delta <= 0 or self.delta_final <= 0:
                raise ValueError("delta and delta_final must be > 0")
        if self.mode == "power_k":
            if not (0 < self.alpha <= 1.0):
                raise ValueError("alpha must be in (0, 1] for power_k mode")
        if self.schedule not in ("static", "linear", "cosine"):
            raise ValueError(f"Unsupported schedule: {self.schedule}")
        if self.schedule_frac <= 0:
            raise ValueError("schedule_frac must be > 0")
        if self.ns_steps < 1:
            raise ValueError("ns_steps must be >= 1")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")
        if self.max_update_norm < 0:
            raise ValueError("max_update_norm must be >= 0")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "KConfig":
        return KConfig(**data).validate()


@dataclass(frozen=True)
class SearchConfig:
    study_id: str
    trials_root: str
    script94: str
    script95: str
    script96: str
    target_acc: float = 0.94
    max_overhead_pct: float = 10.0
    delta_min: float = 1e-3
    delta_max: float = 8e-1
    alpha_min: float = 0.1
    alpha_max: float = 1.0
    ns_step_min: int = 2
    ns_step_max: int = 4
    f1_budget: int = 80
    f2_budget: int = 20
    f3_budget: int = 6
    seeds_f1: int = 2  # raised from 1 to reduce F1 noise
    seeds_f2: int = 3
    seeds_f3: int = 5
    search_seed: int = 1337
    promote_epsilon: float = 1e-3
    python_exe: str = "python"

    def validate(self) -> "SearchConfig":
        if self.delta_min <= 0 or self.delta_max <= 0:
            raise ValueError("delta_min and delta_max must be > 0")
        if self.delta_min >= self.delta_max:
            raise ValueError("delta_min must be < delta_max")
        if not (0 < self.alpha_min <= self.alpha_max <= 1.0):
            raise ValueError("alpha bounds must satisfy 0 < alpha_min <= alpha_max <= 1")
        if self.ns_step_min < 1 or self.ns_step_max < self.ns_step_min:
            raise ValueError("Invalid ns step bounds")
        if self.f1_budget < 1 or self.f2_budget < 1 or self.f3_budget < 1:
            raise ValueError("All budgets must be >= 1")
        if self.seeds_f1 < 1 or self.seeds_f2 < 1 or self.seeds_f3 < 1:
            raise ValueError("All seed counts must be >= 1")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SearchConfig":
        return SearchConfig(**data).validate()
