from __future__ import annotations

from statistics import median
from typing import Dict, Iterable, List, Optional


def first_time_to_target(
    points: Iterable[Dict[str, float]],
    target_acc: float,
    time_key: str = "time_seconds",
    acc_key: str = "tta_val_acc",
) -> Optional[float]:
    for point in points:
        acc = point.get(acc_key)
        if acc is None:
            continue
        if acc >= target_acc:
            return float(point.get(time_key, 0.0))
    return None


def safe_median(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(median(filtered))


def objective_from_trial(
    time_to_target: Optional[float],
    overhead_pct: float,
    max_overhead_pct: float,
    unstable: bool,
    lambda_overhead: float = 0.2,
    gamma_instability: float = 1000.0,
) -> float:
    if time_to_target is None:
        base = 1e9
    else:
        base = float(time_to_target)

    overhead_excess = max(0.0, overhead_pct - max_overhead_pct)
    penalty = lambda_overhead * overhead_excess * overhead_excess
    if unstable:
        penalty += gamma_instability
    return base + penalty


def summarize_seed_metrics(seed_metrics: List[Dict[str, float]]) -> Dict[str, Optional[float]]:
    times = [m.get("time_to_target") for m in seed_metrics]
    final_accs = [m.get("tta_val_acc") for m in seed_metrics]

    final_accs_filtered = [float(x) for x in final_accs if x is not None]
    return {
        "median_time_to_target": safe_median(times),
        "median_tta_val_acc": safe_median(final_accs),
        "mean_tta_val_acc": (sum(final_accs_filtered) / len(final_accs_filtered)) if final_accs_filtered else None,
    }
