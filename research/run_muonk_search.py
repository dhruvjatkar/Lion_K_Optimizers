from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from research.lionk.config import SearchConfig
from research.lionk.search import run_search


def _default_study_id(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{prefix}"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Lion-K multi-fidelity spectral search")
    p.add_argument("--study-id", type=str, default="")
    p.add_argument("--trials-root", type=str, default="research/lionk/trials")
    p.add_argument("--script94", type=str, default="research/airbench94_muon_simple.py")
    p.add_argument("--script95", type=str, default="research/airbench95_muonk_transfer.py")
    p.add_argument("--script96", type=str, default="research/airbench96_muonk_transfer.py")
    p.add_argument("--target-acc", type=float, default=0.94)
    p.add_argument("--max-overhead-pct", type=float, default=10.0)
    p.add_argument("--delta-min", type=float, default=1e-3)
    p.add_argument("--delta-max", type=float, default=8e-1)
    p.add_argument("--alpha-min", type=float, default=0.1)
    p.add_argument("--alpha-max", type=float, default=1.0)
    p.add_argument("--ns-step-min", type=int, default=2)
    p.add_argument("--ns-step-max", type=int, default=4)
    p.add_argument("--f1-budget", type=int, default=80)
    p.add_argument("--f2-budget", type=int, default=20)
    p.add_argument("--f3-budget", type=int, default=6)
    p.add_argument("--seeds-f1", type=int, default=2)
    p.add_argument("--seeds-f2", type=int, default=3)
    p.add_argument("--seeds-f3", type=int, default=5)
    p.add_argument("--search-seed", type=int, default=1337)
    p.add_argument("--promote-epsilon", type=float, default=1e-3)
    p.add_argument("--python-exe", type=str, default="python")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    study_id = args.study_id or _default_study_id("lionk_spectral")

    cfg = SearchConfig(
        study_id=study_id,
        trials_root=args.trials_root,
        script94=args.script94,
        script95=args.script95,
        script96=args.script96,
        target_acc=args.target_acc,
        max_overhead_pct=args.max_overhead_pct,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        ns_step_min=args.ns_step_min,
        ns_step_max=args.ns_step_max,
        f1_budget=args.f1_budget,
        f2_budget=args.f2_budget,
        f3_budget=args.f3_budget,
        seeds_f1=args.seeds_f1,
        seeds_f2=args.seeds_f2,
        seeds_f3=args.seeds_f3,
        search_seed=args.search_seed,
        promote_epsilon=args.promote_epsilon,
        python_exe=args.python_exe,
    ).validate()

    result = run_search(cfg)
    print(f"Study complete: {result['study_id']}")
    print(f"Study root: {result['study_root']}")
    print(f"Best trial: {result['best'].get('trial_id')}")


if __name__ == "__main__":
    main()
