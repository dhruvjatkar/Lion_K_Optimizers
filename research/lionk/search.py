from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import SearchConfig
from .io import (
    append_jsonl,
    atomic_write_json,
    make_study_layout,
    read_jsonl,
    trial_artifact_paths,
    write_summary,
)
from .kernels import (
    power_spectral_update,
    soft_huber_spectral_update,
    zeropower_via_newtonschulz5,
)
from .metrics import objective_from_trial, summarize_seed_metrics


def _candidate_signature(candidate: Dict[str, Any]) -> str:
    payload = json.dumps(candidate, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _build_trial_id(fidelity: str, index: int, candidate: Dict[str, Any]) -> str:
    return f"{fidelity.lower()}_{index:04d}_{_candidate_signature(candidate)}"


# ---------------------------------------------------------------------------
#  Candidate generation
# ---------------------------------------------------------------------------

# Schedule choices mapped to Sobol-dimension index.
_SCHEDULES = ("static", "linear", "cosine")


def _sobol_warmstart(cfg: SearchConfig, count: int, seed: int) -> List[Dict[str, Any]]:
    """
    Generate candidates via 5-D quasi-random Sobol sequence.

    Dimensions:
      0 - log(delta)       in [log(delta_min), log(delta_max)]
      1 - log(delta_final) in [log(delta_min), log(delta_max)]  (independent of delta)
      2 - schedule          categorical {static, linear, cosine}
      3 - ns_steps           in [ns_step_min .. ns_step_max]
      4 - alpha              in [alpha_min, alpha_max]  (for power_k)

    Half the budget uses soft_huber_k (alpha is ignored at runtime),
    the other half uses power_k with the sampled alpha.
    """
    if count <= 0:
        return []

    sobol = torch.quasirandom.SobolEngine(dimension=5, scramble=True, seed=seed)
    draws = sobol.draw(count)

    log_min = math.log(cfg.delta_min)
    log_max = math.log(cfg.delta_max)
    ns_span = cfg.ns_step_max - cfg.ns_step_min + 1

    candidates: List[Dict[str, Any]] = []
    for idx, row in enumerate(draws):
        log_d = log_min + float(row[0]) * (log_max - log_min)
        log_df = log_min + float(row[1]) * (log_max - log_min)
        sched_idx = min(len(_SCHEDULES) - 1, int(float(row[2]) * len(_SCHEDULES)))
        ns = cfg.ns_step_min + min(ns_span - 1, int(float(row[3]) * ns_span))
        alpha = cfg.alpha_min + float(row[4]) * (cfg.alpha_max - cfg.alpha_min)

        # Alternate between soft_huber_k and power_k across the budget
        mode = "power_k" if idx % 2 == 0 else "soft_huber_k"

        candidates.append(
            {
                "mode": mode,
                "delta": float(math.exp(log_d)),
                "delta_final": float(math.exp(log_df)),
                "schedule": _SCHEDULES[sched_idx],
                "schedule_frac": 1.0,
                "ns_steps": int(ns),
                "eps": 1e-7,
                "max_update_norm": 0.0,
                "alpha": float(alpha),
            }
        )
    return candidates


def _local_gaussian_proposals(
    cfg: SearchConfig,
    best: Dict[str, Any],
    count: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Gaussian local proposals centred on the best candidate found so far."""
    if count <= 0:
        return []

    proposals: List[Dict[str, Any]] = []
    mean_log_delta = math.log(float(best["delta"]))
    mean_log_delta_final = math.log(float(best.get("delta_final", best["delta"])))
    best_mode = str(best.get("mode", "soft_huber_k"))
    best_alpha = float(best.get("alpha", 0.5))
    best_sched = str(best.get("schedule", "static"))

    for _ in range(count):
        log_d = rng.gauss(mean_log_delta, 0.35)
        log_d = min(max(log_d, math.log(cfg.delta_min)), math.log(cfg.delta_max))

        log_df = rng.gauss(mean_log_delta_final, 0.35)
        log_df = min(max(log_df, math.log(cfg.delta_min)), math.log(cfg.delta_max))

        ns = int(round(rng.gauss(float(best["ns_steps"]), 0.8)))
        ns = min(max(ns, cfg.ns_step_min), cfg.ns_step_max)

        alpha = rng.gauss(best_alpha, 0.15)
        alpha = min(max(alpha, cfg.alpha_min), cfg.alpha_max)

        # Small chance of switching schedule / mode for diversity
        sched = best_sched
        if rng.random() < 0.2:
            sched = rng.choice(list(_SCHEDULES))

        mode = best_mode
        if rng.random() < 0.2:
            mode = rng.choice(["soft_huber_k", "power_k"])

        proposals.append(
            {
                "mode": mode,
                "delta": float(math.exp(log_d)),
                "delta_final": float(math.exp(log_df)),
                "schedule": sched,
                "schedule_frac": 1.0,
                "ns_steps": int(ns),
                "eps": 1e-7,
                "max_update_norm": 0.0,
                "alpha": float(alpha),
            }
        )

    return proposals


# ---------------------------------------------------------------------------
#  F0 overhead benchmarking
# ---------------------------------------------------------------------------

# Representative shapes from the actual CifarNet architecture:
#   Conv(24, 64, 3x3) -> (64, 216),  Conv(64, 256, 3x3) -> (256, 576),
#   Conv(256, 256, 3x3) -> (256, 2304).
# These match the gradient tensors that get reshaped via g.reshape(len(g), -1)
# in the optimizer step, giving much more realistic overhead estimates than
# the old square/small shapes.
_BENCH_SHAPES = [(64, 216), (256, 576), (256, 2304)]


def _kernel_time_ms(
    mode: str,
    delta: float,
    ns_steps: int,
    eps: float,
    alpha: float,
    device: torch.device,
) -> float:
    """Benchmark a single kernel configuration averaged over representative shapes."""

    def run_once(g: torch.Tensor) -> None:
        if mode == "muon":
            zeropower_via_newtonschulz5(g, steps=ns_steps, eps=eps)
        elif mode == "power_k":
            power_spectral_update(g, alpha=alpha, delta=delta, eps=eps)
        else:
            soft_huber_spectral_update(g, delta=delta, steps=ns_steps, eps=eps)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        total_ms = 0.0
        for m, n in _BENCH_SHAPES:
            g = torch.randn(m, n, device=device, dtype=torch.float16)
            for _ in range(3):
                run_once(g)
            torch.cuda.synchronize()
            start.record()
            for _ in range(8):
                run_once(g)
            end.record()
            torch.cuda.synchronize()
            total_ms += start.elapsed_time(end) / 8.0
        return total_ms / len(_BENCH_SHAPES)

    total_sec = 0.0
    for m, n in _BENCH_SHAPES:
        g = torch.randn(m, n, device=device, dtype=torch.float32)
        for _ in range(3):
            run_once(g)
        t0 = time.perf_counter()
        for _ in range(8):
            run_once(g)
        total_sec += (time.perf_counter() - t0) / 8.0
    return (total_sec / len(_BENCH_SHAPES)) * 1000.0


def _f0_overhead(candidate: Dict[str, Any]) -> float:
    """
    Measure kernel overhead relative to Muon baseline.

    The baseline ALWAYS uses Muon with ns_steps=3 (its standard config),
    regardless of the candidate's ns_steps.  This ensures a fair comparison:
    a candidate using more NS steps correctly shows higher overhead.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Baseline: standard Muon with 3 NS steps
    base_ms = _kernel_time_ms(
        mode="muon", delta=1.0, ns_steps=3, eps=1e-7, alpha=1.0, device=device
    )
    cand_ms = _kernel_time_ms(
        mode=str(candidate["mode"]),
        delta=float(candidate["delta"]),
        ns_steps=int(candidate["ns_steps"]),
        eps=float(candidate["eps"]),
        alpha=float(candidate.get("alpha", 0.5)),
        device=device,
    )
    if base_ms <= 0:
        return 0.0
    return 100.0 * (cand_ms - base_ms) / base_ms


# ---------------------------------------------------------------------------
#  Subprocess evaluation
# ---------------------------------------------------------------------------


def _run_subprocess(cmd: List[str], stdout_path: Path) -> Tuple[int, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    combined = proc.stdout
    if proc.stderr:
        combined += "\n[stderr]\n" + proc.stderr
    stdout_path.write_text(combined)
    return proc.returncode, combined


def _script_metrics(
    python_exe: str,
    script: str,
    fidelity: str,
    seed: int,
    candidate: Dict[str, Any],
    target_acc: float,
    metrics_path: Path,
    stdout_path: Path,
) -> Dict[str, Any]:
    cmd = [
        python_exe,
        script,
        "--runs",
        "1",
        "--seed",
        str(seed),
        "--fidelity",
        fidelity,
        "--target-acc",
        str(target_acc),
        "--matrix-opt",
        str(candidate["mode"]),
        "--k-delta",
        str(candidate["delta"]),
        "--k-delta-final",
        str(candidate["delta_final"]),
        "--k-schedule",
        str(candidate["schedule"]),
        "--k-schedule-frac",
        str(candidate["schedule_frac"]),
        "--k-ns-steps",
        str(candidate["ns_steps"]),
        "--k-eps",
        str(candidate["eps"]),
        "--k-max-update-norm",
        str(candidate["max_update_norm"]),
        "--k-alpha",
        str(candidate.get("alpha", 0.5)),
        "--json-output",
        str(metrics_path),
        "--quiet",
    ]

    code, _ = _run_subprocess(cmd, stdout_path)
    if code != 0:
        return {
            "status": "failed",
            "reason": f"Script failed with exit code {code}",
            "time_to_target": None,
            "tta_val_acc": None,
        }

    if not metrics_path.exists():
        return {
            "status": "failed",
            "reason": "Missing metrics JSON output",
            "time_to_target": None,
            "tta_val_acc": None,
        }

    with metrics_path.open("r") as f:
        payload = json.load(f)

    return {
        "status": "ok",
        "reason": "",
        "time_to_target": payload.get("time_to_target"),
        "tta_val_acc": payload.get("mean_tta_val_acc"),
        "raw": payload,
    }


# ---------------------------------------------------------------------------
#  Fidelity evaluation
# ---------------------------------------------------------------------------


def _evaluate_fidelity(
    cfg: SearchConfig,
    candidate: Dict[str, Any],
    fidelity: str,
    seed_count: int,
    base_seed: int,
    script: str,
    artifact_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    seed_metrics: List[Dict[str, Any]] = []

    for seed_offset in range(seed_count):
        seed = base_seed + seed_offset
        metrics_path = artifact_dir / f"metrics_seed_{seed}.json"
        stdout_path = artifact_dir / f"stdout_seed_{seed}.log"
        result = _script_metrics(
            python_exe=cfg.python_exe,
            script=script,
            fidelity=fidelity,
            seed=seed,
            candidate=candidate,
            target_acc=cfg.target_acc,
            metrics_path=metrics_path,
            stdout_path=stdout_path,
        )
        row = {
            "seed": seed,
            "status": result["status"],
            "reason": result["reason"],
            "time_to_target": result.get("time_to_target"),
            "tta_val_acc": result.get("tta_val_acc"),
        }
        seed_metrics.append(row)

    good = [m for m in seed_metrics if m["status"] == "ok"]
    summary = summarize_seed_metrics(good)
    unstable = len(good) != len(seed_metrics)
    return {
        "unstable": unstable,
        "summary": summary,
    }, seed_metrics


# ---------------------------------------------------------------------------
#  Main search loop
# ---------------------------------------------------------------------------


def run_search(study_cfg: SearchConfig | Dict[str, Any]) -> Dict[str, Any]:
    cfg = (
        study_cfg
        if isinstance(study_cfg, SearchConfig)
        else SearchConfig.from_dict(study_cfg)
    )
    cfg = cfg.validate()

    layout = make_study_layout(Path(cfg.trials_root), cfg.study_id)
    atomic_write_json(layout["study_config"], cfg.to_dict())

    previous_rows = read_jsonl(layout["trials"])
    seen_trial_ids = {row.get("trial_id") for row in previous_rows}

    rng = random.Random(cfg.search_seed)

    baseline_candidate = {
        "mode": "muon",
        "delta": 1.0,
        "delta_final": 1.0,
        "schedule": "static",
        "schedule_frac": 1.0,
        "ns_steps": 3,
        "eps": 1e-7,
        "max_update_norm": 0.0,
        "alpha": 1.0,
    }

    sobol_count = max(0, cfg.f1_budget - 1)
    candidates = [baseline_candidate] + _sobol_warmstart(
        cfg, sobol_count, cfg.search_seed
    )

    promoted_f2: List[Dict[str, Any]] = []
    f1_results: List[Dict[str, Any]] = []
    best_candidate = baseline_candidate
    best_objective = float("inf")

    # Phase F0 + F1
    for index, candidate in enumerate(candidates):
        trial_id = _build_trial_id("F1", index, candidate)
        if trial_id in seen_trial_ids:
            continue

        artifacts = trial_artifact_paths(layout["study_root"], trial_id)
        atomic_write_json(artifacts["params"], candidate)

        overhead_pct = _f0_overhead(candidate)
        if overhead_pct > cfg.max_overhead_pct:
            row = {
                "trial_id": trial_id,
                "fidelity": "F1",
                "seed": cfg.search_seed,
                "params": candidate,
                "overhead_pct": overhead_pct,
                "time_to_target": None,
                "tta_val_acc": None,
                "status": "pruned",
                "reason": "overhead_cap_exceeded",
            }
            append_jsonl(layout["trials"], row)
            artifacts["status"].write_text("pruned_overhead\n")
            continue

        eval_result, seed_rows = _evaluate_fidelity(
            cfg=cfg,
            candidate=candidate,
            fidelity="F1",
            seed_count=cfg.seeds_f1,
            base_seed=cfg.search_seed + index * 100,
            script=cfg.script94,
            artifact_dir=artifacts["dir"],
        )

        summary = eval_result["summary"]
        t_target = summary["median_time_to_target"]
        tta = summary["median_tta_val_acc"]
        unstable = bool(eval_result["unstable"])
        obj = objective_from_trial(
            time_to_target=t_target,
            overhead_pct=overhead_pct,
            max_overhead_pct=cfg.max_overhead_pct,
            unstable=unstable,
        )

        row = {
            "trial_id": trial_id,
            "fidelity": "F1",
            "seed": cfg.search_seed,
            "params": candidate,
            "overhead_pct": overhead_pct,
            "time_to_target": t_target,
            "tta_val_acc": tta,
            "status": "ok" if not unstable else "unstable",
            "reason": "",
            "objective": obj,
            "seed_rows": seed_rows,
        }
        append_jsonl(layout["trials"], row)
        atomic_write_json(artifacts["metrics"], row)
        artifacts["status"].write_text(
            ("ok" if not unstable else "unstable") + "\n"
        )

        f1_results.append(row)
        if obj + cfg.promote_epsilon < best_objective:
            best_objective = obj
            best_candidate = candidate

    f1_sorted = sorted(
        [r for r in f1_results if r["status"] == "ok"],
        key=lambda r: r["objective"],
    )

    for row in f1_sorted:
        if len(promoted_f2) >= cfg.f2_budget:
            break
        promoted_f2.append(row)
        append_jsonl(
            layout["promotions"],
            {
                "from": "F1",
                "to": "F2",
                "trial_id": row["trial_id"],
                "objective": row["objective"],
            },
        )

    # Optional local refinement around current best for remaining F2 slots.
    if len(promoted_f2) < cfg.f2_budget:
        extra = cfg.f2_budget - len(promoted_f2)
        local_candidates = _local_gaussian_proposals(cfg, best_candidate, extra, rng)
        for offset, candidate in enumerate(local_candidates):
            trial_id = _build_trial_id("F1", len(candidates) + offset, candidate)
            artifacts = trial_artifact_paths(layout["study_root"], trial_id)
            atomic_write_json(artifacts["params"], candidate)

            overhead_pct = _f0_overhead(candidate)
            if overhead_pct > cfg.max_overhead_pct:
                row = {
                    "trial_id": trial_id,
                    "fidelity": "F1",
                    "seed": cfg.search_seed,
                    "params": candidate,
                    "overhead_pct": overhead_pct,
                    "time_to_target": None,
                    "tta_val_acc": None,
                    "status": "pruned",
                    "reason": "overhead_cap_exceeded",
                }
                append_jsonl(layout["trials"], row)
                artifacts["status"].write_text("pruned_overhead\n")
                continue

            eval_result, seed_rows = _evaluate_fidelity(
                cfg=cfg,
                candidate=candidate,
                fidelity="F1",
                seed_count=cfg.seeds_f1,
                base_seed=cfg.search_seed + 100000 + offset * 100,
                script=cfg.script94,
                artifact_dir=artifacts["dir"],
            )
            summary = eval_result["summary"]
            t_target = summary["median_time_to_target"]
            tta = summary["median_tta_val_acc"]
            unstable = bool(eval_result["unstable"])
            obj = objective_from_trial(
                time_to_target=t_target,
                overhead_pct=overhead_pct,
                max_overhead_pct=cfg.max_overhead_pct,
                unstable=unstable,
            )
            row = {
                "trial_id": trial_id,
                "fidelity": "F1",
                "seed": cfg.search_seed,
                "params": candidate,
                "overhead_pct": overhead_pct,
                "time_to_target": t_target,
                "tta_val_acc": tta,
                "status": "ok" if not unstable else "unstable",
                "reason": "",
                "objective": obj,
                "seed_rows": seed_rows,
            }
            append_jsonl(layout["trials"], row)
            atomic_write_json(artifacts["metrics"], row)
            artifacts["status"].write_text(
                ("ok" if not unstable else "unstable") + "\n"
            )

            if row["status"] == "ok":
                promoted_f2.append(row)
                append_jsonl(
                    layout["promotions"],
                    {
                        "from": "F1",
                        "to": "F2",
                        "trial_id": row["trial_id"],
                        "objective": row["objective"],
                    },
                )

            if len(promoted_f2) >= cfg.f2_budget:
                break

    # Phase F2
    f2_rows: List[Dict[str, Any]] = []
    for index, base_row in enumerate(promoted_f2):
        candidate = base_row["params"]
        trial_id = _build_trial_id("F2", index, candidate)
        artifacts = trial_artifact_paths(layout["study_root"], trial_id)
        atomic_write_json(artifacts["params"], candidate)

        eval_result, seed_rows = _evaluate_fidelity(
            cfg=cfg,
            candidate=candidate,
            fidelity="F2",
            seed_count=cfg.seeds_f2,
            base_seed=cfg.search_seed + 200000 + index * 100,
            script=cfg.script94,
            artifact_dir=artifacts["dir"],
        )

        summary = eval_result["summary"]
        overhead_pct = float(base_row.get("overhead_pct", 0.0))
        unstable = bool(eval_result["unstable"])
        obj = objective_from_trial(
            time_to_target=summary["median_time_to_target"],
            overhead_pct=overhead_pct,
            max_overhead_pct=cfg.max_overhead_pct,
            unstable=unstable,
        )

        row = {
            "trial_id": trial_id,
            "fidelity": "F2",
            "seed": cfg.search_seed,
            "params": candidate,
            "overhead_pct": overhead_pct,
            "time_to_target": summary["median_time_to_target"],
            "tta_val_acc": summary["median_tta_val_acc"],
            "status": "ok" if not unstable else "unstable",
            "reason": "",
            "objective": obj,
            "seed_rows": seed_rows,
        }
        append_jsonl(layout["trials"], row)
        atomic_write_json(artifacts["metrics"], row)
        artifacts["status"].write_text(
            ("ok" if not unstable else "unstable") + "\n"
        )

        if row["status"] == "ok":
            f2_rows.append(row)

    f2_sorted = sorted(f2_rows, key=lambda r: r["objective"])[: cfg.f3_budget]
    for row in f2_sorted:
        append_jsonl(
            layout["promotions"],
            {
                "from": "F2",
                "to": "F3",
                "trial_id": row["trial_id"],
                "objective": row["objective"],
            },
        )

    # Phase F3 + transfer checks
    f3_rows: List[Dict[str, Any]] = []
    for index, base_row in enumerate(f2_sorted):
        candidate = base_row["params"]
        trial_id = _build_trial_id("F3", index, candidate)
        artifacts = trial_artifact_paths(layout["study_root"], trial_id)
        atomic_write_json(artifacts["params"], candidate)

        eval94, seed_rows = _evaluate_fidelity(
            cfg=cfg,
            candidate=candidate,
            fidelity="F3",
            seed_count=cfg.seeds_f3,
            base_seed=cfg.search_seed + 300000 + index * 100,
            script=cfg.script94,
            artifact_dir=artifacts["dir"],
        )

        transfer95 = _script_metrics(
            python_exe=cfg.python_exe,
            script=cfg.script95,
            fidelity="F3",
            seed=cfg.search_seed + 400000 + index,
            candidate=candidate,
            target_acc=0.95,
            metrics_path=artifacts["dir"] / "metrics_95.json",
            stdout_path=artifacts["dir"] / "stdout_95.log",
        )
        transfer96 = _script_metrics(
            python_exe=cfg.python_exe,
            script=cfg.script96,
            fidelity="F3",
            seed=cfg.search_seed + 500000 + index,
            candidate=candidate,
            target_acc=0.96,
            metrics_path=artifacts["dir"] / "metrics_96.json",
            stdout_path=artifacts["dir"] / "stdout_96.log",
        )

        summary94 = eval94["summary"]
        overhead_pct = float(base_row.get("overhead_pct", 0.0))
        unstable = (
            bool(eval94["unstable"])
            or transfer95["status"] != "ok"
            or transfer96["status"] != "ok"
        )
        obj = objective_from_trial(
            time_to_target=summary94["median_time_to_target"],
            overhead_pct=overhead_pct,
            max_overhead_pct=cfg.max_overhead_pct,
            unstable=unstable,
        )

        row = {
            "trial_id": trial_id,
            "fidelity": "F3",
            "seed": cfg.search_seed,
            "params": candidate,
            "overhead_pct": overhead_pct,
            "time_to_target": summary94["median_time_to_target"],
            "tta_val_acc": summary94["median_tta_val_acc"],
            "status": "ok" if not unstable else "unstable",
            "reason": "",
            "objective": obj,
            "seed_rows": seed_rows,
            "transfer95": {
                "status": transfer95["status"],
                "tta_val_acc": transfer95.get("tta_val_acc"),
                "time_to_target": transfer95.get("time_to_target"),
                "reason": transfer95.get("reason", ""),
            },
            "transfer96": {
                "status": transfer96["status"],
                "tta_val_acc": transfer96.get("tta_val_acc"),
                "time_to_target": transfer96.get("time_to_target"),
                "reason": transfer96.get("reason", ""),
            },
        }
        append_jsonl(layout["trials"], row)
        atomic_write_json(artifacts["metrics"], row)
        artifacts["status"].write_text(
            ("ok" if not unstable else "unstable") + "\n"
        )

        if row["status"] == "ok":
            f3_rows.append(row)

    if f3_rows:
        best = min(f3_rows, key=lambda r: r["objective"])
    elif f2_rows:
        best = min(f2_rows, key=lambda r: r["objective"])
    elif f1_results:
        best = min(f1_results, key=lambda r: r.get("objective", float("inf")))
    else:
        best = {
            "trial_id": "none",
            "params": baseline_candidate,
            "objective": float("inf"),
            "time_to_target": None,
            "tta_val_acc": None,
            "status": "failed",
        }

    atomic_write_json(layout["best"], best)
    atomic_write_json(
        layout["search_state"],
        {
            "study_id": cfg.study_id,
            "phase": "done",
            "next_trial_index": len(read_jsonl(layout["trials"])),
            "best_trial_id": best.get("trial_id"),
            "best_objective": best.get("objective"),
        },
    )

    summary_lines = [
        "# Lion-K Spectral Search Summary",
        "",
        f"Study ID: {cfg.study_id}",
        f"Trials root: {cfg.trials_root}",
        f"Best trial: {best.get('trial_id')}",
        f"Best objective: {best.get('objective')}",
        f"Best time_to_target: {best.get('time_to_target')}",
        f"Best tta_val_acc: {best.get('tta_val_acc')}",
        f"Best params: {json.dumps(best.get('params', {}), sort_keys=True)}",
    ]
    write_summary(layout["summary"], summary_lines)

    return {
        "study_id": cfg.study_id,
        "study_root": str(layout["study_root"]),
        "best": best,
    }
