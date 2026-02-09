# Added Code: Exact Technical Note

This note documents the MuonK search stack code added under `research/` and exactly how each part works.

## New/Updated File Paths

### New files
- `research/lionk/__init__.py`
- `research/lionk/config.py`
- `research/lionk/kernels.py`
- `research/lionk/optimizer.py`
- `research/lionk/metrics.py`
- `research/lionk/io.py`
- `research/lionk/search.py`
- `research/lionk/docs/ADDED_CODE_EXACT.md`
- `research/lionk/trials/README.md`
- `research/lionk/trials/studies/.gitkeep`
- `research/lionk/trials/archive/.gitkeep`
- `research/airbench95_muonk_transfer.py`
- `research/airbench96_muonk_transfer.py`
- `research/run_muonk_search.py`

### Updated files
- `research/airbench94_muon_simple.py`
- `README.md` (section appended)

### Added tests (not executed)
- `research/lionk/tests/test_optimizer_contract.py`
- `research/lionk/tests/test_kernels.py`
- `research/lionk/tests/test_io.py`

---

## Public Types and Function Signatures

### `research/lionk/config.py`

- `class KConfig:`
  - fields:
    - `mode: Literal["muon", "soft_huber_k"]`
    - `delta: float`
    - `delta_final: float`
    - `schedule: Literal["static", "linear", "cosine"]`
    - `schedule_frac: float`
    - `ns_steps: int`
    - `eps: float`
    - `max_update_norm: float`
  - methods:
    - `validate(self) -> KConfig`
    - `to_dict(self) -> Dict[str, Any]`
    - `from_dict(data: Dict[str, Any]) -> KConfig`

- `class TrialConfig:`
  - fields:
    - `study_id: str`
    - `fidelity: Literal["F0", "F1", "F2", "F3"]`
    - `seed: int`
    - `target_acc: float`
    - `script: str`
    - `max_overhead_pct: float`
  - methods:
    - `validate(self) -> TrialConfig`
    - `to_dict(self) -> Dict[str, Any]`

- `class SearchConfig:`
  - fields:
    - `study_id: str`
    - `trials_root: str`
    - `script94: str`
    - `script95: str`
    - `script96: str`
    - `target_acc: float`
    - `max_overhead_pct: float`
    - `delta_min: float`
    - `delta_max: float`
    - `ns_step_min: int`
    - `ns_step_max: int`
    - `f1_budget: int`
    - `f2_budget: int`
    - `f3_budget: int`
    - `seeds_f1: int`
    - `seeds_f2: int`
    - `seeds_f3: int`
    - `search_seed: int`
    - `promote_epsilon: float`
    - `python_exe: str`
  - methods:
    - `validate(self) -> SearchConfig`
    - `to_dict(self) -> Dict[str, Any]`
    - `from_dict(data: Dict[str, Any]) -> SearchConfig`

### `research/lionk/kernels.py`

- `zeropower_via_newtonschulz5(G, steps=3, eps=1e-7)`
- `_apply_shape_preserving(matrix: torch.Tensor, fn)`
- `_inverse_sqrt_newton_schulz_spd(mat: torch.Tensor, steps: int, eps: float) -> torch.Tensor`
- `soft_huber_spectral_update(G: torch.Tensor, delta: float, steps: int = 3, eps: float = 1e-7) -> torch.Tensor`

### `research/lionk/optimizer.py`

- `class MuonK(torch.optim.Optimizer):`
  - `__init__(self, params, lr=1e-3, momentum=0.0, nesterov=False, k_config: Optional[KConfig]=None)`
  - `_delta_for_step(self, group_step: int, group: dict) -> float`
  - `_compute_update(self, g2d: torch.Tensor, delta: float) -> torch.Tensor`
  - `step(self)`

### `research/lionk/metrics.py`

- `first_time_to_target(points, target_acc, time_key="time_seconds", acc_key="tta_val_acc") -> Optional[float]`
- `safe_median(values) -> Optional[float]`
- `objective_from_trial(time_to_target, overhead_pct, max_overhead_pct, unstable, lambda_overhead=0.2, gamma_instability=1000.0) -> float`
- `summarize_seed_metrics(seed_metrics) -> Dict[str, Optional[float]]`

### `research/lionk/io.py`

- `ensure_dir(path: Path) -> None`
- `atomic_write_json(path: Path, obj: Dict[str, Any]) -> None`
- `read_json(path: Path, default=None) -> Dict[str, Any]`
- `append_jsonl(path: Path, row: Dict[str, Any]) -> None`
- `read_jsonl(path: Path) -> List[Dict[str, Any]]`
- `make_study_layout(trials_root: Path, study_id: str) -> Dict[str, Path]`
- `trial_artifact_paths(study_root: Path, trial_id: str) -> Dict[str, Path]`
- `write_summary(path: Path, lines: Iterable[str]) -> None`

### `research/lionk/search.py`

- `_candidate_signature(candidate) -> str`
- `_build_trial_id(fidelity, index, candidate) -> str`
- `_sobol_warmstart(cfg, count, seed) -> List[Dict[str, Any]]`
- `_local_gaussian_proposals(cfg, best, count, rng) -> List[Dict[str, Any]]`
- `_kernel_time_ms(mode, delta, ns_steps, eps, device) -> float`
- `_f0_overhead(candidate) -> float`
- `_run_subprocess(cmd, stdout_path) -> Tuple[int, str]`
- `_script_metrics(python_exe, script, fidelity, seed, candidate, target_acc, metrics_path, stdout_path) -> Dict[str, Any]`
- `_evaluate_fidelity(cfg, candidate, fidelity, seed_count, base_seed, script, artifact_dir) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]`
- `run_search(study_cfg) -> Dict[str, Any]`

### Script entrypoints

- `research/airbench94_muon_simple.py`
  - `_fidelity_epochs_and_tta(fidelity: str) -> tuple[float, int]`
  - `_run_once(args, run_idx) -> Dict[str, float | None]`
  - `_build_arg_parser() -> argparse.ArgumentParser`
  - `main() -> None`

- `research/airbench95_muonk_transfer.py`
  - `_fidelity_epochs_and_tta(base_epochs, base_tta, fidelity)`
  - `_run_once(args, run_idx)`
  - `_build_arg_parser()`
  - `main()`

- `research/airbench96_muonk_transfer.py`
  - `_fidelity_epochs_and_tta(base_epochs, base_tta, fidelity)`
  - `_run_once(args, run_idx)`
  - `_build_arg_parser()`
  - `main()`

- `research/run_muonk_search.py`
  - `_default_study_id(prefix: str) -> str`
  - `_build_parser() -> argparse.ArgumentParser`
  - `main() -> None`

---

## Exact Kernel Formulas

### Muon kernel
- Function: `zeropower_via_newtonschulz5`
- Uses quintic Newton-Schulz coefficients:
  - `a = 3.4445`
  - `b = -4.7750`
  - `c = 2.0315`
- Iteration:
  - `A = X @ X.T`
  - `B = b * A + c * A @ A`
  - `X = a * X + B @ X`
- Input normalization: `X /= (||X|| + eps)`
- For tall matrices, computation is done on transposed view and transposed back.

### Soft-Huber kernel
- Spectral transfer target:
  - `h_delta(s) = s / sqrt(s^2 + delta^2)`
- Matrix form:
  - `update = (G G^T + delta^2 I)^(-1/2) G`
- Inverse square-root approximation:
  - normalize SPD matrix `M` by `scale = trace(M)/n`
  - set `Y0 = M/scale`, `Z0 = I`
  - iterate:
    - `T = 0.5 * (3I - Z @ Y)`
    - `Y <- Y @ T`
    - `Z <- T @ Z`
  - return `Z / sqrt(scale)` as `M^{-1/2}` approximation

---

## Tensor Shapes, Dtypes, and Mutations

### `zeropower_via_newtonschulz5`
- Input: `G` is rank-2 tensor `[m, n]`, usually gradient reshape from `[out_dim, ...]` to `[out_dim, -1]`.
- Internal dtype cast: `bfloat16`.
- Output shape: `[m, n]`.

### `soft_huber_spectral_update`
- Input: `G` rank-2 `[m, n]`.
- Internal dtype for cov/iteration: `float32`.
- Output cast back to input dtype.
- Output shape unchanged.

### `MuonK.step` mutations
- In-place mutates optimizer state and parameters.
- In-place state keys per parameter:
  - `state[p]["momentum_buffer"]: Tensor same shape/dtype as grad`
- In-place on parameter tensor:
  - `p.data.mul_(...)` for re-normalization
  - `p.data.add_(update, alpha=-lr)` for update step
- In-place on momentum buffer:
  - `buf.mul_(momentum).add_(g)`
- In-place on update clipping (if enabled):
  - `update.mul_(clip_factor)`

---

## Exact `delta` Scheduling Formulas

For `k_config.schedule`:

- `static`:
  - `delta_t = delta`

- `linear`:
  - `horizon = max(1, int(total_steps * schedule_frac))`
  - `progress = clamp(step / horizon, 0, 1)`
  - `delta_t = delta + progress * (delta_final - delta)`

- `cosine`:
  - `horizon = max(1, int(total_steps * schedule_frac))`
  - `progress = clamp(step / horizon, 0, 1)`
  - `cosine = 0.5 * (1 - cos(pi * progress))`
  - `delta_t = delta + cosine * (delta_final - delta)`

Final clamp:
- `delta_t = clamp(delta_t, min(delta, delta_final), max(delta, delta_final))`

---

## Step-by-step for one `MuonK.step()` call

1. Iterate each optimizer param group.
2. Read `group["lr"]`, `group["momentum"]`, `group["step"]`.
3. Compute `delta_t` using `_delta_for_step(...)`.
4. For each parameter tensor `p`:
5. Read gradient `g = p.grad`; skip if `None`.
6. If `g` contains non-finite values, skip this parameter.
7. Initialize `state[p]["momentum_buffer"]` if missing.
8. Momentum update in place: `buf = momentum*buf + g`.
9. If Nesterov: `g_eff = g + momentum*buf`; else `g_eff = buf`.
10. Compute `p_norm = ||p||`; if finite and positive, apply re-normalization:
    - `p <- p * (sqrt(numel(p))/p_norm)`
11. Reshape effective grad to 2D: `g2d = g_eff.reshape(len(g_eff), -1)`.
12. Dispatch update kernel:
    - Muon mode: `update2d = zeropower_via_newtonschulz5(g2d, steps, eps)`
    - Soft-Huber mode: `update2d = soft_huber_spectral_update(g2d, delta_t, steps, eps)`
13. Reshape update back to parameter shape.
14. Optional clipping if `max_update_norm > 0`.
15. If update is finite, apply parameter update:
    - `p <- p - lr * update`
16. After parameter loop, increment `group["step"] += 1`.

---

## Search Procedure and Pruning Rules

Search phases in `run_search`:
- F0 overhead microbenchmark
- F1 short 94-run screening
- F2 promoted 94-run evaluation
- F3 finalist 94-run evaluation + transfer checks (95 and 96)

### Candidate generation
- Warmstart with Sobol sequence across:
  - `log(delta)` in `[log(delta_min), log(delta_max)]`
  - `ns_steps` in `[ns_step_min, ns_step_max]`
- Additional local Gaussian proposals around best F1 point.

### Hard pruning
- Candidate is immediately pruned when:
  - `overhead_pct > max_overhead_pct` from F0

### Promotion rules
- F1 -> F2:
  - keep status `ok`
  - rank by objective ascending
  - promote top rows until `f2_budget`
  - if there are fewer than `f2_budget` promoted rows, generate local Gaussian proposals around the best F1 candidate and evaluate additional F1 rows to fill remaining slots
- F2 -> F3:
  - sort F2 `ok` by objective and take top `f3_budget`

### Objective function
`objective = base + overhead_penalty + instability_penalty`
- `base = time_to_target` if available else `1e9`
- `overhead_penalty = lambda_overhead * max(0, overhead_pct - max_overhead_pct)^2`
- `instability_penalty = gamma_instability` if unstable else `0`

Constants in code:
- `lambda_overhead = 0.2`
- `gamma_instability = 1000.0`

---

## JSONL Schema Semantics

`trials.jsonl` rows include:
- `trial_id`: deterministic id from fidelity/index/hash(params)
- `fidelity`: `F1`, `F2`, or `F3`
- `seed`: study-level seed marker
- `params`: candidate parameter dict
- `overhead_pct`: F0 measured overhead vs Muon
- `time_to_target`: median time-to-threshold across evaluated seeds (or `null`)
- `tta_val_acc`: median TTA accuracy across evaluated seeds (or `null`)
- `status`: `ok`, `unstable`, or `pruned`
- `reason`: explicit reason string
- `objective`: scalar ranking value (for evaluated rows)
- `seed_rows`: per-seed metrics snapshot
- `transfer95` / `transfer96` (F3 only): transfer status and metrics

`promotions.jsonl` rows include:
- `from`: source fidelity
- `to`: destination fidelity
- `trial_id`
- `objective`

---

## Failure Modes and Guards

### 1) NaN/Inf detection
- In `MuonK.step()`, if gradient is non-finite, that parameter update is skipped.
- Update tensor is also checked for finiteness before `p.data.add_`.

### 2) Overhead cap rejection
- F0 overhead is computed against Muon baseline microbenchmark.
- If `overhead_pct > max_overhead_pct`, trial status is `pruned` and no training subprocess runs.

### 3) Invalid schedules/parameters
- `KConfig.validate()` and `SearchConfig.validate()` enforce mode/schedule bounds and positivity constraints.
- Invalid config raises `ValueError` before training/search proceeds.

### 4) External runner failure
- If subprocess exits non-zero or misses JSON output, trial status becomes `failed`/`unstable` with explicit `reason`.

---

## How to Reproduce Search Bookkeeping (exact file write order)

Given `study_id`, `run_search` writes in this order:

1. Create study directories under:
   - `research/lionk/trials/studies/<study_id>/`
   - `research/lionk/trials/studies/<study_id>/artifacts/`
2. Write `study_config.json` atomically.
3. Ensure empty `trials.jsonl` and `promotions.jsonl` exist.
4. Ensure `summary.md` exists with heading.
5. For each trial:
   - create `artifacts/<trial_id>/`
   - write `params.json`
   - append row to `trials.jsonl`
   - write `metrics.json`
   - write `status.txt`
   - for promoted trials append row to `promotions.jsonl`
6. At end:
   - write `best.json`
   - write `search_state.json`
   - overwrite `summary.md` with final summary lines

---

## Script Contracts Used by Search

Each runner script accepts:
- `--runs`
- `--seed`
- `--fidelity`
- `--target-acc`
- `--matrix-opt`
- `--k-delta`
- `--k-delta-final`
- `--k-schedule`
- `--k-schedule-frac`
- `--k-ns-steps`
- `--k-eps`
- `--k-max-update-norm`
- `--json-output`
- `--quiet`

Each runner writes JSON with at least:
- `mean_tta_val_acc`
- `time_to_target`

These fields are consumed by `search.py` in `_script_metrics`.
