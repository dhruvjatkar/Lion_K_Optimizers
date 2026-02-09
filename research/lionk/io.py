from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        json.dump(obj, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def read_json(path: Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    with path.open("r") as f:
        return json.load(f)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Append a single JSON line.  Uses fsync to survive crashes/power loss."""
    ensure_dir(path.parent)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def make_study_layout(trials_root: Path, study_id: str) -> Dict[str, Path]:
    study_root = trials_root / "studies" / study_id
    artifacts_root = study_root / "artifacts"

    ensure_dir(study_root)
    ensure_dir(artifacts_root)

    paths = {
        "study_root": study_root,
        "artifacts_root": artifacts_root,
        "study_config": study_root / "study_config.json",
        "trials": study_root / "trials.jsonl",
        "promotions": study_root / "promotions.jsonl",
        "best": study_root / "best.json",
        "summary": study_root / "summary.md",
        "search_state": study_root / "search_state.json",
    }

    for key in ("trials", "promotions"):
        if not paths[key].exists():
            paths[key].touch()

    if not paths["summary"].exists():
        paths["summary"].write_text("# MuonK Search Summary\n\n")

    return paths


def trial_artifact_paths(study_root: Path, trial_id: str) -> Dict[str, Path]:
    artifact_dir = study_root / "artifacts" / trial_id
    ensure_dir(artifact_dir)
    return {
        "dir": artifact_dir,
        "params": artifact_dir / "params.json",
        "metrics": artifact_dir / "metrics.json",
        "stdout": artifact_dir / "stdout.log",
        "status": artifact_dir / "status.txt",
    }


def write_summary(path: Path, lines: Iterable[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")
