from __future__ import annotations

from pathlib import Path

from research.lionk.io import append_jsonl, make_study_layout, read_jsonl


def test_make_study_layout_and_jsonl_roundtrip(tmp_path: Path):
    layout = make_study_layout(tmp_path, "test_study")
    assert layout["study_root"].exists()
    assert layout["trials"].exists()

    row = {
        "trial_id": "t0",
        "fidelity": "F1",
        "seed": 0,
        "params": {"delta": 0.1},
        "overhead_pct": 0.0,
        "time_to_target": 1.0,
        "tta_val_acc": 0.94,
        "status": "ok",
        "reason": "",
    }
    append_jsonl(layout["trials"], row)
    rows = read_jsonl(layout["trials"])
    assert len(rows) == 1
    assert rows[0]["trial_id"] == "t0"


def test_append_jsonl_multiple_rows(tmp_path: Path):
    path = tmp_path / "test.jsonl"
    for i in range(5):
        append_jsonl(path, {"idx": i})
    rows = read_jsonl(path)
    assert len(rows) == 5
    assert [r["idx"] for r in rows] == [0, 1, 2, 3, 4]


def test_append_jsonl_survives_reopen(tmp_path: Path):
    """Verify that fsync'd data is readable after reopening."""
    path = tmp_path / "fsync_test.jsonl"
    append_jsonl(path, {"key": "value1"})
    # Re-read immediately
    rows = read_jsonl(path)
    assert len(rows) == 1
    assert rows[0]["key"] == "value1"
    # Append again
    append_jsonl(path, {"key": "value2"})
    rows = read_jsonl(path)
    assert len(rows) == 2
