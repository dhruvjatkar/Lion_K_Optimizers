# MuonK Trial Storage

This directory stores search studies and trial artifacts.

## Layout

- `studies/<study_id>/study_config.json`: immutable search configuration.
- `studies/<study_id>/trials.jsonl`: append-only trial rows.
- `studies/<study_id>/promotions.jsonl`: append-only promotion log.
- `studies/<study_id>/best.json`: best candidate snapshot.
- `studies/<study_id>/summary.md`: human-readable summary.
- `studies/<study_id>/search_state.json`: resumable search state.
- `studies/<study_id>/artifacts/<trial_id>/params.json`: trial parameter payload.
- `studies/<study_id>/artifacts/<trial_id>/metrics.json`: trial aggregate metrics.
- `studies/<study_id>/artifacts/<trial_id>/stdout.log`: command output capture.
- `studies/<study_id>/artifacts/<trial_id>/status.txt`: one-line trial status marker.

Archived studies can be moved into `archive/`.
