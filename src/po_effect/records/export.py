from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from po_effect.measurement import (
    GenerationAttempt,
    IndexCandidate,
    ScoredCandidatesAttempt,
)

if TYPE_CHECKING:
    from pathlib import Path

    from po_effect.measurement import RecordStore
    from po_effect.tasks import ItemIndex

SCORE_FIELDS = (
    ("log_likelihood", pa.float64()),
    ("token_count", pa.int64()),
    ("char_count", pa.int64()),
    ("byte_count", pa.int64()),
    ("unconditional_log_likelihood", pa.float64()),
)
GENERATION_FIELDS = (
    ("text", pa.string()),
    ("stop_reason", pa.string()),
    ("prompt_tokens", pa.int64()),
    ("completion_tokens", pa.int64()),
)


def export_wide(
    store: RecordStore,
    measurement_hash: str,
    items: ItemIndex,
    path: Path,
) -> None:
    spec = store.measurement(measurement_hash)
    attempts = store.attempts(measurement_hash)
    item_ids = spec.item_set.item_ids
    missing = [i for i in item_ids if i not in attempts]
    if missing:
        raise ValueError(f"{len(missing)} items have no stored attempt")
    rows = [items.get(i) for i in item_ids]
    columns: dict[str, pa.Array] = {
        "item_id": pa.array(item_ids, pa.string()),
        "native_id": pa.array([i.native_id for i in rows], pa.string()),
        "gold_index": pa.array(
            [i.reference.gold_index for i in rows], pa.int64()
        ),
    }
    selected_attempts = [attempts[i].attempt for i in item_ids]
    scored = [
        a for a in selected_attempts if isinstance(a, ScoredCandidatesAttempt)
    ]
    generated = [
        a for a in selected_attempts if isinstance(a, GenerationAttempt)
    ]
    if len(scored) == len(item_ids):
        for name, kind in SCORE_FIELDS:
            columns[name] = pa.array(
                [[getattr(score, name) for score in a.scores] for a in scored],
                pa.list_(kind),
            )
    elif len(generated) == len(item_ids):
        for name, kind in GENERATION_FIELDS:
            columns[name] = pa.array(
                [getattr(a.output, name) for a in generated], kind
            )
    else:
        raise ValueError("measurement mixes attempt kinds")
    visibility: list[str] | None = None
    for readout in store.readouts(measurement_hash):
        records = store.readout_records(readout)
        prefix = f"{readout.readout_id}_{readout.judge_id}"
        if f"{prefix}_predicted_index" in columns:
            raise ValueError(f"duplicate readout columns {prefix!r}")
        selected = [records[i] for i in item_ids]
        current = [r.reasoning_visibility.value for r in selected]
        if visibility is None:
            visibility = current
        elif visibility != current:
            raise ValueError("readouts disagree on reasoning visibility")
        columns[f"{prefix}_predicted_index"] = pa.array(
            [
                r.candidate.index
                if isinstance(r.candidate, IndexCandidate)
                else None
                for r in selected
            ],
            pa.int64(),
        )
        columns[f"{prefix}_correct"] = pa.array(
            [r.scores.accuracy == 1 for r in selected], pa.bool_()
        )
        columns[f"{prefix}_abstained"] = pa.array(
            [r.scores.abstained == 1 for r in selected], pa.bool_()
        )
    if visibility is not None:
        columns["reasoning_visibility"] = pa.array(visibility, pa.string())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
