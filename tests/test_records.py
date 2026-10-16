from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest
from conftest import CPU_CONDITIONS, readout_spec

from po_effect.measurement import DecisionRule, apply_readout, run_measurement
from po_effect.records import MeasurementDocument, export_wide

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import ExampleMeasurement


def test_store_layout(example: ExampleMeasurement) -> None:
    result = run_measurement(example.spec, example.runtime)
    spec = readout_spec(example, DecisionRule.PMI)
    apply_readout(spec, example.runtime.store, example.runtime.items)
    root = example.root
    item_files = {f"{i}.json" for i in example.spec.item_set.item_ids}
    readout = f"readouts/{spec.identity_hash}"
    assert {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    } == {
        "spec.json",
        "evidence.jsonl",
        f"{readout}/spec.json",
        *(f"attempts/{name}" for name in item_files),
        *(f"{readout}/{name}" for name in item_files),
    }
    store = example.runtime.store
    assert store.measurement(example.spec.identity_hash) == example.spec
    assert tuple(store.attempts(example.spec.identity_hash).values()) == (
        tuple(sorted(result.attempts, key=lambda r: r.item_id))
    )


def test_export_wide(example: ExampleMeasurement, tmp_path: Path) -> None:
    run_measurement(example.spec, example.runtime)
    store = example.runtime.store
    for rule in DecisionRule:
        apply_readout(
            readout_spec(example, rule), store, example.runtime.items
        )
    path = tmp_path / "export.parquet"
    export_wide(store, example.spec.identity_hash, example.runtime.items, path)
    table = pq.read_table(path).to_pydict()
    assert table["item_id"] == list(example.spec.item_set.item_ids)
    assert table["native_id"] == ["x", "y"]
    assert table["gold_index"] == [1, 0]
    assert table["log_likelihood"] == [[-2, -3], [-3, -8]]
    assert table["unconditional_log_likelihood"] == [[-3, -8], [-3, -8]]
    assert table["token_count"] == [[2, 2], [2, 2]]
    assert table["char_count"] == [[2, 3], [2, 2]]
    assert table["byte_count"] == [[2, 5], [2, 2]]
    assert table["pmi_index_match_predicted_index"] == [1, 0]
    assert table["pmi_index_match_correct"] == [True, True]
    assert table["raw_index_match_correct"] == [False, True]
    assert table["raw_index_match_abstained"] == [False, False]
    assert len(table) == 8 + 3 * len(DecisionRule)


def test_stored_records_must_match_their_paths(
    example: ExampleMeasurement,
) -> None:
    result = run_measurement(example.spec, example.runtime)
    store = example.runtime.store
    root = example.root
    first, second = result.attempts
    (root / "attempts" / f"{first.item_id}.json").write_text(
        second.model_dump_json()
    )
    with pytest.raises(ValueError, match="item id"):
        store.attempts(example.spec.identity_hash)
    (root / "spec.json").write_text(
        MeasurementDocument(
            spec=example.spec.model_copy(update={"solver_id": "other"}),
            run_conditions=CPU_CONDITIONS,
        ).model_dump_json()
    )
    with pytest.raises(ValueError, match="differs from its hash"):
        store.measurement(example.spec.identity_hash)
