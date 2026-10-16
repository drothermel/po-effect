from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest
from conftest import CPU_CONDITIONS, readout_spec

from po_effect.measurement import (
    AttemptRecord,
    AttemptTiming,
    DecisionRule,
    MeasurementSpec,
    ReadoutRecord,
    ReadoutSpec,
    apply_readout,
    run_measurement,
)
from po_effect.records import export_wide, open_record_store
from po_effect.records.store import (
    RUN_CONDITIONS_SCHEMA,
    STORE_FILE,
    attempt_key,
    conditions_key,
    evidence_key,
    readout_record_key,
    readout_spec_key,
    spec_key,
    timing_key,
)

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import ExampleMeasurement

    from po_effect.identity import Identified


def test_binding_keys() -> None:
    m, r, i, e = "1" * 64, "2" * 64, "3" * 64, "4" * 64
    assert spec_key(m) == f"measurement/{m}/spec"
    assert conditions_key(m) == f"measurement/{m}/conditions"
    assert attempt_key(m, i) == f"measurement/{m}/attempt/{i}"
    assert timing_key(m, i) == f"measurement/{m}/timing/{i}"
    assert readout_spec_key(m, r) == f"measurement/{m}/readout/{r}/spec"
    assert readout_record_key(m, r, i) == f"measurement/{m}/readout/{r}/{i}"
    assert evidence_key(e) == f"evidence/{e}"
    assert RUN_CONDITIONS_SCHEMA == "po_effect.run_conditions"
    assert STORE_FILE == "store.sqlite3"


def test_store_file_under_records_root(tmp_path: Path) -> None:
    root = tmp_path / "records"
    with open_record_store(root) as store:
        assert store.path == root / STORE_FILE
    assert (root / STORE_FILE).is_file()


def assert_bound(
    example: ExampleMeasurement, key: str, record: Identified
) -> None:
    objects = example.store.objects
    reference = objects.resolve(key)
    assert reference is not None
    assert reference.schema == record.identity_schema
    assert reference.content_hash == record.identity_hash
    assert objects.get(reference) == record.identity_document().to_json_dict()


def test_put_bind_round_trip_keeps_identity(
    example: ExampleMeasurement,
) -> None:
    result = run_measurement(example.spec, example.runtime)
    store = example.store
    measurement_hash = example.spec.identity_hash
    spec = readout_spec(example, DecisionRule.PMI)
    records = apply_readout(spec, store, example.runtime.items)
    assert_bound(example, spec_key(measurement_hash), example.spec)
    assert store.measurement(measurement_hash) == example.spec
    assert store.run_conditions(measurement_hash) == CPU_CONDITIONS
    reference = store.objects.resolve(conditions_key(measurement_hash))
    assert reference is not None
    assert reference.schema == RUN_CONDITIONS_SCHEMA
    timings = store.timings(measurement_hash)
    for record in result.attempts:
        assert_bound(
            example, attempt_key(measurement_hash, record.item_id), record
        )
        timing = timings[record.item_id]
        assert_bound(
            example, timing_key(measurement_hash, record.item_id), timing
        )
        assert timing.wall_time_seconds >= 0
        assert "wall_time_seconds" not in record.model_dump()
        for ref in record.attempt.evidence:
            evidence = store.objects.resolve(evidence_key(ref.evidence_hash))
            assert evidence is not None
            assert evidence.content_hash == ref.evidence_hash
    assert store.attempts(measurement_hash) == {
        r.item_id: r for r in result.attempts
    }
    assert_bound(
        example,
        readout_spec_key(measurement_hash, spec.identity_hash),
        spec,
    )
    assert store.readout(measurement_hash, spec.identity_hash) == spec
    for record in records:
        assert_bound(
            example,
            readout_record_key(
                measurement_hash, spec.identity_hash, record.item_id
            ),
            record,
        )
    assert store.readout_records(spec) == {r.item_id: r for r in records}


def test_records_persist_across_sessions(
    example: ExampleMeasurement, tmp_path: Path
) -> None:
    result = run_measurement(example.spec, example.runtime)
    with open_record_store(tmp_path / "records") as reopened:
        assert reopened.attempts(example.spec.identity_hash) == {
            r.item_id: r for r in result.attempts
        }


def test_readout_records_are_kept_per_spec(
    example: ExampleMeasurement,
) -> None:
    run_measurement(example.spec, example.runtime)
    store = example.store
    raw = readout_spec(example, DecisionRule.RAW)
    pmi = readout_spec(example, DecisionRule.PMI)
    apply_readout(raw, store, example.runtime.items)
    assert store.readout_records(raw).keys() == set(
        example.spec.item_set.item_ids
    )
    with pytest.raises(KeyError, match="nothing is bound"):
        store.readout_records(pmi)
    apply_readout(pmi, store, example.runtime.items)
    raw_records = store.readout_records(raw)
    pmi_records = store.readout_records(pmi)
    assert {r.readout_hash for r in raw_records.values()} == {
        raw.identity_hash
    }
    assert {r.readout_hash for r in pmi_records.values()} == {
        pmi.identity_hash
    }
    first, _ = example.spec.item_set.item_ids
    assert raw_records[first].candidate != pmi_records[first].candidate


def test_export_wide(example: ExampleMeasurement, tmp_path: Path) -> None:
    run_measurement(example.spec, example.runtime)
    store = example.store
    applied = []
    for rule in DecisionRule:
        spec = readout_spec(example, rule)
        apply_readout(spec, store, example.runtime.items)
        applied.append(spec.identity_hash)
    path = tmp_path / "export.parquet"
    export_wide(
        store,
        example.spec.identity_hash,
        applied,
        example.runtime.items,
        path,
    )
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
    assert table["reasoning_visibility"] == ["none", "none"]
    assert len(table) == 9 + 3 * len(DecisionRule)


def test_bindings_must_hold_the_expected_schema(
    example: ExampleMeasurement,
) -> None:
    result = run_measurement(example.spec, example.runtime)
    store = example.store
    first, _ = result.attempts
    other = "0" * 64
    reference = store.objects.resolve(
        attempt_key(example.spec.identity_hash, first.item_id)
    )
    assert reference is not None
    store.objects.bind(spec_key(other), reference)
    with pytest.raises(ValueError, match=r"holds a po_effect\.attempt_record"):
        store.measurement(other)
    with pytest.raises(KeyError, match="nothing is bound"):
        store.measurement("9" * 64)


def test_timing_must_match_its_attempt(example: ExampleMeasurement) -> None:
    first, second = example.spec.item_set.item_ids
    measurement_hash = example.spec.identity_hash
    record = AttemptRecord.model_validate(
        {
            "measurement_hash": measurement_hash,
            "item_id": first,
            "attempt": {
                "kind": "score",
                "candidates": [" a"],
                "scores": [
                    {
                        "log_likelihood": -1.0,
                        "token_count": 1,
                        "char_count": 1,
                        "byte_count": 1,
                        "unconditional_log_likelihood": None,
                    }
                ],
                "evidence": [],
                "chat_template": None,
            },
        }
    )
    timing = AttemptTiming(
        measurement_hash=measurement_hash,
        item_id=second,
        wall_time_seconds=0.1,
    )
    with pytest.raises(ValueError, match="another attempt"):
        example.store.put_attempt(record, timing)
    assert not example.store.has_attempt(measurement_hash, first)


def test_schema_names() -> None:
    assert MeasurementSpec.identity_schema == "po_effect.measurement_spec"
    assert AttemptRecord.identity_schema_version == 5
    assert AttemptTiming.identity_schema == "po_effect.attempt_timing"
    assert ReadoutSpec.identity_schema == "po_effect.readout_spec"
    assert ReadoutRecord.identity_schema == "po_effect.readout_record"
