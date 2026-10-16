from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from dr_serialize import build_identity_document, identity_document_hash
from dr_store.sync import open_sqlite

from po_effect.measurement import (
    AttemptRecord,
    AttemptTiming,
    MeasurementSpec,
    ReadoutRecord,
    ReadoutSpec,
)
from po_effect.models import RunConditions

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from dr_serialize import IdentityDocument
    from dr_store.sync import BlockingObjectStore

    from po_effect.identity import Identified

STORE_FILE = "store.sqlite3"
RUN_CONDITIONS_SCHEMA = "po_effect.run_conditions"
RUN_CONDITIONS_SCHEMA_VERSION = 1


def measurement_key(measurement_hash: str) -> str:
    return f"measurement/{measurement_hash}"


def spec_key(measurement_hash: str) -> str:
    return f"{measurement_key(measurement_hash)}/spec"


def conditions_key(measurement_hash: str) -> str:
    return f"{measurement_key(measurement_hash)}/conditions"


def attempt_key(measurement_hash: str, item_id: str) -> str:
    return f"{measurement_key(measurement_hash)}/attempt/{item_id}"


def timing_key(measurement_hash: str, item_id: str) -> str:
    return f"{measurement_key(measurement_hash)}/timing/{item_id}"


def readout_key(measurement_hash: str, readout_hash: str) -> str:
    return f"{measurement_key(measurement_hash)}/readout/{readout_hash}"


def readout_spec_key(measurement_hash: str, readout_hash: str) -> str:
    return f"{readout_key(measurement_hash, readout_hash)}/spec"


def readout_record_key(
    measurement_hash: str, readout_hash: str, item_id: str
) -> str:
    return f"{readout_key(measurement_hash, readout_hash)}/{item_id}"


def evidence_key(evidence_hash: str) -> str:
    return f"evidence/{evidence_hash}"


def conditions_document(conditions: RunConditions) -> IdentityDocument:
    return build_identity_document(
        schema=RUN_CONDITIONS_SCHEMA,
        schema_version=RUN_CONDITIONS_SCHEMA_VERSION,
        payload=conditions.model_dump(mode="json"),
    )


class DrStoreRecordStore:
    def __init__(self, objects: BlockingObjectStore, path: Path) -> None:
        self.objects = objects
        self.path = path

    def _put(self, document: IdentityDocument, key: str) -> str:
        identity_hash = str(identity_document_hash(document))
        reference, _ = self.objects.put(
            document.schema, document.to_json_dict()
        )
        if reference.content_hash != identity_hash:
            raise RuntimeError(
                f"stored {document.schema} hash differs from its identity"
            )
        self.objects.bind(key, reference)
        return identity_hash

    def _put_record(self, record: Identified, key: str) -> str:
        return self._put(record.identity_document(), key)

    def _get[T: Identified](self, key: str, kind: type[T]) -> T | None:
        reference = self.objects.resolve(key)
        if reference is None:
            return None
        if reference.schema != kind.identity_schema:
            raise ValueError(f"binding {key!r} holds a {reference.schema}")
        document = self.objects.get(reference)
        if not isinstance(document, dict) or (
            document.get("schema_version") != kind.identity_schema_version
        ):
            raise ValueError(
                f"binding {key!r} holds another {kind.identity_schema} version"
            )
        record = kind.model_validate(document["payload"])
        if record.identity_hash != reference.content_hash:
            raise ValueError(f"record at {key!r} differs from its hash")
        return record

    def _require[T: Identified](self, key: str, kind: type[T]) -> T:
        record = self._get(key, kind)
        if record is None:
            raise KeyError(f"nothing is bound at {key!r}")
        return record

    def run_conditions(self, measurement_hash: str) -> RunConditions | None:
        reference = self.objects.resolve(conditions_key(measurement_hash))
        if reference is None:
            return None
        document = self.objects.get(reference)
        if not isinstance(document, dict) or (
            reference.schema,
            document.get("schema_version"),
        ) != (RUN_CONDITIONS_SCHEMA, RUN_CONDITIONS_SCHEMA_VERSION):
            raise ValueError("stored run conditions have another schema")
        return RunConditions.model_validate(document["payload"])

    def put_measurement(
        self, spec: MeasurementSpec, conditions: RunConditions
    ) -> str:
        measurement_hash = spec.identity_hash
        recorded = self.run_conditions(measurement_hash)
        if recorded is None:
            self._put_record(spec, spec_key(measurement_hash))
            self._put(
                conditions_document(conditions),
                conditions_key(measurement_hash),
            )
            return measurement_hash
        for field in RunConditions.model_fields:
            stored = getattr(recorded, field)
            current = getattr(conditions, field)
            if stored != current:
                raise ValueError(
                    f"run condition {field!r} is {current!r} but the "
                    f"measurement was recorded with {stored!r}"
                )
        self._put_record(spec, spec_key(measurement_hash))
        return measurement_hash

    def measurement(self, measurement_hash: str) -> MeasurementSpec:
        spec = self._require(spec_key(measurement_hash), MeasurementSpec)
        if spec.identity_hash != measurement_hash:
            raise ValueError("stored measurement spec differs from its hash")
        return spec

    def has_attempt(self, measurement_hash: str, item_id: str) -> bool:
        key = attempt_key(measurement_hash, item_id)
        return self.objects.resolve(key) is not None

    def put_attempt(self, record: AttemptRecord, timing: AttemptTiming) -> str:
        if (timing.measurement_hash, timing.item_id) != (
            record.measurement_hash,
            record.item_id,
        ):
            raise ValueError("timing belongs to another attempt")
        attempt_hash = self._put_record(
            record, attempt_key(record.measurement_hash, record.item_id)
        )
        self._put_record(
            timing, timing_key(timing.measurement_hash, timing.item_id)
        )
        return attempt_hash

    def attempts(self, measurement_hash: str) -> dict[str, AttemptRecord]:
        records = {}
        for item_id in self.measurement(measurement_hash).item_set.item_ids:
            record = self._get(
                attempt_key(measurement_hash, item_id), AttemptRecord
            )
            if record is None:
                continue
            if (record.measurement_hash, record.item_id) != (
                measurement_hash,
                item_id,
            ):
                raise ValueError("attempt record differs from its binding")
            records[item_id] = record
        return records

    def timings(self, measurement_hash: str) -> dict[str, AttemptTiming]:
        timings = {}
        for item_id in self.measurement(measurement_hash).item_set.item_ids:
            timing = self._get(
                timing_key(measurement_hash, item_id), AttemptTiming
            )
            if timing is not None:
                timings[item_id] = timing
        return timings

    def put_evidence(self, document: IdentityDocument) -> str:
        evidence_hash = str(identity_document_hash(document))
        return self._put(document, evidence_key(evidence_hash))

    def put_readout(self, spec: ReadoutSpec) -> str:
        return self._put_record(
            spec,
            readout_spec_key(spec.measurement_hash, spec.identity_hash),
        )

    def readout(self, measurement_hash: str, readout_hash: str) -> ReadoutSpec:
        spec = self._require(
            readout_spec_key(measurement_hash, readout_hash), ReadoutSpec
        )
        if (spec.measurement_hash, spec.identity_hash) != (
            measurement_hash,
            readout_hash,
        ):
            raise ValueError("readout spec differs from its binding")
        return spec

    def put_readout_record(self, record: ReadoutRecord) -> str:
        return self._put_record(
            record,
            readout_record_key(
                record.measurement_hash, record.readout_hash, record.item_id
            ),
        )

    def readout_records(self, spec: ReadoutSpec) -> dict[str, ReadoutRecord]:
        measurement_hash = spec.measurement_hash
        readout_hash = self.readout(
            measurement_hash, spec.identity_hash
        ).identity_hash
        records = {}
        for item_id in self.measurement(measurement_hash).item_set.item_ids:
            record = self._get(
                readout_record_key(measurement_hash, readout_hash, item_id),
                ReadoutRecord,
            )
            if record is None:
                continue
            if (record.readout_hash, record.item_id) != (
                readout_hash,
                item_id,
            ):
                raise ValueError("readout record differs from its binding")
            records[item_id] = record
        return records


@contextmanager
def open_record_store(root: Path) -> Generator[DrStoreRecordStore]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / STORE_FILE
    with open_sqlite(str(path)) as objects:
        yield DrStoreRecordStore(objects, path)
