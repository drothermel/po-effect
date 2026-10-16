from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from dr_serialize import Sha256Digest

from po_effect.identity import FrozenContract
from po_effect.measurement import (
    AttemptRecord,
    MeasurementSpec,
    ReadoutRecord,
    ReadoutSpec,
)
from po_effect.models import RunConditions

SPEC_FILE = "spec.json"
ATTEMPTS_DIR = "attempts"
READOUTS_DIR = "readouts"
EVIDENCE_FILE = "evidence.jsonl"


class MeasurementDocument(FrozenContract):
    spec: MeasurementSpec
    run_conditions: RunConditions


def _digest(value: str) -> str:
    return str(Sha256Digest.parse(value))


def _write(path: Path, record: FrozenContract) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8"
    ) as stream:
        stream.write(record.model_dump_json() + "\n")
    Path(stream.name).replace(path)


class FileRecordStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def _measurement_dir(self, measurement_hash: str) -> Path:
        return self.root / _digest(measurement_hash)

    def _attempt_path(self, measurement_hash: str, item_id: str) -> Path:
        return (
            self._measurement_dir(measurement_hash)
            / ATTEMPTS_DIR
            / f"{_digest(item_id)}.json"
        )

    def _readout_dir(self, measurement_hash: str, readout_hash: str) -> Path:
        return (
            self._measurement_dir(measurement_hash)
            / READOUTS_DIR
            / _digest(readout_hash)
        )

    def _measurement_document(
        self, measurement_hash: str
    ) -> MeasurementDocument:
        path = self._measurement_dir(measurement_hash) / SPEC_FILE
        document = MeasurementDocument.model_validate_json(path.read_bytes())
        if document.spec.identity_hash != measurement_hash:
            raise ValueError("stored measurement spec differs from its hash")
        return document

    def put_measurement(
        self, spec: MeasurementSpec, conditions: RunConditions
    ) -> str:
        measurement_hash = spec.identity_hash
        path = self._measurement_dir(measurement_hash) / SPEC_FILE
        if not path.exists():
            _write(
                path,
                MeasurementDocument(spec=spec, run_conditions=conditions),
            )
            return measurement_hash
        recorded = self._measurement_document(measurement_hash).run_conditions
        for field in RunConditions.model_fields:
            stored = getattr(recorded, field)
            current = getattr(conditions, field)
            if stored != current:
                raise ValueError(
                    f"run condition {field!r} is {current!r} but the "
                    f"measurement was recorded with {stored!r}"
                )
        return measurement_hash

    def measurement(self, measurement_hash: str) -> MeasurementSpec:
        return self._measurement_document(measurement_hash).spec

    def has_attempt(self, measurement_hash: str, item_id: str) -> bool:
        return self._attempt_path(measurement_hash, item_id).exists()

    def put_attempt(self, record: AttemptRecord) -> str:
        _write(
            self._attempt_path(record.measurement_hash, record.item_id), record
        )
        return record.identity_hash

    def attempts(self, measurement_hash: str) -> dict[str, AttemptRecord]:
        directory = self._measurement_dir(measurement_hash) / ATTEMPTS_DIR
        records = {}
        for path in sorted(directory.glob("*.json")):
            record = AttemptRecord.model_validate_json(path.read_bytes())
            if record.item_id != path.stem:
                raise ValueError("attempt record differs from its item id")
            if record.measurement_hash != measurement_hash:
                raise ValueError(
                    "attempt record belongs to another measurement"
                )
            records[record.item_id] = record
        return records

    def append_evidence(self, measurement_hash: str, document: str) -> None:
        if "\n" in document:
            raise ValueError("evidence document must be a single line")
        path = self._measurement_dir(measurement_hash) / EVIDENCE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(document + "\n")

    def put_readout(self, spec: ReadoutSpec) -> str:
        readout_hash = spec.identity_hash
        path = (
            self._readout_dir(spec.measurement_hash, readout_hash) / SPEC_FILE
        )
        if not path.exists():
            _write(path, spec)
        return readout_hash

    def readouts(self, measurement_hash: str) -> tuple[ReadoutSpec, ...]:
        directory = self._measurement_dir(measurement_hash) / READOUTS_DIR
        specs = []
        for path in sorted(directory.glob(f"*/{SPEC_FILE}")):
            spec = ReadoutSpec.model_validate_json(path.read_bytes())
            if spec.identity_hash != path.parent.name:
                raise ValueError("stored readout spec differs from its hash")
            if spec.measurement_hash != measurement_hash:
                raise ValueError("readout spec belongs to another measurement")
            specs.append(spec)
        return tuple(specs)

    def put_readout_record(self, record: ReadoutRecord) -> str:
        directory = self._readout_dir(
            record.measurement_hash, record.readout_hash
        )
        _write(directory / f"{_digest(record.item_id)}.json", record)
        return record.identity_hash

    def readout_records(self, spec: ReadoutSpec) -> dict[str, ReadoutRecord]:
        readout_hash = spec.identity_hash
        directory = self._readout_dir(spec.measurement_hash, readout_hash)
        records = {}
        for path in sorted(directory.glob("*.json")):
            if path.name == SPEC_FILE:
                continue
            record = ReadoutRecord.model_validate_json(path.read_bytes())
            if record.item_id != path.stem:
                raise ValueError("readout record differs from its item id")
            if record.readout_hash != readout_hash:
                raise ValueError("readout record belongs to another readout")
            records[record.item_id] = record
        return records
