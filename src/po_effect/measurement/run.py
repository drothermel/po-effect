from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, ClassVar, Protocol

from pydantic import Field, model_validator

from po_effect.identity import FrozenContract, Identified
from po_effect.measurement.attempts import Attempt
from po_effect.measurement.judges import Scores, judge_for
from po_effect.measurement.readouts import (
    Candidate,
    ReadoutId,
    readout_for,
)
from po_effect.measurement.solvers import (
    GenerationSpec,
    SingleCallSpec,
    SolverSpec,
)
from po_effect.models import ModelHandle
from po_effect.tasks import AttemptKind, ItemSet

if TYPE_CHECKING:
    from po_effect.measurement.readouts import Readout
    from po_effect.measurement.solvers import Solver
    from po_effect.models import ModelSession, RunConditions
    from po_effect.tasks import ArcChallengeTask, ItemIndex

MEASUREMENT_SPEC_SCHEMA = "po_effect.measurement_spec"
MEASUREMENT_SPEC_SCHEMA_VERSION = 6
ATTEMPT_RECORD_SCHEMA = "po_effect.attempt_record"
ATTEMPT_RECORD_SCHEMA_VERSION = 2
READOUT_SPEC_SCHEMA = "po_effect.readout_spec"
READOUT_SPEC_SCHEMA_VERSION = 1
READOUT_RECORD_SCHEMA = "po_effect.readout_record"
READOUT_RECORD_SCHEMA_VERSION = 2


class MeasurementSpec(Identified):
    identity_schema: ClassVar[str] = MEASUREMENT_SPEC_SCHEMA
    identity_schema_version: ClassVar[int] = MEASUREMENT_SPEC_SCHEMA_VERSION
    task_id: str
    task_version: int
    item_set: ItemSet
    model: ModelHandle
    solver_id: str
    solver_spec: SolverSpec
    attempt_kind: AttemptKind

    @model_validator(mode="after")
    def validate_prompt_task(self) -> MeasurementSpec:
        prompt = self.solver_spec.prompt
        if (prompt.task_id, prompt.task_version) != (
            self.task_id,
            self.task_version,
        ):
            raise ValueError("prompt was resolved against a different task")
        expected = (
            GenerationSpec
            if self.attempt_kind is AttemptKind.GENERATION
            else SingleCallSpec
        )
        if not isinstance(self.solver_spec, expected):
            raise ValueError(  # noqa: TRY004
                f"{self.attempt_kind.value} measurements require "
                f"{expected.__name__}"
            )
        return self


class AttemptRecord(Identified):
    identity_schema: ClassVar[str] = ATTEMPT_RECORD_SCHEMA
    identity_schema_version: ClassVar[int] = ATTEMPT_RECORD_SCHEMA_VERSION
    measurement_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    item_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt: Attempt
    wall_time_seconds: float = Field(ge=0)


class ReadoutSpec(Identified):
    identity_schema: ClassVar[str] = READOUT_SPEC_SCHEMA
    identity_schema_version: ClassVar[int] = READOUT_SPEC_SCHEMA_VERSION
    measurement_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    readout_id: ReadoutId
    judge_id: str


class ReadoutRecord(Identified):
    identity_schema: ClassVar[str] = READOUT_RECORD_SCHEMA
    identity_schema_version: ClassVar[int] = READOUT_RECORD_SCHEMA_VERSION
    measurement_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    readout_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    item_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate: Candidate
    scores: Scores


class RecordStore(Protocol):
    def put_measurement(
        self, spec: MeasurementSpec, conditions: RunConditions
    ) -> str: ...

    def measurement(self, measurement_hash: str) -> MeasurementSpec: ...

    def has_attempt(self, measurement_hash: str, item_id: str) -> bool: ...

    def put_attempt(self, record: AttemptRecord) -> str: ...

    def attempts(self, measurement_hash: str) -> dict[str, AttemptRecord]: ...

    def append_evidence(
        self, measurement_hash: str, document: str
    ) -> None: ...

    def put_readout(self, spec: ReadoutSpec) -> str: ...

    def readouts(self, measurement_hash: str) -> tuple[ReadoutSpec, ...]: ...

    def put_readout_record(self, record: ReadoutRecord) -> str: ...

    def readout_records(
        self, spec: ReadoutSpec
    ) -> dict[str, ReadoutRecord]: ...


@dataclass(frozen=True, slots=True)
class MeasurementRuntime:
    task: ArcChallengeTask
    items: ItemIndex
    session: ModelSession
    solver: Solver
    store: RecordStore


class MeasurementResult(FrozenContract):
    spec: MeasurementSpec
    attempts: tuple[AttemptRecord, ...]
    measured: int


def run_measurement(
    spec: MeasurementSpec,
    runtime: MeasurementRuntime,
) -> MeasurementResult:
    if (spec.task_id, spec.task_version) != (
        runtime.task.task_id,
        runtime.task.version,
    ) or (spec.item_set.task_id, spec.item_set.task_version) != (
        spec.task_id,
        spec.task_version,
    ):
        raise ValueError("measurement task differs from runtime or item set")
    if spec.model != runtime.session.handle:
        raise ValueError("measurement model differs from session")
    if spec.solver_id != runtime.solver.solver_id:
        raise ValueError("measurement solver differs from runtime")
    if spec.attempt_kind != runtime.solver.kind:
        raise ValueError("measurement attempt kind differs from solver")
    if spec.attempt_kind not in runtime.task.admissible_kinds:
        raise ValueError("task does not admit the measurement attempt kind")
    runtime.solver.validate(spec.solver_spec, spec.item_set)
    store = runtime.store
    measurement_hash = store.put_measurement(spec, runtime.session.conditions)
    measured = 0
    for item_id in spec.item_set.item_ids:
        if store.has_attempt(measurement_hash, item_id):
            continue
        item = runtime.items.get(item_id)
        if (item.task_id, item.task_version) != (
            spec.task_id,
            spec.task_version,
        ):
            raise ValueError("item belongs to a different task")
        start = perf_counter()
        try:
            attempt = runtime.solver.solve(
                item, runtime.session, spec.solver_spec, runtime.items
            )
        finally:
            for document in runtime.session.drain_evidence():
                store.append_evidence(measurement_hash, document)
        store.put_attempt(
            AttemptRecord(
                measurement_hash=measurement_hash,
                item_id=item_id,
                attempt=attempt,
                wall_time_seconds=perf_counter() - start,
            )
        )
        measured += 1
    attempts = store.attempts(measurement_hash)
    return MeasurementResult(
        spec=spec,
        attempts=tuple(attempts[i] for i in spec.item_set.item_ids),
        measured=measured,
    )


def _inapplicable_reason(
    readout: Readout,
    spec: MeasurementSpec,
    attempts: dict[str, AttemptRecord],
) -> str | None:
    return readout.inapplicable_reason(
        spec.attempt_kind,
        spec.solver_spec.prompt.item_format.candidates,
        tuple(record.attempt for record in attempts.values()),
    )


def inapplicable_reason(
    readout_spec: ReadoutSpec, store: RecordStore
) -> str | None:
    return _inapplicable_reason(
        readout_for(readout_spec.readout_id),
        store.measurement(readout_spec.measurement_hash),
        store.attempts(readout_spec.measurement_hash),
    )


def apply_readout(
    readout_spec: ReadoutSpec,
    store: RecordStore,
    items: ItemIndex,
) -> tuple[ReadoutRecord, ...]:
    spec = store.measurement(readout_spec.measurement_hash)
    readout = readout_for(readout_spec.readout_id)
    judge = judge_for(readout_spec.judge_id)
    attempts = store.attempts(readout_spec.measurement_hash)
    missing = [i for i in spec.item_set.item_ids if i not in attempts]
    if missing:
        raise ValueError(f"{len(missing)} items have no stored attempt")
    reason = _inapplicable_reason(readout, spec, attempts)
    if reason is not None:
        raise ValueError(
            f"readout {readout_spec.readout_id.value!r} does not apply: "
            f"{reason}"
        )
    readout_hash = store.put_readout(readout_spec)
    records = []
    for item_id in spec.item_set.item_ids:
        item = items.get(item_id)
        candidate = readout.read(attempts[item_id].attempt, item.problem)
        record = ReadoutRecord(
            measurement_hash=readout_spec.measurement_hash,
            readout_hash=readout_hash,
            item_id=item_id,
            candidate=candidate,
            scores=judge.judge(candidate, item.reference),
        )
        store.put_readout_record(record)
        records.append(record)
    return tuple(records)
