from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from dr_hf import SourceRow
from dr_providers import LocalExecutionEvidence
from dr_providers.local.backend import ScoredContinuation
from dr_providers.surfaces.testing.fake_backend import FakeLocalBackend

from po_effect.measurement import (
    AttemptRecord,
    DecisionRule,
    MeasurementRuntime,
    MeasurementSpec,
    ReadoutSpec,
    ScoredCandidatesAttempt,
    ScoredCandidatesSolver,
    SingleCallSpec,
)
from po_effect.models import (
    Device,
    ModelHandle,
    ModelSession,
    RunConditions,
    load_model_handle,
    provider_config,
)
from po_effect.prompts import (
    CandidateKind,
    ItemFormat,
    MultipleChoiceRenderer,
    PromptConfig,
    PromptSpec,
    ShotRef,
    resolve_prompt,
)
from po_effect.records import FileRecordStore
from po_effect.tasks import (
    ArcChallengeTask,
    IndexReference,
    Item,
    ItemIndex,
    ItemSet,
    ItemSetDerivation,
    ItemSetRole,
    MultipleChoiceProblem,
    build_item_index,
    load_arc_challenge,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExampleMeasurement:
    spec: MeasurementSpec
    runtime: MeasurementRuntime
    backend: FakeLocalBackend
    root: Path


CPU_CONDITIONS = RunConditions(device=Device.CPU, batch_size=8)


def fake_backend(
    handle: ModelHandle, conditions: RunConditions
) -> FakeLocalBackend:
    local = provider_config(handle, conditions).definition.local
    assert local is not None
    execution = LocalExecutionEvidence(
        device=local.device,
        device_name="fixture",
        dtype=local.dtype,
        float32_matmul_precision=local.float32_matmul_precision,
        quantization=local.quantization,
        revision_commit=local.revision,
        torch_version="fixture",
        transformers_version="fixture",
        chat_template_applied=False,
        add_bos_token=local.add_bos_token,
    )
    return FakeLocalBackend(
        execution=execution,
        results=[
            (
                ScoredContinuation(-2.0, 2, None, 10),
                ScoredContinuation(-3.0, 2, None, 10),
            ),
            (
                ScoredContinuation(-3.0, 2, None, 2),
                ScoredContinuation(-8.0, 2, None, 2),
            ),
        ],
    )


def make_row(
    split: str,
    native_id: str,
    question: str,
    choices: tuple[str, ...],
    answer_key: str,
) -> SourceRow:
    return SourceRow(
        pin=load_arc_challenge().source,
        origin_split=split,
        native_id=native_id,
        fields={
            "id": native_id,
            "question": question,
            "choices": {
                "text": list(choices),
                "label": list("ABCDE"[: len(choices)]),
            },
            "answerKey": answer_key,
        },
    )


def make_item(
    *,
    question: str = "q",
    choices: tuple[str, ...] = ("a", "b"),
    gold: int = 0,
    version: int = 1,
    native_id: str = "x",
) -> Item:
    return Item(
        task_id="arc_challenge",
        task_version=version,
        native_id=native_id,
        problem=MultipleChoiceProblem(question=question, choices=choices),
        reference=IndexReference(gold_index=gold),
    )


def example_format() -> ItemFormat:
    return ItemFormat(
        question_prefix="Q: ",
        answer_prefix="A:",
        answer_break="\n",
        continuation_prefix=" ",
        candidates=CandidateKind.CHOICE_TEXT,
        unconditional_context="A:",
    )


def evaluation_item_set(task: ArcChallengeTask, items: ItemIndex) -> ItemSet:
    origin = items.splits[task.config.split]
    return ItemSet(
        name=origin.key.name,
        task_id=task.task_id,
        task_version=task.version,
        role=ItemSetRole.REPORT,
        item_ids=tuple(items.item_ids[key] for key in origin.native_ids),
        derivation=ItemSetDerivation(
            method="origin_split", parent=origin.content_hash
        ),
    )


def scored_attempt(record: AttemptRecord) -> ScoredCandidatesAttempt:
    attempt = record.attempt
    assert isinstance(attempt, ScoredCandidatesAttempt)
    return attempt


def readout_spec(
    example: ExampleMeasurement, rule: DecisionRule
) -> ReadoutSpec:
    return ReadoutSpec(
        measurement_hash=example.spec.identity_hash,
        readout_id=rule,
        judge_id="index_match",
    )


@pytest.fixture
def task() -> ArcChallengeTask:
    return load_arc_challenge()


@pytest.fixture
def rows() -> tuple[SourceRow, ...]:
    return (
        make_row("train", "t1", "Sky color?", ("blue", "red"), "A"),
        make_row("train", "t2", "Two plus two?", ("3", "4"), "B"),
        make_row("test", "x", "q", ("a", "éé"), "B"),
        make_row("test", "y", "r", ("c", "d"), "A"),
    )


@pytest.fixture
def prompt_config() -> PromptConfig:
    return PromptConfig(
        shots=(
            ShotRef(split="train", native_id="t2"),
            ShotRef(split="train", native_id="t1"),
        ),
        item_format=ItemFormat(
            question_prefix="Question: ",
            answer_prefix="Answer:",
            answer_break="\n",
            continuation_prefix=" ",
            candidates=CandidateKind.CHOICE_TEXT,
            unconditional_context="Answer:",
        ),
    )


@pytest.fixture
def items(task: ArcChallengeTask, rows: tuple[SourceRow, ...]) -> ItemIndex:
    return build_item_index(task, rows)


@pytest.fixture
def item_set(task: ArcChallengeTask, items: ItemIndex) -> ItemSet:
    return evaluation_item_set(task, items)


@pytest.fixture
def prompt(
    task: ArcChallengeTask, items: ItemIndex, prompt_config: PromptConfig
) -> PromptSpec:
    return resolve_prompt(prompt_config, task, items)


@pytest.fixture
def example(
    task: ArcChallengeTask,
    items: ItemIndex,
    item_set: ItemSet,
    prompt: PromptSpec,
    tmp_path: Path,
) -> Iterator[ExampleMeasurement]:
    handle = load_model_handle()
    backend = fake_backend(handle, CPU_CONDITIONS)
    solver = ScoredCandidatesSolver(MultipleChoiceRenderer())
    spec = MeasurementSpec(
        task_id=task.task_id,
        task_version=task.version,
        item_set=item_set,
        model=handle,
        solver_id=solver.solver_id,
        solver_spec=SingleCallSpec(prompt=prompt),
        attempt_kind=solver.kind,
    )
    with ModelSession(handle, CPU_CONDITIONS, backend) as session:
        runtime = MeasurementRuntime(
            task,
            items,
            session,
            solver,
            FileRecordStore(tmp_path / "records"),
        )
        yield ExampleMeasurement(
            spec, runtime, backend, tmp_path / "records" / spec.identity_hash
        )
