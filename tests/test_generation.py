from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest
from conftest import CPU_CONDITIONS, fake_backend
from dr_providers import (
    GenerationControls,
    MessageRole,
    PromptRendering,
    ProviderCallConfig,
    ProviderGenerateRequest,
    ProviderStopReason,
)
from dr_providers.local.backend import GenerationOutput as BackendOutput
from dr_providers.surfaces.testing.fake_backend import GenerateCall

from po_effect.measurement import (
    Abstention,
    ChoiceTextPrefixReadout,
    ChoiceTextReadout,
    DecisionRule,
    FirstLetterReadout,
    GenerationAttempt,
    GenerationRule,
    GenerationSolver,
    GenerationSpec,
    IndexCandidate,
    IndexMatchJudge,
    LetterAnywhereReadout,
    MeasurementRuntime,
    MeasurementSpec,
    ReadoutSpec,
    apply_readout,
    inapplicable_reason,
    run_measurement,
)
from po_effect.models import (
    GenerationOutput,
    ModelCapability,
    ModelSession,
    load_model_handle,
    provider_config,
)
from po_effect.prompts import (
    CandidateKind,
    ItemFormat,
    MultipleChoiceRenderer,
    PromptConfig,
    resolve_prompt,
)
from po_effect.records import DrStoreRecordStore, export_wide
from po_effect.records.store import evidence_key
from po_effect.tasks import (
    AttemptKind,
    IndexReference,
    MultipleChoiceProblem,
)

if TYPE_CHECKING:
    from conftest import ExampleMeasurement
    from dr_providers.surfaces.testing.fake_backend import FakeLocalBackend

    from po_effect.tasks import ArcTask, ItemIndex, ItemSet

MAX_NEW_TOKENS = 7


@dataclass(frozen=True, slots=True)
class GenerationRun:
    spec: MeasurementSpec
    store: DrStoreRecordStore
    backend: FakeLocalBackend


def label_format() -> ItemFormat:
    return ItemFormat(
        question_prefix="Question: ",
        answer_prefix="Answer:",
        answer_break="\n",
        continuation_prefix=" ",
        candidates=CandidateKind.CHOICE_LABEL,
        choices_prefix="",
        label_prefix=" ",
        label_suffix=". ",
    )


def generate(  # noqa: PLR0913, PLR0917
    task: ArcTask,
    items: ItemIndex,
    item_set: ItemSet,
    prompt_config: PromptConfig,
    store: DrStoreRecordStore,
    texts: tuple[str, ...],
) -> GenerationRun:
    handle = load_model_handle()
    backend = fake_backend(handle, CPU_CONDITIONS)
    backend = type(backend)(
        execution=backend.execution(),
        results=[
            BackendOutput(text, 11, 3, ProviderStopReason.LENGTH)
            for text in texts
        ],
    )
    solver = GenerationSolver(MultipleChoiceRenderer())
    spec = MeasurementSpec(
        task_id=task.task_id,
        task_version=task.version,
        item_set=item_set,
        model=handle,
        solver_id=solver.solver_id,
        solver_spec=GenerationSpec(
            prompt=resolve_prompt("example", prompt_config, task, items),
            max_new_tokens=MAX_NEW_TOKENS,
        ),
        attempt_kind=solver.kind,
    )
    with ModelSession(handle, CPU_CONDITIONS, None, backend) as session:
        run_measurement(
            spec, MeasurementRuntime(task, items, session, solver, store)
        )
    return GenerationRun(spec, store, backend)


def test_generation_attempt_records_output_and_evidence(
    task: ArcTask,
    items: ItemIndex,
    item_set: ItemSet,
    prompt_config: PromptConfig,
    record_store: DrStoreRecordStore,
) -> None:
    run = generate(
        task, items, item_set, prompt_config, record_store, (" a\nb", " d")
    )
    attempts = run.store.attempts(run.spec.identity_hash)
    first = attempts[item_set.item_ids[0]].attempt
    assert isinstance(first, GenerationAttempt)
    assert first.output == GenerationOutput(
        text=" a\nb",
        stop_reason="length",
        prompt_tokens=11,
        completion_tokens=3,
    )
    prompt = run.spec.solver_spec.prompt
    rendered = MultipleChoiceRenderer().render(
        prompt,
        items.get(item_set.item_ids[0]).problem,
        tuple(items.get(i) for i in prompt.shot_item_ids),
    )
    call = run.backend.calls[0]
    assert isinstance(call, GenerateCall)
    assert [(m.role, m.content) for m in call.transcript.messages] == [
        (MessageRole.USER, rendered.context)
    ]
    assert call.rendering is PromptRendering.FLAT_TEXT
    assert call.controls.token_limit == MAX_NEW_TOKENS
    assert call.controls.temperature is None
    assert call.controls.top_p is None
    request = ProviderGenerateRequest(
        config=ProviderCallConfig(
            definition=provider_config(
                load_model_handle(), CPU_CONDITIONS
            ).definition,
            controls=GenerationControls(token_limit=MAX_NEW_TOKENS),
        ),
        transcript=call.transcript,
    )
    (ref,) = first.evidence
    assert ref.request_hash == request.identity_hash
    documents = []
    for item_id in item_set.item_ids:
        (stored,) = attempts[item_id].attempt.evidence
        reference = run.store.objects.resolve(
            evidence_key(stored.evidence_hash)
        )
        assert reference is not None
        document = run.store.objects.get(reference)
        assert isinstance(document, dict)
        assert isinstance(document["payload"], dict)
        assert document["payload"]["request_hash"] == stored.request_hash
        documents.append(document["payload"])
    response = documents[0]["response"]
    assert isinstance(response, dict)
    assert response["text"] == " a\nb"


def problem(*choices: str) -> MultipleChoiceProblem:
    return MultipleChoiceProblem(question="q", choices=choices)


def attempt(text: str) -> GenerationAttempt:
    return GenerationAttempt(
        output=GenerationOutput(
            text=text,
            stop_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        ),
        evidence=(),
        chat_template=None,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (" B. blue\nC", IndexCandidate(index=1)),
        ("2", IndexCandidate(index=1)),
        ("c) red", IndexCandidate(index=2)),
        (
            "D",
            Abstention(reason="first character 'D' is outside the 3 choices"),
        ),
        (
            "4",
            Abstention(reason="first character '4' is outside the 3 choices"),
        ),
        (
            "(A)",
            Abstention(reason="first character '(' is not a letter or digit"),
        ),
        ("   ", Abstention(reason="truncated text is empty")),
        ("\nA", Abstention(reason="truncated text is empty")),
    ],
)
def test_first_letter_readout(
    text: str, expected: IndexCandidate | Abstention
) -> None:
    readout = FirstLetterReadout(None)
    assert readout.read(attempt(text), problem("x", "y", "z")) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (" red\nblue", IndexCandidate(index=1)),
        ("blue skyline", IndexCandidate(index=2)),
        ("blue", IndexCandidate(index=0)),
        ("green", Abstention(reason="text starts with no choice text")),
        ("", Abstention(reason="truncated text is empty")),
    ],
)
def test_choice_text_prefix_readout(
    text: str, expected: IndexCandidate | Abstention
) -> None:
    readout = ChoiceTextPrefixReadout(None)
    choices = problem("blue", " red ", "blue sky")
    assert readout.read(attempt(text), choices) == expected


NO_MENTION = "text mentions no choice letter or choice text"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is (B).", IndexCandidate(index=1)),
        ("B. blue", IndexCandidate(index=1)),
        ("C: red\nso A", IndexCandidate(index=0)),
        ("<think>maybe A</think>\nAnswer: c", IndexCandidate(index=2)),
        ("<think>A or B", Abstention(reason="text is empty")),
        ("blue sky", Abstention(reason=NO_MENTION)),
        ("ABC", Abstention(reason=NO_MENTION)),
        (
            "then D",
            Abstention(reason="last mention 'D' is outside the 3 choices"),
        ),
        ("", Abstention(reason="text is empty")),
    ],
)
def test_letter_anywhere_readout(
    text: str, expected: IndexCandidate | Abstention
) -> None:
    readout = LetterAnywhereReadout(None)
    choices = problem("blue", "red", "blue sky")
    assert readout.read(attempt(text), choices) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Blue Sky!", IndexCandidate(index=2)),
        ("It is red.", IndexCandidate(index=1)),
        ("B, then blue", IndexCandidate(index=0)),
        ("blue, so the answer is C", IndexCandidate(index=2)),
        ("bluesky", Abstention(reason=NO_MENTION)),
        ("<think>red</think>blue", IndexCandidate(index=0)),
        ("", Abstention(reason="text is empty")),
    ],
)
def test_choice_text_readout(
    text: str, expected: IndexCandidate | Abstention
) -> None:
    readout = ChoiceTextReadout(None)
    choices = problem("blue", "red", "blue sky")
    assert readout.read(attempt(text), choices) == expected


def test_forgiving_readouts_apply_to_label_formats_only() -> None:
    for readout in (LetterAnywhereReadout(None), ChoiceTextReadout(None)):
        assert (
            readout.inapplicable_reason(
                AttemptKind.GENERATE, CandidateKind.CHOICE_LABEL, ()
            )
            is None
        )
        assert readout.inapplicable_reason(
            AttemptKind.GENERATE, CandidateKind.CHOICE_TEXT, ()
        ) == ("applies to choice_label formats, not choice_text formats")


def test_index_match_scores_abstention() -> None:
    judge = IndexMatchJudge()
    reference = IndexReference(gold_index=0)
    abstained = judge.judge(Abstention(reason="none"), reference)
    assert (abstained.accuracy, abstained.abstained) == (0, 1)
    chosen = judge.judge(IndexCandidate(index=0), reference)
    assert (chosen.accuracy, chosen.abstained) == (1, 0)


def readout(measurement: MeasurementSpec, rule: str) -> ReadoutSpec:
    return ReadoutSpec.model_validate(
        {
            "measurement_hash": measurement.identity_hash,
            "readout_id": rule,
            "judge_id": "index_match",
            "response_template": None,
        }
    )


def test_readouts_apply_by_attempt_kind_and_format(
    task: ArcTask,
    items: ItemIndex,
    item_set: ItemSet,
    prompt_config: PromptConfig,
    record_store: DrStoreRecordStore,
) -> None:
    run = generate(task, items, item_set, prompt_config, record_store, ("a",))
    for rule in DecisionRule:
        spec = readout(run.spec, rule)
        assert inapplicable_reason(spec, run.store) == (
            "applies to score attempts, not generate attempts"
        )
        with pytest.raises(ValueError, match="does not apply"):
            apply_readout(spec, run.store, items)
    first_letter = readout(run.spec, GenerationRule.FIRST_LETTER)
    assert inapplicable_reason(first_letter, run.store) == (
        "applies to choice_label formats, not choice_text formats"
    )
    records = apply_readout(
        readout(run.spec, GenerationRule.CHOICE_TEXT_PREFIX), run.store, items
    )
    assert [r.candidate for r in records] == [
        IndexCandidate(index=0),
        Abstention(reason="text starts with no choice text"),
    ]


def test_generation_readouts_refuse_scored_attempts(
    example: ExampleMeasurement,
) -> None:
    items = example.runtime.items
    run_measurement(example.spec, example.runtime)
    for rule in GenerationRule:
        spec = readout(example.spec, rule)
        assert inapplicable_reason(spec, example.runtime.store) == (
            "applies to generate attempts, not score attempts"
        )
        with pytest.raises(ValueError, match="does not apply"):
            apply_readout(spec, example.runtime.store, items)


def limited_session(
    capability: ModelCapability,
) -> tuple[ModelSession, FakeLocalBackend]:
    handle = load_model_handle()
    limited = handle.model_copy(
        update={"capabilities": frozenset({capability})}
    )
    backend = fake_backend(handle, CPU_CONDITIONS)
    return ModelSession(limited, CPU_CONDITIONS, None, backend), backend


def test_generate_requires_generate_capability() -> None:
    session, backend = limited_session(ModelCapability.SCORE)
    with session, pytest.raises(ValueError, match="lacks the generate"):
        session.generate("context", MAX_NEW_TOKENS)
    assert backend.calls == []


def test_score_requires_score_capability() -> None:
    session, backend = limited_session(ModelCapability.GENERATE)
    with session, pytest.raises(ValueError, match="lacks the score"):
        session.score("context", (" a",), unconditional_context=None)
    assert backend.calls == []


def test_export_generation_measurement(
    task: ArcTask,
    items: ItemIndex,
    item_set: ItemSet,
    prompt_config: PromptConfig,
    record_store: DrStoreRecordStore,
) -> None:
    config = prompt_config.model_copy(update={"item_format": label_format()})
    run = generate(
        task,
        items,
        item_set,
        config,
        record_store,
        (" B. éé\nQuestion", "Q"),
    )
    applied = []
    for rule in GenerationRule:
        spec = readout(run.spec, rule)
        if inapplicable_reason(spec, run.store) is None:
            apply_readout(spec, run.store, items)
            applied.append(spec.identity_hash)
    path = record_store.path.parent / "export.parquet"
    export_wide(run.store, run.spec.identity_hash, applied, items, path)
    table = pq.read_table(path).to_pydict()
    assert table == {
        "item_id": list(item_set.item_ids),
        "native_id": ["x", "y"],
        "gold_index": [1, 0],
        "text": [" B. éé\nQuestion", "Q"],
        "stop_reason": ["length", "length"],
        "prompt_tokens": [11, 11],
        "completion_tokens": [3, 3],
        "first_letter_index_match_predicted_index": [1, None],
        "first_letter_index_match_correct": [True, False],
        "first_letter_index_match_abstained": [False, True],
        "answer_letter_index_match_predicted_index": [1, None],
        "answer_letter_index_match_correct": [True, False],
        "answer_letter_index_match_abstained": [False, True],
        "letter_anywhere_index_match_predicted_index": [1, None],
        "letter_anywhere_index_match_correct": [True, False],
        "letter_anywhere_index_match_abstained": [False, True],
        "choice_text_index_match_predicted_index": [1, None],
        "choice_text_index_match_correct": [True, False],
        "choice_text_index_match_abstained": [False, True],
        "reasoning_visibility": ["none", "none"],
    }
