from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING

import pytest
from conftest import CPU_CONDITIONS, fake_backend
from dr_providers import MessageRole, ProviderStopReason
from dr_providers.local.backend import GenerationOutput as BackendOutput
from dr_providers.surfaces.testing.fake_backend import GenerateCall, ScoreCall
from pydantic import ValidationError
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from po_effect.config import DEFAULT_CONFIG_DIR, load_config
from po_effect.measurement import (
    Abstention,
    AnswerLetterReadout,
    FirstLetterReadout,
    GenerationAttempt,
    GenerationRule,
    GenerationSolver,
    GenerationSpec,
    IndexCandidate,
    MeasurementRuntime,
    MeasurementSpec,
    ReadoutSpec,
    ReasoningVisibility,
    ResponseSplit,
    apply_readout,
    run_measurement,
    split_response,
)
from po_effect.models import (
    AppliedTemplate,
    GenerationOutput,
    ModelCapability,
    ModelsConfig,
    ModelSession,
    ResponseTemplate,
    load_model_handle,
    load_models_config,
    load_response_template,
)
from po_effect.prompts import (
    CandidateKind,
    ChatSettings,
    ItemFormat,
    MultipleChoiceRenderer,
    PromptConfig,
    PromptsConfig,
    resolve_prompt,
)
from po_effect.tasks import MultipleChoiceProblem

if TYPE_CHECKING:
    from pathlib import Path

    from dr_providers.surfaces.testing.fake_backend import FakeLocalBackend

    from po_effect.records import DrStoreRecordStore
    from po_effect.tasks import ArcChallengeTask, ItemIndex, ItemSet

TEMPLATE = (
    "{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>"
    "{% endfor %}{% if add_generation_prompt %}<assistant>"
    "{% if enable_thinking is defined and not enable_thinking %}"
    "<think></think>{% endif %}{% endif %}"
)
CHAT_MODEL = "qwen3_0_6b_instruct"
THINK = ResponseTemplate(
    start_anchor="<assistant>",
    reasoning_open="<think>",
    reasoning_close="</think>",
)
POST_TRAINED = {
    "qwen3_0_6b_instruct",
    "qwen3_1_7b_instruct",
    "qwen3_4b_instruct",
    "qwen3_8b_instruct",
    "qwen3_14b_instruct",
    "olmo3_7b_sft",
    "olmo3_7b_dpo",
    "olmo3_7b_instruct",
    "olmoe_1b_7b_instruct",
    "mistral_7b_v0_3_instruct",
}


def chat(*, enable_thinking: bool = False) -> ChatSettings:
    return ChatSettings(
        enable_thinking=enable_thinking,
        shots_as_turns=False,
        preamble_as_system=False,
    )


UNKNOWN = "[UNK]"


def tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = WordLevel({UNKNOWN: 0}, unk_token=UNKNOWN)
    result = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(vocabulary))
    result.chat_template = TEMPLATE
    return result


def chat_session(
    settings: ChatSettings | None, texts: tuple[str, ...] = ("B",)
) -> tuple[ModelSession, FakeLocalBackend]:
    handle = load_model_handle(name=CHAT_MODEL)
    backend = fake_backend(handle, CPU_CONDITIONS)
    backend = type(backend)(
        execution=backend.execution(),
        results=[
            BackendOutput(text, 5, 1, ProviderStopReason.STOP)
            for text in texts
        ],
    )
    session = ModelSession(
        handle, CPU_CONDITIONS, settings, backend, tokenizer()
    )
    return session, backend


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


def test_generation_applies_the_template_as_one_user_message() -> None:
    session, backend = chat_session(chat())
    with session:
        result = session.generate("Q: x\nA:", 4)
    (call,) = backend.calls
    assert isinstance(call, GenerateCall)
    assert [(m.role, m.content) for m in call.transcript.messages] == [
        (MessageRole.USER, "<user>Q: x\nA:</user><assistant><think></think>")
    ]
    assert result.chat_template == AppliedTemplate(
        template_sha256=sha256(TEMPLATE.encode()).hexdigest(),
        template_kwargs={"enable_thinking": False},
    )


def test_scoring_scores_continuations_after_the_generation_prompt() -> None:
    handle = load_model_handle(name=CHAT_MODEL)
    backend = fake_backend(handle, CPU_CONDITIONS)
    session = ModelSession(
        handle,
        CPU_CONDITIONS,
        chat(enable_thinking=True),
        backend,
        tokenizer(),
    )
    with session:
        result = session.score(
            "Q: x\nA:", (" a", " b"), unconditional_context="A:"
        )
    calls = backend.calls
    assert [
        (c.context, c.continuations) for c in calls if isinstance(c, ScoreCall)
    ] == [
        ("<user>Q: x\nA:</user><assistant>", (" a", " b")),
        ("<user>A:</user><assistant>", (" a", " b")),
    ]
    assert result.chat_template is not None
    assert result.chat_template.template_kwargs == {"enable_thinking": True}


def test_chat_handle_without_chat_block_applies_default_kwargs() -> None:
    session, _ = chat_session(None)
    with session:
        assert session.templated("q") == "<user>q</user><assistant>"
        assert session.applied_template is not None
        assert session.applied_template.template_kwargs == {}


def test_base_handle_ignores_default_chat_block() -> None:
    handle = load_model_handle()
    backend = fake_backend(handle, CPU_CONDITIONS)
    with ModelSession(handle, CPU_CONDITIONS, chat(), backend) as session:
        result = session.score("q", (" a", " b"), unconditional_context=None)
    (call,) = backend.calls
    assert isinstance(call, ScoreCall)
    assert call.context == "q"
    assert result.chat_template is None


def test_base_handle_refuses_thinking() -> None:
    handle = load_model_handle()
    backend = fake_backend(handle, CPU_CONDITIONS)
    with pytest.raises(ValueError, match="enable_thinking requires"):
        ModelSession(
            handle, CPU_CONDITIONS, chat(enable_thinking=True), backend
        )


@pytest.mark.parametrize("flag", ["shots_as_turns", "preamble_as_system"])
def test_chat_layout_flags_must_be_false(flag: str) -> None:
    values = {
        "enable_thinking": False,
        "shots_as_turns": False,
        "preamble_as_system": False,
        flag: True,
    }
    with pytest.raises(ValidationError, match=f"{flag} must be false"):
        ChatSettings.model_validate(values)


def test_chat_block_fields_are_required() -> None:
    with pytest.raises(ValidationError, match="enable_thinking"):
        ChatSettings.model_validate(
            {"shots_as_turns": False, "preamble_as_system": False}
        )


def test_prompt_entries_declare_the_baseline_chat_block() -> None:
    prompts = load_config(DEFAULT_CONFIG_DIR / "prompts.toml", PromptsConfig)
    for name in ("olmes_rc", "olmes_mc", "cfpo"):
        assert prompts.prompts[name].chat == chat()


def test_models_declare_chat_templates_and_response_templates() -> None:
    models = load_models_config().models
    assert {
        name
        for name, handle in models.items()
        if ModelCapability.CHAT_TEMPLATE in handle.capabilities
    } == POST_TRAINED
    assert {
        name: handle.response_template
        for name, handle in models.items()
        if handle.response_template is not None
    } == {
        name: "qwen3_think"
        for name in POST_TRAINED
        if name.startswith("qwen3")
    }
    named = load_response_template(DEFAULT_CONFIG_DIR, "qwen3_think")
    assert named is not None
    assert (named.template.reasoning_open, named.template.reasoning_close) == (
        "<think>",
        "</think>",
    )


def test_response_template_requires_chat_capability() -> None:
    handle = load_model_handle()
    with pytest.raises(ValidationError, match="requires the chat_template"):
        ModelsConfig.model_validate(
            {
                "default": "base",
                "models": {
                    "base": {
                        **handle.model_dump(mode="json"),
                        "response_template": "qwen3_think",
                    }
                },
            }
        )


def test_unknown_response_template_is_rejected(tmp_path: Path) -> None:
    source = (DEFAULT_CONFIG_DIR / "models.toml").read_text()
    (tmp_path / "models.toml").write_text(
        source.replace('"qwen3_think"', '"missing"')
    )
    (tmp_path / "response_templates.toml").write_text(
        (DEFAULT_CONFIG_DIR / "response_templates.toml").read_text()
    )
    with pytest.raises(
        ValueError, match="unknown response template 'missing'"
    ):
        load_models_config(tmp_path)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "B",
            ResponseSplit(
                reasoning=None,
                answer="B",
                reasoning_visibility=ReasoningVisibility.NONE,
            ),
        ),
        (
            "\nB",
            ResponseSplit(
                reasoning=None,
                answer="\nB",
                reasoning_visibility=ReasoningVisibility.NONE,
            ),
        ),
        (
            "<think></think>B",
            ResponseSplit(
                reasoning="",
                answer="B",
                reasoning_visibility=ReasoningVisibility.INLINE,
            ),
        ),
        (
            "<think>\n\n</think>\n\nB",
            ResponseSplit(
                reasoning="\n\n",
                answer="B",
                reasoning_visibility=ReasoningVisibility.INLINE,
            ),
        ),
        (
            "<think>\nA or B\n</think>\n\nThe answer is B",
            ResponseSplit(
                reasoning="\nA or B\n",
                answer="The answer is B",
                reasoning_visibility=ReasoningVisibility.INLINE,
            ),
        ),
        (
            "<think>\nA or",
            ResponseSplit(
                reasoning="\nA or",
                answer="",
                reasoning_visibility=ReasoningVisibility.INLINE,
            ),
        ),
    ],
)
def test_think_tag_split(text: str, expected: ResponseSplit) -> None:
    assert split_response(text, THINK) == expected


def test_no_response_template_keeps_the_whole_text() -> None:
    assert split_response("<think>x</think>B", None) == ResponseSplit(
        reasoning=None,
        answer="<think>x</think>B",
        reasoning_visibility=ReasoningVisibility.NONE,
    )


def attempt(text: str) -> GenerationAttempt:
    return GenerationAttempt(
        output=GenerationOutput(
            text=text, stop_reason="stop", prompt_tokens=1, completion_tokens=1
        ),
        evidence=(),
        chat_template=None,
    )


PROBLEM = MultipleChoiceProblem(question="q", choices=("w", "x", "y", "z"))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("B", IndexCandidate(index=1)),
        (" B", IndexCandidate(index=1)),
        ("B.", IndexCandidate(index=1)),
        ("The answer is B", IndexCandidate(index=1)),
        ("Answer: B", IndexCandidate(index=1)),
        ("answer is (c), not A", IndexCandidate(index=2)),
        ("(B)", IndexCandidate(index=1)),
        ("b", IndexCandidate(index=1)),
        ("I think D fits", IndexCandidate(index=3)),
        ("F", Abstention(reason="answer letter 'F' is outside the 4 choices")),
        (
            "The answer is F",
            Abstention(reason="answer letter 'F' is outside the 4 choices"),
        ),
        ("none of them", Abstention(reason="text contains no choice letter")),
        (
            "I know",
            Abstention(reason="letters ['I'] are outside the 4 choices"),
        ),
        ("", Abstention(reason="truncated text is empty")),
    ],
)
def test_answer_letter_readout(
    text: str, expected: IndexCandidate | Abstention
) -> None:
    assert AnswerLetterReadout(None).read(attempt(text), PROBLEM) == expected


@pytest.mark.parametrize(
    ("text", "first_letter", "answer_letter"),
    [
        (
            "<think></think>\n\nB",
            IndexCandidate(index=1),
            IndexCandidate(index=1),
        ),
        (
            "<think>\nmaybe A\n</think>\n\nThe answer is C",
            Abstention(reason="first character 'T' is outside the 4 choices"),
            IndexCandidate(index=2),
        ),
        (
            "<think>\nmaybe A",
            Abstention(reason="truncated text is empty"),
            Abstention(reason="truncated text is empty"),
        ),
    ],
)
def test_letter_readouts_read_the_answer_part(
    text: str,
    first_letter: IndexCandidate | Abstention,
    answer_letter: IndexCandidate | Abstention,
) -> None:
    generated = attempt(text)
    first = FirstLetterReadout(THINK)
    assert first.read(generated, PROBLEM) == first_letter
    assert AnswerLetterReadout(THINK).read(generated, PROBLEM) == answer_letter
    assert first.reasoning_visibility(generated) is ReasoningVisibility.INLINE


def chat_measurement(  # noqa: PLR0913, PLR0917
    task: ArcChallengeTask,
    items: ItemIndex,
    item_set: ItemSet,
    prompt_config: PromptConfig,
    store: DrStoreRecordStore,
    texts: tuple[str, ...],
) -> tuple[MeasurementSpec, DrStoreRecordStore]:
    config = prompt_config.model_copy(
        update={"item_format": label_format(), "chat": chat()}
    )
    solver = GenerationSolver(MultipleChoiceRenderer())
    session, _ = chat_session(chat(), texts)
    spec = MeasurementSpec(
        task_id=task.task_id,
        task_version=task.version,
        item_set=item_set,
        model=session.handle,
        solver_id=solver.solver_id,
        solver_spec=GenerationSpec(
            prompt=resolve_prompt("example", config, task, items),
            max_new_tokens=4,
        ),
        attempt_kind=solver.kind,
    )
    with session:
        run_measurement(
            spec, MeasurementRuntime(task, items, session, solver, store)
        )
    return spec, store


def test_chat_measurement_records_template_and_reasoning_visibility(
    task: ArcChallengeTask,
    items: ItemIndex,
    item_set: ItemSet,
    prompt_config: PromptConfig,
    record_store: DrStoreRecordStore,
) -> None:
    spec, store = chat_measurement(
        task,
        items,
        item_set,
        prompt_config,
        record_store,
        ("<think>\nhm\n</think>\n\nAnswer: B", "A"),
    )
    attempts = store.attempts(spec.identity_hash)
    for record in attempts.values():
        assert record.attempt.chat_template == AppliedTemplate(
            template_sha256=sha256(TEMPLATE.encode()).hexdigest(),
            template_kwargs={"enable_thinking": False},
        )
    named = load_response_template(DEFAULT_CONFIG_DIR, "qwen3_think")
    records = apply_readout(
        ReadoutSpec(
            measurement_hash=spec.identity_hash,
            readout_id=GenerationRule.ANSWER_LETTER,
            judge_id="index_match",
            response_template=named,
        ),
        store,
        items,
    )
    assert [(r.candidate, r.reasoning_visibility) for r in records] == [
        (IndexCandidate(index=1), ReasoningVisibility.INLINE),
        (IndexCandidate(index=0), ReasoningVisibility.NONE),
    ]
    assert [r.scores.accuracy for r in records] == [1, 1]
    with pytest.raises(ValueError, match="differs from the model's"):
        apply_readout(
            ReadoutSpec(
                measurement_hash=spec.identity_hash,
                readout_id=GenerationRule.FIRST_LETTER,
                judge_id="index_match",
                response_template=None,
            ),
            store,
            items,
        )


def test_measurement_refuses_session_with_other_chat_settings(
    task: ArcChallengeTask,
    items: ItemIndex,
    item_set: ItemSet,
    prompt_config: PromptConfig,
    record_store: DrStoreRecordStore,
) -> None:
    solver = GenerationSolver(MultipleChoiceRenderer())
    session, _ = chat_session(chat(enable_thinking=True))
    spec = MeasurementSpec(
        task_id=task.task_id,
        task_version=task.version,
        item_set=item_set,
        model=session.handle,
        solver_id=solver.solver_id,
        solver_spec=GenerationSpec(
            prompt=resolve_prompt(
                "example",
                prompt_config.model_copy(update={"chat": chat()}),
                task,
                items,
            ),
            max_new_tokens=4,
        ),
        attempt_kind=solver.kind,
    )
    with session, pytest.raises(ValueError, match="chat settings differ"):
        run_measurement(
            spec,
            MeasurementRuntime(task, items, session, solver, record_store),
        )
