from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from conftest import make_item, make_row
from pydantic import ValidationError

from po_effect.config import DEFAULT_CONFIG_DIR, load_config
from po_effect.prompts import (
    CandidateKind,
    ItemFormat,
    MultipleChoiceRenderer,
    PromptEntry,
    PromptsConfig,
    PromptSpec,
    ShotRef,
    load_prompt_config,
    resolve_prompt,
)
from po_effect.tasks import (
    AttemptKind,
    MultipleChoiceProblem,
    build_item_index,
)

if TYPE_CHECKING:
    from dr_hf import SourceRow

    from po_effect.prompts import PromptConfig
    from po_effect.tasks import ArcTask, ItemIndex, ItemSet


def test_shots_resolve_to_built_items(
    task: ArcTask,
    rows: tuple[SourceRow, ...],
    prompt: PromptSpec,
) -> None:
    built = {row.native_id: task.build(row).item_id for row in rows}
    assert prompt.shot_item_ids == (built["t2"], built["t1"])
    assert prompt.task_id == task.task_id
    assert prompt.task_version == task.version


@pytest.mark.parametrize(
    ("ref", "message"),
    [
        (ShotRef(split="validation", native_id="t1"), "'validation'"),
        (ShotRef(split="train", native_id="missing"), "'missing'"),
        (ShotRef(split="train", native_id="x"), "'x'"),
    ],
)
def test_unknown_shot_reference_is_named(
    task: ArcTask,
    prompt_config: PromptConfig,
    items: ItemIndex,
    ref: ShotRef,
    message: str,
) -> None:
    config = prompt_config.model_copy(update={"shots": (ref,)})
    with pytest.raises(ValueError, match=message):
        resolve_prompt("example", config, task, items)


def test_prompt_hash_tracks_shot_content(
    task: ArcTask,
    rows: tuple[SourceRow, ...],
    prompt_config: PromptConfig,
    prompt: PromptSpec,
) -> None:
    edited = (
        make_row("train", "t1", "Sky colour?", ("blue", "red"), "A"),
        *rows[1:],
    )
    changed = resolve_prompt(
        "example", prompt_config, task, build_item_index(task, edited)
    )
    assert changed.identity_hash != prompt.identity_hash
    again = resolve_prompt(
        "example", prompt_config, task, build_item_index(task, rows)
    )
    assert again.identity_hash == prompt.identity_hash


def test_renderer_formats_shots_and_target_alike(
    items: ItemIndex, item_set: ItemSet, prompt: PromptSpec
) -> None:
    target = items.get(item_set.item_ids[0])
    rendered = MultipleChoiceRenderer().render(
        prompt,
        target.problem,
        tuple(items.get(i) for i in prompt.shot_item_ids),
    )
    assert rendered.context == (
        "Question: Two plus two?\nAnswer: 4\n\n"
        "Question: Sky color?\nAnswer: blue\n\n"
        "Question: q\nAnswer:"
    )
    assert rendered.candidates == (" a", " éé")
    assert rendered.unconditional_context == "Answer:"
    with pytest.raises(ValueError, match="shot item ids"):
        MultipleChoiceRenderer().render(prompt, target.problem, ())


def test_unconditional_context_comes_from_the_format() -> None:
    shot = make_item(question="Hot or cold?", choices=("hot", "cold"))
    prompt = PromptSpec(
        name="example",
        attempt=AttemptKind.SCORE,
        admits=(AttemptKind.SCORE,),
        task_id="arc_challenge",
        task_version=1,
        preamble="Quiz",
        shot_item_ids=(shot.item_id,),
        shot_separator="\n--\n",
        item_format=ItemFormat(
            question_prefix="Q> ",
            answer_prefix="A>",
            answer_break=" | ",
            continuation_prefix="  ",
            candidates=CandidateKind.CHOICE_TEXT,
            unconditional_context="Reply:",
        ),
    )
    rendered = MultipleChoiceRenderer().render(
        prompt,
        MultipleChoiceProblem(question="Up or down?", choices=("up", "down")),
        (shot,),
    )
    assert rendered.context == (
        "Quiz\n--\nQ> Hot or cold? | A>  hot\n--\nQ> Up or down? | A>"
    )
    assert rendered.candidates == ("  up", "  down")
    assert rendered.unconditional_context == "Reply:"


def listing_format(
    candidates: CandidateKind,
    *,
    choices_prefix: str = "Options:\n",
    label_prefix: str = "(",
    label_suffix: str = ") ",
) -> ItemFormat:
    return ItemFormat(
        question_prefix="Q> ",
        answer_prefix="A>",
        answer_break="\n",
        continuation_prefix=" ",
        candidates=candidates,
        choices_prefix=choices_prefix,
        label_prefix=label_prefix,
        label_suffix=label_suffix,
    )


def render_one(
    fmt: ItemFormat,
    *,
    preamble: str | None = None,
    shots_header: str | None = None,
    shots_header_break: str | None = None,
) -> tuple[str, tuple[str, ...], str | None]:
    shot = make_item(question="Hot or cold?", choices=("hot", "cold"), gold=1)
    prompt = PromptSpec(
        name="example",
        attempt=AttemptKind.SCORE,
        admits=(AttemptKind.SCORE,),
        task_id="arc_challenge",
        task_version=1,
        preamble=preamble,
        shots_header=shots_header,
        shots_header_break=shots_header_break,
        shot_item_ids=(shot.item_id,),
        shot_separator="\n--\n",
        item_format=fmt,
    )
    rendered = MultipleChoiceRenderer().render(
        prompt,
        MultipleChoiceProblem(question="Up or down?", choices=("up", "down")),
        (shot,),
    )
    return (
        rendered.context,
        rendered.candidates,
        rendered.unconditional_context,
    )


@pytest.mark.parametrize(
    ("choices_prefix", "label_prefix", "label_suffix", "lines"),
    [
        ("Options:\n", "(", ") ", ("(A) up", "(B) down")),
        ("", " ", ". ", (" A. up", " B. down")),
        ("Pick:\n", "", ": ", ("A: up", "B: down")),
    ],
)
def test_listing_renders_each_label_style(
    choices_prefix: str, label_prefix: str, label_suffix: str, lines: tuple
) -> None:
    fmt = listing_format(
        CandidateKind.CHOICE_TEXT,
        choices_prefix=choices_prefix,
        label_prefix=label_prefix,
        label_suffix=label_suffix,
    )
    context, _, _ = render_one(fmt)
    assert context.endswith(
        "Q> Up or down?\n" + choices_prefix + "\n".join(lines) + "\nA>"
    )


def test_label_candidates_score_letters() -> None:
    context, candidates, unconditional = render_one(
        listing_format(CandidateKind.CHOICE_LABEL)
    )
    assert context == (
        "Q> Hot or cold?\nOptions:\n(A) hot\n(B) cold\nA> B\n--\n"
        "Q> Up or down?\nOptions:\n(A) up\n(B) down\nA>"
    )
    assert candidates == (" A", " B")
    assert unconditional is None


def test_text_candidates_score_choices_after_a_listing() -> None:
    context, candidates, _ = render_one(
        listing_format(CandidateKind.CHOICE_TEXT)
    )
    assert "A> cold\n--\n" in context
    assert candidates == (" up", " down")


def test_preamble_and_header_precede_the_shots() -> None:
    fmt = listing_format(CandidateKind.CHOICE_LABEL)
    context, _, _ = render_one(
        fmt,
        preamble="Quiz",
        shots_header="Examples:\n",
        shots_header_break="+",
    )
    assert context.startswith("Quiz\n--\nExamples:\n+Q> Hot or cold?\n")
    assert context.count("Examples:") == 1


@pytest.mark.parametrize(
    "fields",
    [
        {"candidates": CandidateKind.CHOICE_LABEL},
        {"candidates": CandidateKind.CHOICE_TEXT, "label_suffix": ". "},
        {
            "candidates": CandidateKind.CHOICE_TEXT,
            "choices_prefix": "",
            "label_prefix": "",
        },
    ],
)
def test_format_rejects_incomplete_listings(fields: dict) -> None:
    with pytest.raises(ValidationError, match=r"listing|set together"):
        ItemFormat(
            question_prefix="Q> ",
            answer_prefix="A>",
            answer_break="\n",
            continuation_prefix=" ",
            **fields,
        )


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"shots_header": "Examples:"}, "set together"),
        ({"shots_header_break": "\n"}, "set together"),
        (
            {"shots_header": "Examples:", "shots_header_break": "\n"},
            "requires shots",
        ),
    ],
)
def test_prompt_rejects_misplaced_headers(fields: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PromptSpec(
            name="example",
            attempt=AttemptKind.SCORE,
            admits=(AttemptKind.SCORE,),
            task_id="arc_challenge",
            task_version=1,
            item_format=listing_format(CandidateKind.CHOICE_TEXT),
            **fields,
        )


def test_prompt_config_lists_named_entries() -> None:
    config = load_config(DEFAULT_CONFIG_DIR / "prompts.toml", PromptsConfig)
    assert {"olmes_rc", "olmes_mc", "cfpo"} <= set(config.prompts)
    for entry in config.prompts.values():
        assert isinstance(entry, PromptEntry)
        assert isinstance(entry.item_format, ItemFormat)
        assert {shot.split for shot in entry.shots} == {"train"}
        assert len(entry.shots) == 5
        assert entry.provenance
    assert config.prompts["olmes_rc"].item_format.unconditional_context
    assert (
        config.prompts["olmes_mc"].item_format.candidates
        is CandidateKind.CHOICE_LABEL
    )
    assert isinstance(config.prompts["cfpo"].preamble, str)
    assert isinstance(config.prompts["cfpo"].shots_header, str)


def test_prompt_config_declares_attempt_kinds() -> None:
    prompts = load_config(
        DEFAULT_CONFIG_DIR / "prompts.toml", PromptsConfig
    ).prompts
    declared = {
        name: (entry.attempt, set(entry.admits))
        for name, entry in prompts.items()
    }
    assert declared == {
        "olmes_rc": (AttemptKind.SCORE, {AttemptKind.SCORE}),
        "olmes_mc": (AttemptKind.SCORE, set(AttemptKind)),
        "cfpo": (AttemptKind.GENERATE, set(AttemptKind)),
        **{
            f"olmes_rc_{task}": (AttemptKind.SCORE, {AttemptKind.SCORE})
            for task in (
                "arc_easy",
                "hellaswag",
                "piqa",
                "winogrande",
                "boolq",
                "openbookqa",
                "csqa",
            )
        },
    }


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"admits": ("score",)}, "include the declared attempt"),
        ({"admits": ("generate", "generate")}, "twice"),
        ({"admits": ()}, "at least 1"),
    ],
)
def test_prompt_entry_admits_its_attempt(fields: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PromptSpec(
            name="example",
            attempt=AttemptKind.GENERATE,
            task_id="arc_challenge",
            task_version=1,
            item_format=listing_format(CandidateKind.CHOICE_TEXT),
            **fields,
        )


def test_unknown_prompt_name_is_named() -> None:
    with pytest.raises(ValueError, match="'missing'"):
        load_prompt_config(DEFAULT_CONFIG_DIR, "missing")
