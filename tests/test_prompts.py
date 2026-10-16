from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from conftest import make_item, make_row

from po_effect.config import DEFAULT_CONFIG_DIR, load_config
from po_effect.prompts import (
    ItemFormat,
    MultipleChoiceRenderer,
    PromptsConfig,
    PromptSpec,
    ShotRef,
    resolve_prompt,
)
from po_effect.tasks import MultipleChoiceProblem, build_item_index

if TYPE_CHECKING:
    from dr_hf import SourceRow

    from po_effect.prompts import PromptConfig
    from po_effect.tasks import ArcChallengeTask, ItemIndex, ItemSet


def test_shots_resolve_to_built_items(
    task: ArcChallengeTask,
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
    task: ArcChallengeTask,
    prompt_config: PromptConfig,
    items: ItemIndex,
    ref: ShotRef,
    message: str,
) -> None:
    config = prompt_config.model_copy(update={"shots": (ref,)})
    with pytest.raises(ValueError, match=message):
        resolve_prompt(config, task, items)


def test_prompt_hash_tracks_shot_content(
    task: ArcChallengeTask,
    rows: tuple[SourceRow, ...],
    prompt_config: PromptConfig,
    prompt: PromptSpec,
) -> None:
    edited = (
        make_row("train", "t1", "Sky colour?", ("blue", "red"), "A"),
        *rows[1:],
    )
    changed = resolve_prompt(
        prompt_config, task, build_item_index(task, edited)
    )
    assert changed.identity_hash != prompt.identity_hash
    again = resolve_prompt(prompt_config, task, build_item_index(task, rows))
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


def test_prompt_config_lists_shots_by_reference() -> None:
    config = load_config(
        DEFAULT_CONFIG_DIR / "prompts.toml", PromptsConfig
    ).olmes_rc
    assert {shot.split for shot in config.shots} == {"train"}
    assert len(config.shots) == 5
    assert config.item_format.unconditional_context
