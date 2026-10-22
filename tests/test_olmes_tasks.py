from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from conftest import CPU_CONDITIONS, fake_backend
from dr_hf import SourceRow
from dr_providers.local.backend import ScoredContinuation
from dr_providers.surfaces.testing.fake_backend import (
    FakeLocalBackend,
    GenerateCall,
    ScoreCall,
)

from po_effect.config import DEFAULT_CONFIG_DIR
from po_effect.measurement import (
    DecisionRule,
    LikelihoodArgmaxReadout,
    ScoredCandidatesAttempt,
    ScoredCandidatesSolver,
    SingleCallSpec,
)
from po_effect.models import ModelSession, load_model_handle
from po_effect.prompts import (
    MultipleChoiceRenderer,
    RenderedPrompt,
    load_prompt_config,
    resolve_prompt,
)
from po_effect.tasks import (
    TaskName,
    build_item_index,
    hellaswag_text,
    load_task,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from po_effect.tasks import Task

type Fields = Callable[[int], dict[str, object]]
SHOT_COUNT = 5


def arc_fields(index: int) -> dict[str, object]:
    return {
        "id": f"arc-{index}",
        "question": f"Which is {index}?",
        "choices": {"text": ["one", "two"], "label": ["A", "B"]},
        "answerKey": "B",
    }


def hellaswag_fields(index: int) -> dict[str, object]:
    return {
        "ind": index,
        "activity_label": "Health",
        "ctx_a": f"[header] Step {index} [title] Rest.",
        "ctx_b": "then sleep",
        "ctx": "unused",
        "endings": ["Wake up.", "Run."],
        "source_id": "wikihow~1",
        "split": "train",
        "split_type": "indomain",
        "label": "1",
    }


def piqa_fields(index: int) -> dict[str, object]:
    return {"goal": f"Goal {index}", "sol1": "a", "sol2": "b", "label": 1}


def boolq_fields(index: int) -> dict[str, object]:
    return {
        "question": f"is it {index}",
        "passage": f"Passage {index}",
        "idx": index,
        "label": 0,
    }


def openbookqa_fields(index: int) -> dict[str, object]:
    return {
        "id": f"obqa-{index}",
        "question_stem": f"Stem {index}",
        "choices": {"text": ["c", "d"], "label": ["A", "B"]},
        "answerKey": "B",
    }


def csqa_fields(index: int) -> dict[str, object]:
    return {
        "id": f"csqa-{index}",
        "question": f"Where {index}?",
        "question_concept": "place",
        "choices": {"text": ["e", "f"], "label": ["A", "B"]},
        "answerKey": "B",
    }


CASES: Mapping[TaskName, tuple[str, Fields, str, str, tuple[str, ...]]] = {
    TaskName.ARC_EASY: (
        "olmes_rc_arc_easy",
        arc_fields,
        "Question: Which is 0?\nAnswer: two",
        "Question: Which is 9?\nAnswer:",
        (" one", " two"),
    ),
    TaskName.HELLASWAG: (
        "olmes_rc_hellaswag",
        hellaswag_fields,
        "Health: Step 0. Rest. Then sleep Run.",
        "Health: Step 9. Rest. Then sleep",
        (" Wake up.", " Run."),
    ),
    TaskName.PIQA: (
        "olmes_rc_piqa",
        piqa_fields,
        "Goal: Goal 0\nAnswer: b",
        "Goal: Goal 9\nAnswer:",
        (" a", " b"),
    ),
    TaskName.BOOLQ: (
        "olmes_rc_boolq",
        boolq_fields,
        "Passage 0\nQuestion: is it 0?\nAnswer: no",
        "Passage 9\nQuestion: is it 9?\nAnswer:",
        (" yes", " no"),
    ),
    TaskName.OPENBOOKQA: (
        "olmes_rc_openbookqa",
        openbookqa_fields,
        "Question: Stem 0\nAnswer: d",
        "Question: Stem 9\nAnswer:",
        (" c", " d"),
    ),
    TaskName.CSQA: (
        "olmes_rc_csqa",
        csqa_fields,
        "Question: Where 0?\nAnswer: f",
        "Question: Where 9?\nAnswer:",
        (" e", " f"),
    ),
}


def source_row(
    task: Task, split: str, native_id: str, fields: dict[str, object]
) -> SourceRow:
    if "id" in fields:
        fields = {**fields, "id": native_id}
    return SourceRow(
        pin=task.source,
        origin_split=split,
        native_id=native_id,
        fields=fields,
    )


@pytest.mark.parametrize("name", list(CASES))
def test_olmes_task_renders_one_item(name: TaskName) -> None:
    prompt_name, fields, first_shot, query, candidates = CASES[name]
    task = load_task(name)
    config = load_prompt_config(DEFAULT_CONFIG_DIR, prompt_name)
    assert len(config.shots) == SHOT_COUNT
    assert {shot.split for shot in config.shots} <= set(task.splits)
    shot_rows = [
        source_row(task, shot.split, shot.native_id, fields(position))
        for position, shot in enumerate(config.shots)
    ]
    evaluated = source_row(task, task.config.split, "eval-9", fields(9))
    items = build_item_index(task, (*shot_rows, evaluated))
    prompt = resolve_prompt(prompt_name, config, task, items)
    item = items.lookup(task.config.split, "eval-9")
    rendered = MultipleChoiceRenderer().render(
        prompt, item.problem, tuple(items.get(i) for i in prompt.shot_item_ids)
    )
    blocks = rendered.context.split(prompt.shot_separator)
    assert len(blocks) == SHOT_COUNT + 1
    assert blocks[0] == first_shot
    assert blocks[-1] == query
    assert rendered.candidates == candidates
    assert item.task_id == task.task_id


def test_olmes_item_ids_hash_content() -> None:
    task = load_task(TaskName.PIQA)
    first = task.build(source_row(task, "validation", "v:0", piqa_fields(0)))
    again = task.build(source_row(task, "validation", "v:0", piqa_fields(0)))
    changed = task.build(source_row(task, "validation", "v:0", piqa_fields(1)))
    assert first.item_id == again.item_id
    assert first.item_id != changed.item_id


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "[header] How to cope [title] Put off plans.",
            " How to cope. Put off plans.",
        ),
        ("Done. [title] Next", "Done. Next"),
        ("[step] Rest.", " Rest."),
    ],
)
def test_hellaswag_text_matches_olmes_preprocess(
    text: str, expected: str
) -> None:
    assert hellaswag_text(text) == expected


@pytest.mark.parametrize(
    ("name", "fields", "message"),
    [
        (TaskName.BOOLQ, {**boolq_fields(0), "label": 2}, "unknown label"),
        (
            TaskName.OPENBOOKQA,
            {**openbookqa_fields(0), "answerKey": "E"},
            "unknown answer key",
        ),
        (
            TaskName.CSQA,
            {**csqa_fields(0), "answerKey": ""},
            "unknown answer key",
        ),
    ],
)
def test_olmes_tasks_reject_unlabeled_rows(
    name: TaskName, fields: dict[str, object], message: str
) -> None:
    task = load_task(name)
    with pytest.raises(ValueError, match=message):
        task.build(source_row(task, task.config.split, "x", fields))


def winogrande_fields(index: int) -> dict[str, object]:
    return {
        "sentence": f"Ann {index} helped Bo because _ was kind .",
        "option1": "Ann",
        "option2": "Bo",
        "answer": "2",
    }


def winogrande_run() -> tuple[
    RenderedPrompt, ScoredCandidatesAttempt, list[GenerateCall | ScoreCall]
]:
    task = load_task(TaskName.WINOGRANDE)
    config = load_prompt_config(DEFAULT_CONFIG_DIR, "olmes_rc_winogrande")
    rows = [
        source_row(task, shot.split, shot.native_id, winogrande_fields(i))
        for i, shot in enumerate(config.shots)
    ]
    rows.append(source_row(task, "validation", "eval-9", winogrande_fields(9)))
    items = build_item_index(task, rows)
    prompt = resolve_prompt("olmes_rc_winogrande", config, task, items)
    item = items.lookup("validation", "eval-9")
    shots = tuple(items.get(i) for i in prompt.shot_item_ids)
    rendered = MultipleChoiceRenderer().render(prompt, item.problem, shots)
    handle = load_model_handle()
    backend = FakeLocalBackend(
        execution=fake_backend(handle, CPU_CONDITIONS).execution(),
        results=[
            (ScoredContinuation(-4.0, 3, None, 3),),
            (ScoredContinuation(-1.0, 3, None, 3),),
        ],
    )
    with ModelSession(handle, CPU_CONDITIONS, None, backend) as session:
        attempt = ScoredCandidatesSolver(MultipleChoiceRenderer()).solve(
            item, session, SingleCallSpec(prompt=prompt), items
        )
    return rendered, attempt, backend.calls


def test_winogrande_scores_each_option_context() -> None:
    rendered, attempt, calls = winogrande_run()
    blocks = rendered.context.split("\n\n")
    assert len(blocks) == SHOT_COUNT + 1
    assert blocks[0] == "Ann 0 helped Bo because Bo was kind ."
    assert blocks[-1] == "Ann 9 helped Bo because "
    assert rendered.candidates == (" was kind .", " was kind .")
    assert rendered.unconditional_context is None
    scored = [call for call in calls if isinstance(call, ScoreCall)]
    assert len(scored) == len(calls) == 2
    assert [call.context for call in scored] == [
        rendered.context + "Ann",
        rendered.context + "Bo",
    ]
    assert [call.continuations for call in scored] == [(" was kind .",)] * 2
    assert [score.log_likelihood for score in attempt.scores] == [-4.0, -1.0]
    readout = LikelihoodArgmaxReadout(DecisionRule.RAW)
    task = load_task(TaskName.WINOGRANDE)
    item = task.build(
        source_row(task, "validation", "eval-9", winogrande_fields(9))
    )
    assert (
        readout.read(attempt, item.problem).index == item.reference.gold_index
    )
