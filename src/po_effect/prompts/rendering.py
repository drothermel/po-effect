from __future__ import annotations

from string import ascii_uppercase
from typing import TYPE_CHECKING

from po_effect.identity import FrozenContract
from po_effect.prompts.spec import CandidateKind

if TYPE_CHECKING:
    from collections.abc import Sequence

    from po_effect.prompts.spec import ItemFormat, PromptSpec
    from po_effect.tasks import Item, MultipleChoiceProblem

CHOICE_LABELS = ascii_uppercase
CHOICE_LINE_BREAK = "\n"
BLANK = "_"


class RenderedPrompt(FrozenContract):
    context: str
    candidates: tuple[str, ...]
    unconditional_context: str | None
    candidate_contexts: tuple[str, ...] | None = None


def choice_labels(problem: MultipleChoiceProblem) -> tuple[str, ...]:
    if len(problem.choices) > len(CHOICE_LABELS):
        raise ValueError(
            f"{len(problem.choices)} choices exceed the available labels"
        )
    return tuple(CHOICE_LABELS[: len(problem.choices)])


def answers(
    fmt: ItemFormat, problem: MultipleChoiceProblem
) -> tuple[str, ...]:
    if fmt.candidates is CandidateKind.CHOICE_LABEL:
        return choice_labels(problem)
    return problem.choices


def query(fmt: ItemFormat, problem: MultipleChoiceProblem) -> str:
    text = fmt.question_prefix + problem.question + fmt.answer_break
    if problem.passage is not None:
        text = problem.passage + fmt.answer_break + text
    if (
        fmt.choices_prefix is not None
        and fmt.label_prefix is not None
        and fmt.label_suffix is not None
    ):
        lines = (
            fmt.label_prefix + label + fmt.label_suffix + choice
            for label, choice in zip(
                choice_labels(problem), problem.choices, strict=True
            )
        )
        text += (
            fmt.choices_prefix
            + CHOICE_LINE_BREAK.join(lines)
            + fmt.answer_break
        )
    return text + fmt.answer_prefix


def blank_split(problem: MultipleChoiceProblem) -> tuple[str, str]:
    position = problem.question.find(BLANK)
    if position < 0:
        raise ValueError("blank-fill question has no blank")
    return problem.question[:position], problem.question[position + 1 :]


def blank_continuation(fmt: ItemFormat, problem: MultipleChoiceProblem) -> str:
    return fmt.continuation_prefix + blank_split(problem)[1].strip()


def blank_context(
    fmt: ItemFormat, problem: MultipleChoiceProblem, choice: str
) -> str:
    return fmt.question_prefix + blank_split(problem)[0] + choice


class MultipleChoiceRenderer:
    def render(
        self,
        spec: PromptSpec,
        problem: MultipleChoiceProblem,
        shots: Sequence[Item],
    ) -> RenderedPrompt:
        if tuple(shot.item_id for shot in shots) != spec.shot_item_ids:
            raise ValueError("shots differ from the prompt's shot item ids")
        fmt = spec.item_format

        def block(item: MultipleChoiceProblem, answer: str | None) -> str:
            if answer is None:
                return query(fmt, item)
            return query(fmt, item) + fmt.continuation_prefix + answer

        def shot_block(shot: Item) -> str:
            if fmt.candidates is CandidateKind.BLANK_FILL:
                gold = shot.problem.choices[shot.reference.gold_index]
                return blank_context(
                    fmt, shot.problem, gold
                ) + blank_continuation(fmt, shot.problem)
            return block(
                shot.problem,
                answers(fmt, shot.problem)[shot.reference.gold_index],
            )

        shot_blocks = [shot_block(shot) for shot in shots]
        if (
            spec.shots_header is not None
            and spec.shots_header_break is not None
        ):
            shot_blocks[0] = (
                spec.shots_header + spec.shots_header_break + shot_blocks[0]
            )
        blocks = [] if spec.preamble is None else [spec.preamble]
        blocks.extend(shot_blocks)
        if fmt.candidates is CandidateKind.BLANK_FILL:
            prefix = spec.shot_separator.join(
                [*blocks, fmt.question_prefix + blank_split(problem)[0]]
            )
            continuation = blank_continuation(fmt, problem)
            return RenderedPrompt(
                context=prefix,
                candidates=(continuation,) * len(problem.choices),
                unconditional_context=None,
                candidate_contexts=tuple(
                    prefix + choice for choice in problem.choices
                ),
            )
        blocks.append(block(problem, None))
        return RenderedPrompt(
            context=spec.shot_separator.join(blocks),
            candidates=tuple(
                fmt.continuation_prefix + answer
                for answer in answers(fmt, problem)
            ),
            unconditional_context=fmt.unconditional_context,
        )
