from __future__ import annotations

from typing import TYPE_CHECKING

from po_effect.identity import FrozenContract

if TYPE_CHECKING:
    from collections.abc import Sequence

    from po_effect.prompts.spec import PromptSpec
    from po_effect.tasks import Item, MultipleChoiceProblem


class RenderedPrompt(FrozenContract):
    context: str
    candidates: tuple[str, ...]
    unconditional_context: str


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

        def block(question: str, answer: str | None) -> str:
            query = (
                fmt.question_prefix
                + question
                + fmt.answer_break
                + fmt.answer_prefix
            )
            if answer is None:
                return query
            return query + fmt.continuation_prefix + answer

        blocks = [] if spec.preamble is None else [spec.preamble]
        blocks.extend(
            block(
                shot.problem.question,
                shot.problem.choices[shot.reference.gold_index],
            )
            for shot in shots
        )
        blocks.append(block(problem.question, None))
        return RenderedPrompt(
            context=spec.shot_separator.join(blocks),
            candidates=tuple(
                fmt.continuation_prefix + choice for choice in problem.choices
            ),
            unconditional_context=fmt.unconditional_context,
        )
