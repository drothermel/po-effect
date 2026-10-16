from __future__ import annotations

from typing import TYPE_CHECKING

from po_effect.identity import FrozenContract
from po_effect.measurement.attempts import ScoredCandidatesAttempt
from po_effect.prompts import PromptSpec
from po_effect.tasks import AttemptKind

if TYPE_CHECKING:
    from po_effect.models import ModelSession
    from po_effect.prompts import MultipleChoiceRenderer
    from po_effect.tasks import Item, ItemIndex, ItemSet


class SingleCallSpec(FrozenContract):
    prompt: PromptSpec


class ScoredCandidatesSolver:
    solver_id = "scored_candidates"
    kind = AttemptKind.SCORED_CANDIDATES
    spec_type = SingleCallSpec

    def __init__(self, renderer: MultipleChoiceRenderer) -> None:
        self.renderer = renderer

    def validate(self, spec: SingleCallSpec, item_set: ItemSet) -> None:
        overlap = set(spec.prompt.shot_item_ids) & set(item_set.item_ids)
        if overlap:
            raise ValueError(
                f"shot items {sorted(overlap)!r} are in the evaluated item set"
            )

    def solve(
        self,
        item: Item,
        session: ModelSession,
        spec: SingleCallSpec,
        items: ItemIndex,
    ) -> ScoredCandidatesAttempt:
        rendered = self.renderer.render(
            spec.prompt,
            item.problem,
            tuple(items.get(i) for i in spec.prompt.shot_item_ids),
        )
        result = session.score(
            rendered.context,
            rendered.candidates,
            unconditional_context=rendered.unconditional_context,
        )
        return ScoredCandidatesAttempt(
            candidates=rendered.candidates,
            scores=result.scores,
            evidence=result.evidence,
        )
