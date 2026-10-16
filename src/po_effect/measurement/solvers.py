from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from po_effect.identity import FrozenContract
from po_effect.measurement.attempts import (
    GenerationAttempt,
    ScoredCandidatesAttempt,
)
from po_effect.prompts import PromptSpec
from po_effect.tasks import AttemptKind

if TYPE_CHECKING:
    from po_effect.models import ModelSession
    from po_effect.prompts import MultipleChoiceRenderer, RenderedPrompt
    from po_effect.tasks import Item, ItemIndex, ItemSet


class SingleCallSpec(FrozenContract):
    prompt: PromptSpec


class GenerationSpec(FrozenContract):
    prompt: PromptSpec
    max_new_tokens: int = Field(gt=0)


SolverSpec = SingleCallSpec | GenerationSpec


def _require_spec[Spec: FrozenContract](
    spec: SolverSpec, spec_type: type[Spec]
) -> Spec:
    if not isinstance(spec, spec_type):
        raise TypeError(
            f"solver requires {spec_type.__name__}, got {type(spec).__name__}"
        )
    return spec


def _validate_shots(spec: SolverSpec, item_set: ItemSet) -> None:
    overlap = set(spec.prompt.shot_item_ids) & set(item_set.item_ids)
    if overlap:
        raise ValueError(
            f"shot items {sorted(overlap)!r} are in the evaluated item set"
        )


def _render(
    renderer: MultipleChoiceRenderer,
    item: Item,
    spec: SolverSpec,
    items: ItemIndex,
) -> RenderedPrompt:
    return renderer.render(
        spec.prompt,
        item.problem,
        tuple(items.get(i) for i in spec.prompt.shot_item_ids),
    )


class ScoredCandidatesSolver:
    solver_id = "scored_candidates"
    kind = AttemptKind.SCORED_CANDIDATES
    spec_type = SingleCallSpec

    def __init__(self, renderer: MultipleChoiceRenderer) -> None:
        self.renderer = renderer

    def validate(self, spec: SolverSpec, item_set: ItemSet) -> None:
        _validate_shots(_require_spec(spec, self.spec_type), item_set)

    def solve(
        self,
        item: Item,
        session: ModelSession,
        spec: SolverSpec,
        items: ItemIndex,
    ) -> ScoredCandidatesAttempt:
        rendered = _render(
            self.renderer, item, _require_spec(spec, self.spec_type), items
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


class GenerationSolver:
    solver_id = "generation"
    kind = AttemptKind.GENERATION
    spec_type = GenerationSpec

    def __init__(self, renderer: MultipleChoiceRenderer) -> None:
        self.renderer = renderer

    def validate(self, spec: SolverSpec, item_set: ItemSet) -> None:
        _validate_shots(_require_spec(spec, self.spec_type), item_set)

    def solve(
        self,
        item: Item,
        session: ModelSession,
        spec: SolverSpec,
        items: ItemIndex,
    ) -> GenerationAttempt:
        generation = _require_spec(spec, self.spec_type)
        rendered = _render(self.renderer, item, generation, items)
        result = session.generate(rendered.context, generation.max_new_tokens)
        return GenerationAttempt(
            output=result.output, evidence=result.evidence
        )


Solver = ScoredCandidatesSolver | GenerationSolver
