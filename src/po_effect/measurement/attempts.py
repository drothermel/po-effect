from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from po_effect.identity import FrozenContract
from po_effect.models import CandidateScore, EvidenceRef, GenerationOutput
from po_effect.tasks import AttemptKind


class ScoredCandidatesAttempt(FrozenContract):
    kind: Literal[AttemptKind.SCORED_CANDIDATES] = (
        AttemptKind.SCORED_CANDIDATES
    )
    candidates: tuple[str, ...] = Field(min_length=1)
    scores: tuple[CandidateScore, ...]
    evidence: tuple[EvidenceRef, ...]

    @model_validator(mode="after")
    def validate_score_count(self) -> ScoredCandidatesAttempt:
        if len(self.candidates) != len(self.scores):
            raise ValueError("candidate and score counts differ")
        return self


class GenerationAttempt(FrozenContract):
    kind: Literal[AttemptKind.GENERATION] = AttemptKind.GENERATION
    output: GenerationOutput
    evidence: tuple[EvidenceRef, ...]


Attempt = Annotated[
    ScoredCandidatesAttempt | GenerationAttempt, Field(discriminator="kind")
]
