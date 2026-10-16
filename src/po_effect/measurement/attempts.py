from __future__ import annotations

from pydantic import Field, model_validator

from po_effect.identity import FrozenContract
from po_effect.models import CandidateScore, EvidenceRef
from po_effect.tasks import AttemptKind


class ScoredCandidatesAttempt(FrozenContract):
    kind: AttemptKind = AttemptKind.SCORED_CANDIDATES
    candidates: tuple[str, ...] = Field(min_length=1)
    scores: tuple[CandidateScore, ...]
    evidence: tuple[EvidenceRef, ...]

    @model_validator(mode="after")
    def validate_score_count(self) -> ScoredCandidatesAttempt:
        if len(self.candidates) != len(self.scores):
            raise ValueError("candidate and score counts differ")
        return self
