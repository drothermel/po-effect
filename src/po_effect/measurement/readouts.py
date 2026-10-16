from __future__ import annotations

from enum import UNIQUE, StrEnum, auto, verify
from typing import TYPE_CHECKING

from pydantic import Field

from po_effect.identity import FrozenContract
from po_effect.tasks import AttemptKind

if TYPE_CHECKING:
    from collections.abc import Iterable

    from po_effect.measurement.attempts import ScoredCandidatesAttempt
    from po_effect.models import CandidateScore
    from po_effect.tasks import MultipleChoiceProblem


@verify(UNIQUE)
class DecisionRule(StrEnum):
    RAW = auto()
    PER_TOKEN = auto()
    PER_CHAR = auto()
    PER_BYTE = auto()
    PMI = auto()


MISSING_UNCONDITIONAL = "attempts have no unconditional log-likelihood"


class IndexCandidate(FrozenContract):
    index: int = Field(ge=0)


def decision_value(rule: DecisionRule, score: CandidateScore) -> float:
    match rule:
        case DecisionRule.RAW:
            return score.log_likelihood
        case DecisionRule.PER_TOKEN:
            return score.log_likelihood / score.token_count
        case DecisionRule.PER_CHAR:
            return score.log_likelihood / score.char_count
        case DecisionRule.PER_BYTE:
            return score.log_likelihood / score.byte_count
        case DecisionRule.PMI:
            if score.unconditional_log_likelihood is None:
                raise ValueError(MISSING_UNCONDITIONAL)
            return score.log_likelihood - score.unconditional_log_likelihood


class LikelihoodArgmaxReadout:
    accepts = AttemptKind.SCORED_CANDIDATES

    def __init__(self, rule: DecisionRule) -> None:
        self.rule = rule
        self.readout_id = rule

    def inapplicable_reason(
        self, attempts: Iterable[ScoredCandidatesAttempt]
    ) -> str | None:
        if self.rule is DecisionRule.PMI and any(
            score.unconditional_log_likelihood is None
            for attempt in attempts
            for score in attempt.scores
        ):
            return MISSING_UNCONDITIONAL
        return None

    def read(
        self,
        attempt: ScoredCandidatesAttempt,
        problem: MultipleChoiceProblem,
    ) -> IndexCandidate:
        if len(attempt.scores) != len(problem.choices):
            raise ValueError("attempt and problem choice counts differ")
        values = [decision_value(self.rule, score) for score in attempt.scores]
        return IndexCandidate(
            index=max(range(len(values)), key=values.__getitem__)
        )
