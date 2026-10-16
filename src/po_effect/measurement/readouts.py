from __future__ import annotations

from enum import UNIQUE, StrEnum, auto, verify
from typing import TYPE_CHECKING

from pydantic import Field

from po_effect.identity import FrozenContract
from po_effect.measurement.attempts import (
    GenerationAttempt,
    ScoredCandidatesAttempt,
)
from po_effect.prompts import CandidateKind, choice_labels
from po_effect.tasks import AttemptKind

if TYPE_CHECKING:
    from collections.abc import Iterable

    from po_effect.measurement.attempts import Attempt
    from po_effect.models import CandidateScore, GenerationOutput
    from po_effect.tasks import MultipleChoiceProblem


@verify(UNIQUE)
class DecisionRule(StrEnum):
    RAW = auto()
    PER_TOKEN = auto()
    PER_CHAR = auto()
    PER_BYTE = auto()
    PMI = auto()


@verify(UNIQUE)
class GenerationRule(StrEnum):
    FIRST_LETTER = auto()
    CHOICE_TEXT_PREFIX = auto()


ReadoutId = DecisionRule | GenerationRule

MISSING_UNCONDITIONAL = "attempts have no unconditional log-likelihood"
GENERATION_STOP = "\n"
DIGIT_LABELS = "12345"


class IndexCandidate(FrozenContract):
    index: int = Field(ge=0)


class Abstention(FrozenContract):
    reason: str = Field(min_length=1)


Candidate = IndexCandidate | Abstention


def truncated_text(output: GenerationOutput) -> str:
    return output.text.split(GENERATION_STOP, 1)[0]


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


def _kind_reason(accepts: AttemptKind, kind: AttemptKind) -> str | None:
    if kind is accepts:
        return None
    return f"applies to {accepts.value} attempts, not {kind.value} attempts"


class LikelihoodArgmaxReadout:
    accepts = AttemptKind.SCORED_CANDIDATES

    def __init__(self, rule: DecisionRule) -> None:
        self.rule = rule
        self.readout_id = rule

    def inapplicable_reason(
        self,
        kind: AttemptKind,
        candidates: CandidateKind,
        attempts: Iterable[Attempt],
    ) -> str | None:
        del candidates
        reason = _kind_reason(self.accepts, kind)
        if reason is not None:
            return reason
        if self.rule is DecisionRule.PMI and any(
            score.unconditional_log_likelihood is None
            for attempt in attempts
            if isinstance(attempt, ScoredCandidatesAttempt)
            for score in attempt.scores
        ):
            return MISSING_UNCONDITIONAL
        return None

    def read(
        self, attempt: Attempt, problem: MultipleChoiceProblem
    ) -> IndexCandidate:
        if not isinstance(attempt, ScoredCandidatesAttempt):
            raise TypeError(
                "likelihood readouts read scored-candidate attempts"
            )
        if len(attempt.scores) != len(problem.choices):
            raise ValueError("attempt and problem choice counts differ")
        values = [decision_value(self.rule, score) for score in attempt.scores]
        return IndexCandidate(
            index=max(range(len(values)), key=values.__getitem__)
        )


class GenerationReadout:
    accepts = AttemptKind.GENERATION
    readout_id: GenerationRule
    candidates: CandidateKind

    def inapplicable_reason(
        self,
        kind: AttemptKind,
        candidates: CandidateKind,
        attempts: Iterable[Attempt],
    ) -> str | None:
        del attempts
        reason = _kind_reason(self.accepts, kind)
        if reason is not None:
            return reason
        if candidates is not self.candidates:
            return (
                f"applies to {self.candidates.value} formats, not "
                f"{candidates.value} formats"
            )
        return None

    def read(
        self, attempt: Attempt, problem: MultipleChoiceProblem
    ) -> Candidate:
        if not isinstance(attempt, GenerationAttempt):
            raise TypeError("generation readouts read generation attempts")
        text = truncated_text(attempt.output).strip()
        if not text:
            return Abstention(reason="truncated text is empty")
        return self.choose(text, problem)

    def choose(self, text: str, problem: MultipleChoiceProblem) -> Candidate:
        raise NotImplementedError


class FirstLetterReadout(GenerationReadout):
    readout_id = GenerationRule.FIRST_LETTER
    candidates = CandidateKind.CHOICE_LABEL

    def choose(self, text: str, problem: MultipleChoiceProblem) -> Candidate:
        labels = choice_labels(problem)
        first = text[0]
        if first in DIGIT_LABELS:
            position = DIGIT_LABELS.index(first)
        elif first.isalpha():
            upper = first.upper()
            position = labels.index(upper) if upper in labels else len(labels)
        else:
            return Abstention(
                reason=f"first character {first!r} is not a letter or digit"
            )
        if position >= len(labels):
            return Abstention(
                reason=(
                    f"first character {first!r} is outside the "
                    f"{len(labels)} choices"
                )
            )
        return IndexCandidate(index=position)


class ChoiceTextPrefixReadout(GenerationReadout):
    readout_id = GenerationRule.CHOICE_TEXT_PREFIX
    candidates = CandidateKind.CHOICE_TEXT

    def choose(self, text: str, problem: MultipleChoiceProblem) -> Candidate:
        matches = [
            (len(choice.strip()), -position)
            for position, choice in enumerate(problem.choices)
            if choice.strip() and text.startswith(choice.strip())
        ]
        if not matches:
            return Abstention(reason="text starts with no choice text")
        return IndexCandidate(index=-max(matches)[1])


Readout = (
    LikelihoodArgmaxReadout | FirstLetterReadout | ChoiceTextPrefixReadout
)


def readout_for(readout_id: ReadoutId) -> Readout:
    if isinstance(readout_id, DecisionRule):
        return LikelihoodArgmaxReadout(readout_id)
    match readout_id:
        case GenerationRule.FIRST_LETTER:
            return FirstLetterReadout()
        case GenerationRule.CHOICE_TEXT_PREFIX:
            return ChoiceTextPrefixReadout()


def readout_ids_for(kind: AttemptKind) -> tuple[ReadoutId, ...]:
    match kind:
        case AttemptKind.SCORED_CANDIDATES:
            return tuple(DecisionRule)
        case AttemptKind.GENERATION:
            return tuple(GenerationRule)
