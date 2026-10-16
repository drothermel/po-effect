from __future__ import annotations

import re
from enum import UNIQUE, StrEnum, auto, verify
from typing import TYPE_CHECKING

from pydantic import Field
from transformers.utils.chat_parsing import parse_response

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
    from po_effect.models import CandidateScore, ResponseTemplate
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
    ANSWER_LETTER = auto()


@verify(UNIQUE)
class ReasoningVisibility(StrEnum):
    NONE = auto()
    INLINE = auto()
    HIDDEN = auto()


ReadoutId = DecisionRule | GenerationRule

MISSING_UNCONDITIONAL = "attempts have no unconditional log-likelihood"
GENERATION_STOP = "\n"
DIGIT_LABELS = "12345"
REASONING_FIELD = "reasoning"
ANSWER_FIELD = "answer"
ANSWER_PHRASE = re.compile(
    r"answer\s*(?:is|:)\s*:?\s*\(?([a-z])\)?(?![a-z])", re.IGNORECASE
)
WHOLE_LETTER = re.compile(r"\(?([a-z])\)?[.:)]?", re.IGNORECASE)
LETTER_TOKEN = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")


class IndexCandidate(FrozenContract):
    index: int = Field(ge=0)


class Abstention(FrozenContract):
    reason: str = Field(min_length=1)


Candidate = IndexCandidate | Abstention


class ResponseSplit(FrozenContract):
    reasoning: str | None
    answer: str
    reasoning_visibility: ReasoningVisibility


def parser_template(template: ResponseTemplate) -> dict[str, object]:
    return {
        "version": 1,
        "start_anchor": template.start_anchor,
        "defaults": {REASONING_FIELD: None, ANSWER_FIELD: ""},
        "fields": {
            REASONING_FIELD: {
                "open": template.reasoning_open,
                "close_pattern": re.escape(template.reasoning_close) + r"\n*",
                "content_args": {"strip": False},
            },
            ANSWER_FIELD: {"content_args": {"strip": False}},
        },
    }


def split_response(
    text: str, template: ResponseTemplate | None
) -> ResponseSplit:
    if template is None:
        return ResponseSplit(
            reasoning=None,
            answer=text,
            reasoning_visibility=ReasoningVisibility.NONE,
        )
    parsed = parse_response(text, parser_template(template), prefix="")
    reasoning = parsed.get(REASONING_FIELD)
    answer = parsed.get(ANSWER_FIELD)
    if not isinstance(answer, str) or not (
        reasoning is None or isinstance(reasoning, str)
    ):
        raise TypeError("response parser returned non-text fields")
    return ResponseSplit(
        reasoning=reasoning,
        answer=answer,
        reasoning_visibility=(
            ReasoningVisibility.NONE
            if reasoning is None
            else ReasoningVisibility.INLINE
        ),
    )


def truncated_text(text: str) -> str:
    return text.split(GENERATION_STOP, 1)[0]


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
    accepts = AttemptKind.SCORE

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

    def reasoning_visibility(self, attempt: Attempt) -> ReasoningVisibility:
        del attempt
        return ReasoningVisibility.NONE


class GenerationReadout:
    accepts = AttemptKind.GENERATE
    readout_id: GenerationRule
    candidates: CandidateKind

    def __init__(self, response_template: ResponseTemplate | None) -> None:
        self.response_template = response_template

    def split(self, attempt: Attempt) -> ResponseSplit:
        if not isinstance(attempt, GenerationAttempt):
            raise TypeError("generation readouts read generation attempts")
        return split_response(attempt.output.text, self.response_template)

    def reasoning_visibility(self, attempt: Attempt) -> ReasoningVisibility:
        return self.split(attempt).reasoning_visibility

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
        text = truncated_text(self.split(attempt).answer).strip()
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


def _label_candidate(
    letter: str, labels: tuple[str, ...], source: str
) -> Candidate:
    upper = letter.upper()
    if upper not in labels:
        return Abstention(
            reason=(f"{source} {upper!r} is outside the {len(labels)} choices")
        )
    return IndexCandidate(index=labels.index(upper))


class AnswerLetterReadout(GenerationReadout):
    readout_id = GenerationRule.ANSWER_LETTER
    candidates = CandidateKind.CHOICE_LABEL

    def choose(self, text: str, problem: MultipleChoiceProblem) -> Candidate:
        labels = choice_labels(problem)
        phrase = ANSWER_PHRASE.search(text)
        if phrase is not None:
            return _label_candidate(phrase[1], labels, "answer letter")
        whole = WHOLE_LETTER.fullmatch(text)
        if whole is not None:
            return _label_candidate(whole[1], labels, "answer letter")
        tokens = [match[1] for match in LETTER_TOKEN.finditer(text)]
        for token in tokens:
            if token in labels:
                return IndexCandidate(index=labels.index(token))
        if tokens:
            return Abstention(
                reason=(
                    f"letters {tokens!r} are outside the {len(labels)} choices"
                )
            )
        return Abstention(reason="text contains no choice letter")


Readout = (
    LikelihoodArgmaxReadout
    | FirstLetterReadout
    | ChoiceTextPrefixReadout
    | AnswerLetterReadout
)


def readout_for(
    readout_id: ReadoutId, response_template: ResponseTemplate | None
) -> Readout:
    if isinstance(readout_id, DecisionRule):
        return LikelihoodArgmaxReadout(readout_id)
    match readout_id:
        case GenerationRule.FIRST_LETTER:
            return FirstLetterReadout(response_template)
        case GenerationRule.CHOICE_TEXT_PREFIX:
            return ChoiceTextPrefixReadout(response_template)
        case GenerationRule.ANSWER_LETTER:
            return AnswerLetterReadout(response_template)


def readout_ids_for(kind: AttemptKind) -> tuple[ReadoutId, ...]:
    match kind:
        case AttemptKind.SCORE:
            return tuple(DecisionRule)
        case AttemptKind.GENERATE:
            return tuple(GenerationRule)
