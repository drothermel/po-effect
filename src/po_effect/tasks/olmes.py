from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from po_effect.tasks.arc import ArcChoices
from po_effect.tasks.items import MultipleChoiceProblem, SourceTask, TaskConfig

if TYPE_CHECKING:
    from dr_hf import SourceRow

HELLASWAG_TITLE = re.compile(r"\.? \[title\]")
HELLASWAG_BRACKETS = re.compile(r"\[.*?\]")
OPENBOOKQA_LETTERS = "ABCD"
CSQA_LETTERS = "ABCDE"
BOOLQ_CHOICES = ("yes", "no")
WINOGRANDE_ANSWERS = ("1", "2")


class SourceFields(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HellaSwagRow(SourceFields):
    ind: int
    activity_label: str
    ctx_a: str
    ctx_b: str
    ctx: str
    endings: tuple[str, ...] = Field(min_length=1)
    source_id: str
    split: str
    split_type: str
    label: str


class PiqaRow(SourceFields):
    goal: str
    sol1: str
    sol2: str
    label: int


class WinoGrandeRow(SourceFields):
    sentence: str
    option1: str
    option2: str
    answer: str


class BoolQRow(SourceFields):
    question: str
    passage: str
    idx: int
    label: int


class OpenBookQARow(SourceFields):
    id: str
    question_stem: str
    choices: ArcChoices
    answer_key: str = Field(alias="answerKey")


class CsqaRow(SourceFields):
    id: str
    question: str
    question_concept: str
    choices: ArcChoices
    answer_key: str = Field(alias="answerKey")


def hellaswag_text(text: str) -> str:
    text = HELLASWAG_TITLE.sub(". ", text.strip())
    return HELLASWAG_BRACKETS.sub("", text).replace("  ", " ")


def letter_index(letters: str, key: str) -> int:
    if len(key) != 1 or key not in letters:
        raise ValueError(f"unknown answer key {key!r}")
    return letters.index(key)


def require_native_id(row: SourceRow, value: str) -> None:
    if value != row.native_id:
        raise ValueError("native id differs from source row id")


class HellaSwagTask(SourceTask[TaskConfig]):
    def parse(self, row: SourceRow) -> tuple[MultipleChoiceProblem, int]:
        fields = HellaSwagRow.model_validate(row.fields)
        context = fields.ctx_a + " " + fields.ctx_b.capitalize()
        return (
            MultipleChoiceProblem(
                question=hellaswag_text(
                    fields.activity_label + ": " + context
                ),
                choices=tuple(map(hellaswag_text, fields.endings)),
            ),
            int(fields.label),
        )


class PiqaTask(SourceTask[TaskConfig]):
    def parse(self, row: SourceRow) -> tuple[MultipleChoiceProblem, int]:
        fields = PiqaRow.model_validate(row.fields)
        return (
            MultipleChoiceProblem(
                question=fields.goal, choices=(fields.sol1, fields.sol2)
            ),
            fields.label,
        )


class WinoGrandeTask(SourceTask[TaskConfig]):
    def parse(self, row: SourceRow) -> tuple[MultipleChoiceProblem, int]:
        fields = WinoGrandeRow.model_validate(row.fields)
        if fields.answer not in WINOGRANDE_ANSWERS:
            raise ValueError(f"unknown answer {fields.answer!r}")
        return (
            MultipleChoiceProblem(
                question=fields.sentence,
                choices=(fields.option1, fields.option2),
            ),
            WINOGRANDE_ANSWERS.index(fields.answer),
        )


class BoolQTask(SourceTask[TaskConfig]):
    def parse(self, row: SourceRow) -> tuple[MultipleChoiceProblem, int]:
        fields = BoolQRow.model_validate(row.fields)
        if fields.label not in {0, 1}:
            raise ValueError(f"unknown label {fields.label!r}")
        return (
            MultipleChoiceProblem(
                question=fields.question + "?",
                choices=BOOLQ_CHOICES,
                passage=fields.passage,
            ),
            1 - fields.label,
        )


class OpenBookQATask(SourceTask[TaskConfig]):
    def parse(self, row: SourceRow) -> tuple[MultipleChoiceProblem, int]:
        fields = OpenBookQARow.model_validate(row.fields)
        require_native_id(row, fields.id)
        return (
            MultipleChoiceProblem(
                question=fields.question_stem, choices=fields.choices.text
            ),
            letter_index(OPENBOOKQA_LETTERS, fields.answer_key.strip()),
        )


class CsqaTask(SourceTask[TaskConfig]):
    def parse(self, row: SourceRow) -> tuple[MultipleChoiceProblem, int]:
        fields = CsqaRow.model_validate(row.fields)
        require_native_id(row, fields.id)
        return (
            MultipleChoiceProblem(
                question=fields.question, choices=fields.choices.text
            ),
            letter_index(CSQA_LETTERS, fields.answer_key),
        )
