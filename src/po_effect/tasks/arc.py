from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from po_effect.identity import FrozenContract
from po_effect.tasks.items import (
    MultipleChoiceProblem,
    SourceTask,
    TaskConfig,
)

if TYPE_CHECKING:
    from dr_hf import SourceRow


class NumericAnswerKeys(FrozenContract):
    one: str = Field(alias="1")
    two: str = Field(alias="2")
    three: str = Field(alias="3")
    four: str = Field(alias="4")
    five: str = Field(alias="5")


class ArcConfig(TaskConfig):
    answer_letters: str
    numeric_answer_keys: NumericAnswerKeys


class ArcChoices(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    text: tuple[str, ...] = Field(min_length=1)
    label: tuple[str, ...]


class ArcRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    question: str
    choices: ArcChoices
    answer_key: str = Field(alias="answerKey")


class ArcTask(SourceTask[ArcConfig]):
    def parse(self, row: SourceRow) -> tuple[MultipleChoiceProblem, int]:
        fields = ArcRow.model_validate(row.fields)
        if fields.id != row.native_id:
            raise ValueError("native id differs from source row id")
        letter = self.config.numeric_answer_keys.model_dump(by_alias=True).get(
            fields.answer_key, fields.answer_key
        )
        if letter not in self.config.answer_letters:
            raise ValueError(f"unknown answer key {fields.answer_key!r}")
        return (
            MultipleChoiceProblem(
                question=fields.question, choices=fields.choices.text
            ),
            self.config.answer_letters.index(letter),
        )
