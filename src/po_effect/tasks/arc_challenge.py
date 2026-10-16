from __future__ import annotations

from typing import TYPE_CHECKING

from dr_hf import DatasetPin
from pydantic import BaseModel, ConfigDict, Field

from po_effect.config import DEFAULT_CONFIG_DIR, load_config
from po_effect.identity import FrozenContract
from po_effect.tasks.items import (
    AttemptKind,
    IndexReference,
    Item,
    MultipleChoiceProblem,
)

if TYPE_CHECKING:
    from pathlib import Path

    from dr_hf import SourceRow


class NumericAnswerKeys(FrozenContract):
    one: str = Field(alias="1")
    two: str = Field(alias="2")
    three: str = Field(alias="3")
    four: str = Field(alias="4")
    five: str = Field(alias="5")


class ArcChallengeConfig(FrozenContract):
    task_id: str
    version: int = Field(ge=1)
    source: DatasetPin
    split: str
    answer_letters: str
    numeric_answer_keys: NumericAnswerKeys


class TasksConfig(FrozenContract):
    arc_challenge: ArcChallengeConfig


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


class ArcChallengeTask:
    admissible_kinds = frozenset({AttemptKind.SCORED_CANDIDATES})

    def __init__(self, config: ArcChallengeConfig) -> None:
        self.config = config
        self.task_id = config.task_id
        self.version = config.version
        self.source = config.source

    def build(self, row: SourceRow) -> Item:
        if row.pin != self.source:
            raise ValueError("source row belongs to a different dataset pin")
        fields = ArcRow.model_validate(row.fields)
        if fields.id != row.native_id:
            raise ValueError("native id differs from source row id")
        letter = self.config.numeric_answer_keys.model_dump(by_alias=True).get(
            fields.answer_key, fields.answer_key
        )
        if letter not in self.config.answer_letters:
            raise ValueError(f"unknown answer key {fields.answer_key!r}")
        return Item(
            task_id=self.task_id,
            task_version=self.version,
            native_id=row.native_id,
            problem=MultipleChoiceProblem(
                question=fields.question, choices=fields.choices.text
            ),
            reference=IndexReference(
                gold_index=self.config.answer_letters.index(letter)
            ),
        )


def load_arc_challenge(
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> ArcChallengeTask:
    config = load_config(config_dir / "tasks.toml", TasksConfig)
    return ArcChallengeTask(config.arc_challenge)
