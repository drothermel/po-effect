from __future__ import annotations

from dataclasses import dataclass
from enum import UNIQUE, StrEnum, auto, verify
from typing import TYPE_CHECKING, ClassVar

from dr_hf import DatasetPin, origin_subsets
from dr_serialize import IdentityDocument, build_identity_document, json_hash
from pydantic import Field, model_validator

from po_effect.identity import FrozenContract, Identified

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from dr_hf import NamedSubset, SourceRow

    from po_effect.tasks.registry import Task

ITEM_SCHEMA = "po_effect.item"
ITEM_SCHEMA_VERSION = 2


@verify(UNIQUE)
class AttemptKind(StrEnum):
    SCORE = auto()
    GENERATE = auto()


@verify(UNIQUE)
class ItemSetRole(StrEnum):
    REPORT = auto()


class MultipleChoiceProblem(FrozenContract):
    question: str
    choices: tuple[str, ...] = Field(min_length=1)
    passage: str | None = None


class IndexReference(FrozenContract):
    gold_index: int = Field(ge=0)


class Item(Identified):
    identity_schema: ClassVar[str] = ITEM_SCHEMA
    identity_schema_version: ClassVar[int] = ITEM_SCHEMA_VERSION
    task_id: str
    task_version: int = Field(ge=1)
    native_id: str
    problem: MultipleChoiceProblem
    reference: IndexReference

    @model_validator(mode="after")
    def validate_reference(self) -> Item:
        if self.reference.gold_index >= len(self.problem.choices):
            raise ValueError("answer key beyond the choices")
        return self

    def identity_document(self) -> IdentityDocument:
        return build_identity_document(
            schema=ITEM_SCHEMA,
            schema_version=ITEM_SCHEMA_VERSION,
            payload={
                "task_id": self.task_id,
                "task_version": self.task_version,
                "native_id": self.native_id,
                "content_hash": self.content_hash,
            },
        )

    @property
    def content_hash(self) -> str:
        return str(
            json_hash(
                {
                    "problem": self.problem.model_dump(mode="json"),
                    "reference": self.reference.model_dump(mode="json"),
                }
            )
        )

    @property
    def item_id(self) -> str:
        return self.identity_hash


class ItemSetDerivation(FrozenContract):
    method: str
    seed: int | None = None
    parent: str | None = None
    excluded_item_ids: tuple[str, ...] = ()


class ItemSet(FrozenContract):
    name: str
    task_id: str
    task_version: int
    role: ItemSetRole
    item_ids: tuple[str, ...] = Field(min_length=1)
    derivation: ItemSetDerivation


class TaskConfig(FrozenContract):
    task_id: str
    version: int = Field(ge=1)
    source: DatasetPin
    split: str
    splits: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_split(self) -> TaskConfig:
        if self.split not in self.splits:
            raise ValueError(
                f"evaluation split {self.split!r} is not a loaded split"
            )
        return self


class SourceTask[Config: TaskConfig]:
    admissible_kinds = frozenset({AttemptKind.SCORE, AttemptKind.GENERATE})

    def __init__(self, config: Config) -> None:
        self.config = config
        self.task_id = config.task_id
        self.version = config.version
        self.source = config.source
        self.splits = config.splits

    def build(self, row: SourceRow) -> Item:
        if row.pin != self.source:
            raise ValueError("source row belongs to a different dataset pin")
        problem, gold = self.parse(row)
        return Item(
            task_id=self.task_id,
            task_version=self.version,
            native_id=row.native_id,
            problem=problem,
            reference=IndexReference(gold_index=gold),
        )

    def parse(self, row: SourceRow) -> tuple[MultipleChoiceProblem, int]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ItemIndex:
    items: Mapping[str, Item]
    splits: Mapping[str, NamedSubset]
    item_ids: Mapping[str, str]

    def get(self, item_id: str) -> Item:
        return self.items[item_id]

    def all(self) -> tuple[Item, ...]:
        return tuple(self.items.values())

    def lookup(self, split: str, native_id: str) -> Item:
        subset = self.splits.get(split)
        if subset is None:
            raise ValueError(
                f"unknown split {split!r}; available: {sorted(self.splits)!r}"
            )
        if native_id not in subset.native_ids:
            raise ValueError(
                f"unknown native id {native_id!r} in split {split!r}"
            )
        return self.items[self.item_ids[native_id]]


def build_item_index(task: Task, rows: Sequence[SourceRow]) -> ItemIndex:
    splits = {
        subset.key.name: subset for subset in origin_subsets(task.source, rows)
    }
    items = {}
    item_ids = {}
    for row in rows:
        item = task.build(row)
        items[item.item_id] = item
        item_ids[row.native_id] = item.item_id
    return ItemIndex(items, splits, item_ids)


def prefix_item_set(item_set: ItemSet, limit: int) -> ItemSet:
    if not 0 < limit <= len(item_set.item_ids):
        raise ValueError(f"limit {limit} outside the item set")
    return ItemSet(
        name=f"{item_set.name}_first_{limit}",
        task_id=item_set.task_id,
        task_version=item_set.task_version,
        role=item_set.role,
        item_ids=item_set.item_ids[:limit],
        derivation=ItemSetDerivation(
            method="prefix",
            parent=str(json_hash(item_set.model_dump(mode="json"))),
        ),
    )
