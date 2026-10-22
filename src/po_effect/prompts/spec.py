from __future__ import annotations

from enum import UNIQUE, StrEnum, auto, verify
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field, StrictBool, model_validator

from po_effect.config import load_config
from po_effect.identity import FrozenContract, Identified
from po_effect.tasks.items import AttemptKind

if TYPE_CHECKING:
    from pathlib import Path

    from po_effect.tasks import ItemIndex, Task

PROMPT_SPEC_SCHEMA = "po_effect.prompt_spec"
PROMPT_SPEC_SCHEMA_VERSION = 5


@verify(UNIQUE)
class CandidateKind(StrEnum):
    CHOICE_TEXT = auto()
    CHOICE_LABEL = auto()
    BLANK_FILL = auto()


class ItemFormat(FrozenContract):
    question_prefix: str
    answer_prefix: str
    answer_break: str
    continuation_prefix: str
    candidates: CandidateKind
    unconditional_context: str | None = None
    choices_prefix: str | None = None
    label_prefix: str | None = None
    label_suffix: str | None = None

    @property
    def lists_choices(self) -> bool:
        return self.label_suffix is not None

    @model_validator(mode="after")
    def validate_listing(self) -> ItemFormat:
        listing = (self.choices_prefix, self.label_prefix, self.label_suffix)
        if len({value is None for value in listing}) > 1:
            raise ValueError(
                "choices_prefix, label_prefix, and label_suffix must be set "
                "together"
            )
        if self.candidates is CandidateKind.CHOICE_LABEL and not (
            self.lists_choices
        ):
            raise ValueError("label candidates require a choice listing")
        if self.candidates is CandidateKind.BLANK_FILL and (
            self.lists_choices or self.unconditional_context is not None
        ):
            raise ValueError(
                "blank-fill candidates take no choice listing and no "
                "unconditional context"
            )
        return self


class ChatSettings(FrozenContract):
    enable_thinking: StrictBool
    shots_as_turns: StrictBool
    preamble_as_system: StrictBool

    @model_validator(mode="after")
    def validate_layout(self) -> ChatSettings:
        for name in ("shots_as_turns", "preamble_as_system"):
            if getattr(self, name):
                raise ValueError(f"{name} must be false; it is not built")
        return self

    def template_kwargs(self) -> dict[str, bool]:
        return {"enable_thinking": self.enable_thinking}


class PromptLayout(FrozenContract):
    preamble: str | None = None
    shots_header: str | None = None
    shots_header_break: str | None = None
    shot_separator: str = "\n\n"
    item_format: ItemFormat
    chat: ChatSettings | None = None

    @model_validator(mode="after")
    def validate_header(self) -> PromptLayout:
        if (self.shots_header is None) != (self.shots_header_break is None):
            raise ValueError(
                "shots_header and shots_header_break must be set together"
            )
        return self


class PromptAttempts(FrozenContract):
    attempt: AttemptKind
    admits: tuple[AttemptKind, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_admits(self) -> PromptAttempts:
        if len(set(self.admits)) != len(self.admits):
            raise ValueError("admits lists an attempt kind twice")
        if self.attempt not in self.admits:
            raise ValueError(
                "admits must include the declared attempt "
                f"{self.attempt.value!r}"
            )
        return self


class PromptSpec(Identified, PromptLayout, PromptAttempts):
    identity_schema: ClassVar[str] = PROMPT_SPEC_SCHEMA
    identity_schema_version: ClassVar[int] = PROMPT_SPEC_SCHEMA_VERSION
    name: str
    task_id: str
    task_version: int = Field(ge=1)
    shot_item_ids: tuple[str, ...] = ()

    def require_admitted(self, kind: AttemptKind) -> None:
        if kind not in self.admits:
            raise ValueError(
                f"prompt {self.name!r} does not admit attempt kind "
                f"{kind.value!r}; admits "
                f"{[admitted.value for admitted in self.admits]!r}"
            )

    @model_validator(mode="after")
    def validate_header_has_shots(self) -> PromptSpec:
        if self.shots_header is not None and not self.shot_item_ids:
            raise ValueError("shots_header requires shots")
        return self


class UpstreamProvenance(FrozenContract):
    repo: str
    commit: str
    path: str
    symbol: str
    license: str
    covers: str


class ShotRef(FrozenContract):
    split: str
    native_id: str


class PromptConfig(PromptLayout, PromptAttempts):
    shots: tuple[ShotRef, ...] = ()


class PromptEntry(PromptConfig):
    provenance: tuple[UpstreamProvenance, ...] = Field(min_length=1)


class PromptsConfig(FrozenContract):
    prompts: dict[str, PromptEntry] = Field(min_length=1)


def resolve_prompt(
    name: str, config: PromptConfig, task: Task, items: ItemIndex
) -> PromptSpec:
    shots = tuple(
        items.lookup(ref.split, ref.native_id) for ref in config.shots
    )
    for shot in shots:
        if (shot.task_id, shot.task_version) != (task.task_id, task.version):
            raise ValueError(
                f"shot {shot.native_id!r} belongs to a different task"
            )
    return PromptSpec(
        name=name,
        task_id=task.task_id,
        task_version=task.version,
        preamble=config.preamble,
        shots_header=config.shots_header,
        shots_header_break=config.shots_header_break,
        shot_item_ids=tuple(shot.item_id for shot in shots),
        shot_separator=config.shot_separator,
        item_format=config.item_format,
        chat=config.chat,
        attempt=config.attempt,
        admits=config.admits,
    )


def load_prompt_config(config_dir: Path, name: str) -> PromptEntry:
    prompts = load_config(config_dir / "prompts.toml", PromptsConfig).prompts
    entry = prompts.get(name)
    if entry is None:
        raise ValueError(
            f"unknown prompt {name!r}; available: {sorted(prompts)!r}"
        )
    return entry
