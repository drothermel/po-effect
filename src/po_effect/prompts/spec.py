from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from po_effect.config import DEFAULT_CONFIG_DIR, load_config
from po_effect.identity import FrozenContract, Identified

if TYPE_CHECKING:
    from pathlib import Path

    from po_effect.tasks import ArcChallengeTask, ItemIndex

PROMPT_SPEC_SCHEMA = "po_effect.prompt_spec"
PROMPT_SPEC_SCHEMA_VERSION = 2


class ItemFormat(FrozenContract):
    question_prefix: str
    answer_prefix: str
    answer_break: str
    continuation_prefix: str
    unconditional_context: str


class PromptSpec(Identified):
    identity_schema: ClassVar[str] = PROMPT_SPEC_SCHEMA
    identity_schema_version: ClassVar[int] = PROMPT_SPEC_SCHEMA_VERSION
    task_id: str
    task_version: int = Field(ge=1)
    preamble: str | None = None
    shot_item_ids: tuple[str, ...] = ()
    shot_separator: str = "\n\n"
    item_format: ItemFormat


class UpstreamProvenance(FrozenContract):
    repo: str
    commit: str
    path: str
    symbol: str
    license: str


class ShotRef(FrozenContract):
    split: str
    native_id: str


class PromptConfig(FrozenContract):
    preamble: str | None = None
    shots: tuple[ShotRef, ...] = ()
    shot_separator: str = "\n\n"
    item_format: ItemFormat


class OlmesRcConfig(PromptConfig):
    provenance: UpstreamProvenance


class PromptsConfig(FrozenContract):
    olmes_rc: OlmesRcConfig


def resolve_prompt(
    config: PromptConfig, task: ArcChallengeTask, items: ItemIndex
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
        task_id=task.task_id,
        task_version=task.version,
        preamble=config.preamble,
        shot_item_ids=tuple(shot.item_id for shot in shots),
        shot_separator=config.shot_separator,
        item_format=config.item_format,
    )


def load_prompt_config(
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> OlmesRcConfig:
    return load_config(config_dir / "prompts.toml", PromptsConfig).olmes_rc
