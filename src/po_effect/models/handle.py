from __future__ import annotations

from enum import UNIQUE, StrEnum, auto, verify
from typing import TYPE_CHECKING

from dr_hf import ModelPin
from pydantic import (
    Field,
    StrictBool,
    ValidationInfo,
    field_serializer,
    field_validator,
)

from po_effect.config import DEFAULT_CONFIG_DIR, load_config
from po_effect.identity import FrozenContract

if TYPE_CHECKING:
    from pathlib import Path


@verify(UNIQUE)
class Device(StrEnum):
    CUDA = auto()
    MPS = auto()
    CPU = auto()


@verify(UNIQUE)
class Dtype(StrEnum):
    FLOAT32 = auto()
    BFLOAT16 = auto()


@verify(UNIQUE)
class ContinuationTokenization(StrEnum):
    LM_EVAL_ENCODE_PAIR = auto()


@verify(UNIQUE)
class ModelCapability(StrEnum):
    SCORE = auto()


class ModelHandle(FrozenContract):
    pin: ModelPin
    dtype: Dtype
    max_sequence_length: int = Field(gt=0)
    add_bos_token: StrictBool
    continuation_tokenization: ContinuationTokenization
    capabilities: frozenset[ModelCapability] = Field(min_length=1)

    @field_serializer("capabilities")
    def serialize_capabilities(
        self, capabilities: frozenset[ModelCapability]
    ) -> list[str]:
        return sorted(capabilities)


class RunConditions(FrozenContract):
    device: Device
    batch_size: int = Field(gt=0)


class ModelsConfig(FrozenContract):
    models: dict[str, ModelHandle] = Field(min_length=1)
    default: str

    @field_validator("default")
    @classmethod
    def default_names_entry(cls, default: str, info: ValidationInfo) -> str:
        models = info.data.get("models")
        if models is not None and default not in models:
            raise ValueError(
                f"default {default!r} is not a models entry; "
                f"available: {sorted(models)!r}"
            )
        return default


def load_models_config(config_dir: Path = DEFAULT_CONFIG_DIR) -> ModelsConfig:
    return load_config(config_dir / "models.toml", ModelsConfig)


def load_model_handle(
    config_dir: Path = DEFAULT_CONFIG_DIR, name: str | None = None
) -> ModelHandle:
    config = load_models_config(config_dir)
    key = config.default if name is None else name
    handle = config.models.get(key)
    if handle is None:
        raise ValueError(
            f"unknown model {key!r}; available: {sorted(config.models)!r}"
        )
    return handle
