from __future__ import annotations

from enum import UNIQUE, StrEnum, auto, verify
from typing import TYPE_CHECKING

from dr_hf import ModelPin
from pydantic import Field, StrictBool, field_serializer

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
    datadec_150m: ModelHandle


def load_model_handle(config_dir: Path = DEFAULT_CONFIG_DIR) -> ModelHandle:
    return load_config(config_dir / "models.toml", ModelsConfig).datadec_150m
