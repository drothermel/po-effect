from pathlib import Path

import pytest
from dr_providers import ContinuationTokenization as ProviderTokenization
from pydantic import ValidationError

from po_effect.config import DEFAULT_CONFIG_DIR, load_config
from po_effect.models import (
    CandidateScore,
    Device,
    ModelsConfig,
    RunConditions,
    load_model_handle,
    load_models_config,
    provider_config,
)


def write_models(tmp_path: Path, field: str, line: str | None) -> Path:
    source = (DEFAULT_CONFIG_DIR / "models.toml").read_text().splitlines()
    lines = [
        (line if text.startswith(f"{field} =") else text) for text in source
    ]
    path = tmp_path / "models.toml"
    path.write_text("\n".join(text for text in lines if text is not None))
    return path


@pytest.mark.parametrize(
    ("field", "line"),
    [
        ("continuation_tokenization", None),
        ("continuation_tokenization", 'continuation_tokenization = "whole"'),
        ("add_bos_token", None),
        ("add_bos_token", 'add_bos_token = "no"'),
    ],
)
def test_model_entry_requires_known_tokenization_fields(
    tmp_path: Path, field: str, line: str | None
) -> None:
    with pytest.raises(
        ValidationError, match=rf"models\.dclm_baseline_150m_seed0\.{field}\n"
    ):
        load_config(write_models(tmp_path, field, line), ModelsConfig)


def test_config_declares_every_model_entry() -> None:
    config = load_models_config()
    assert len(config.models) == 45


def test_default_resolves_to_dclm_baseline_1b_seed0() -> None:
    handle = load_model_handle()
    assert load_models_config().default == "dclm_baseline_1b_seed0"
    assert handle.pin.repo_id == "drotherm/DataDecide-dclm-baseline-1B"
    assert handle == load_model_handle(name="dclm_baseline_1b_seed0")


def test_unknown_default_is_rejected(tmp_path: Path) -> None:
    path = write_models(tmp_path, "default", 'default = "missing"')
    with pytest.raises(ValidationError, match=r"default\n.*'missing'"):
        load_config(path, ModelsConfig)


def test_unknown_model_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model 'missing'"):
        load_model_handle(name="missing")


def test_model_entry_maps_to_provider_config() -> None:
    handle = load_model_handle()
    local = provider_config(
        handle, RunConditions(device=Device.CPU, batch_size=3)
    ).definition.local
    assert local is not None
    assert local.continuation_tokenization == (
        ProviderTokenization.LM_EVAL_ENCODE_PAIR
    )
    assert local.add_bos_token is handle.add_bos_token
    assert local.batch_size == 3


def test_invalid_count_and_nonfinite_scores() -> None:
    with pytest.raises(ValidationError):
        CandidateScore(
            log_likelihood=float("nan"),
            token_count=0,
            char_count=1,
            byte_count=1,
            unconditional_log_likelihood=0,
        )
