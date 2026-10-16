from pathlib import Path

import pytest
from pydantic import ValidationError

from po_effect.config import DEFAULT_CONFIG_DIR, load_config
from po_effect.identity import FrozenContract
from po_effect.models import ModelsConfig
from po_effect.prompts import PromptsConfig
from po_effect.tasks import TasksConfig


class ExampleConfig(FrozenContract):
    value: str


def test_configuration_boundary(tmp_path: Path) -> None:
    path = tmp_path / "example.toml"
    path.write_text('value = "hello"\n')
    config = load_config(path, ExampleConfig)
    assert config.value == "hello"
    with pytest.raises(ValidationError, match="frozen"):
        config.__setattr__("value", "changed")
    path.write_text('value = "hello"\nunknown = 1\n')
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_config(path, ExampleConfig)


@pytest.mark.parametrize(
    ("filename", "contract"),
    [
        ("tasks.toml", TasksConfig),
        ("prompts.toml", PromptsConfig),
        ("models.toml", ModelsConfig),
    ],
)
def test_configs_parse_as_frozen_contracts(
    filename: str, contract: type[FrozenContract]
) -> None:
    config = load_config(DEFAULT_CONFIG_DIR / filename, contract)
    assert config.model_config["frozen"]
    assert config.model_config["extra"] == "forbid"
