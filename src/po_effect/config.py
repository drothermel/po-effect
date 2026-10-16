import tomllib
from pathlib import Path

from po_effect.identity import FrozenContract

DEFAULT_CONFIG_DIR = Path("configs")


def load_config[Contract: FrozenContract](
    path: Path, contract: type[Contract]
) -> Contract:
    with path.open("rb") as stream:
        return contract.model_validate(tomllib.load(stream))
