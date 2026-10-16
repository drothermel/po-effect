from __future__ import annotations

from enum import UNIQUE, StrEnum, auto, verify
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

JOB_SUFFIX = ".json"


@verify(UNIQUE)
class QueueState(StrEnum):
    PENDING = auto()
    RUNNING = auto()
    DONE = auto()
    FAILED = auto()


STATES = tuple(QueueState)


def job_file(queue: Path, state: QueueState, name: str) -> Path:
    return queue / state.value / f"{name}{JOB_SUFFIX}"
