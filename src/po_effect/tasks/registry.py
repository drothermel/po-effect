from __future__ import annotations

from enum import UNIQUE, StrEnum, auto, verify
from typing import TYPE_CHECKING

from dr_hf import read_rows

from po_effect.config import DEFAULT_CONFIG_DIR, load_config
from po_effect.identity import FrozenContract
from po_effect.tasks.arc import ArcConfig, ArcTask
from po_effect.tasks.items import TaskConfig
from po_effect.tasks.olmes import (
    BoolQTask,
    CsqaTask,
    HellaSwagTask,
    OpenBookQATask,
    PiqaTask,
    WinoGrandeTask,
)

if TYPE_CHECKING:
    from pathlib import Path

    from dr_hf import SourceRow


@verify(UNIQUE)
class TaskName(StrEnum):
    ARC_CHALLENGE = auto()
    ARC_EASY = auto()
    HELLASWAG = auto()
    PIQA = auto()
    WINOGRANDE = auto()
    BOOLQ = auto()
    OPENBOOKQA = auto()
    CSQA = auto()


class TasksConfig(FrozenContract):
    arc_challenge: ArcConfig
    arc_easy: ArcConfig
    hellaswag: TaskConfig
    piqa: TaskConfig
    winogrande: TaskConfig
    boolq: TaskConfig
    openbookqa: TaskConfig
    csqa: TaskConfig


Task = (
    ArcTask
    | HellaSwagTask
    | PiqaTask
    | WinoGrandeTask
    | BoolQTask
    | OpenBookQATask
    | CsqaTask
)


def load_task(name: TaskName, config_dir: Path = DEFAULT_CONFIG_DIR) -> Task:
    config = load_config(config_dir / "tasks.toml", TasksConfig)
    tasks: dict[TaskName, Task] = {
        TaskName.ARC_CHALLENGE: ArcTask(config.arc_challenge),
        TaskName.ARC_EASY: ArcTask(config.arc_easy),
        TaskName.HELLASWAG: HellaSwagTask(config.hellaswag),
        TaskName.PIQA: PiqaTask(config.piqa),
        TaskName.WINOGRANDE: WinoGrandeTask(config.winogrande),
        TaskName.BOOLQ: BoolQTask(config.boolq),
        TaskName.OPENBOOKQA: OpenBookQATask(config.openbookqa),
        TaskName.CSQA: CsqaTask(config.csqa),
    }
    return tasks[name]


def load_arc_challenge(config_dir: Path = DEFAULT_CONFIG_DIR) -> ArcTask:
    config = load_config(config_dir / "tasks.toml", TasksConfig)
    return ArcTask(config.arc_challenge)


def task_rows(task: Task) -> tuple[SourceRow, ...]:
    return tuple(read_rows(task.source, splits=task.splits))
