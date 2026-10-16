from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from po_effect.identity import FrozenContract
from po_effect.measurement.readouts import Abstention

if TYPE_CHECKING:
    from po_effect.measurement.readouts import Candidate
    from po_effect.tasks import IndexReference


class Scores(FrozenContract):
    accuracy: float = Field(ge=0, le=1)
    abstained: int = Field(ge=0, le=1)


class IndexMatchJudge:
    judge_id = "index_match"

    def judge(self, candidate: Candidate, reference: IndexReference) -> Scores:
        if isinstance(candidate, Abstention):
            return Scores(accuracy=0.0, abstained=1)
        return Scores(
            accuracy=float(candidate.index == reference.gold_index),
            abstained=0,
        )


def judge_for(judge_id: str) -> IndexMatchJudge:
    if judge_id != IndexMatchJudge.judge_id:
        raise ValueError(f"unknown judge {judge_id!r}")
    return IndexMatchJudge()
