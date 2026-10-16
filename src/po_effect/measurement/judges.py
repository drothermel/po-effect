from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from po_effect.identity import FrozenContract

if TYPE_CHECKING:
    from po_effect.measurement.readouts import IndexCandidate
    from po_effect.tasks import IndexReference


class Scores(FrozenContract):
    accuracy: float = Field(ge=0, le=1)


class IndexMatchJudge:
    judge_id = "index_match"

    def judge(
        self,
        candidate: IndexCandidate,
        reference: IndexReference,
    ) -> Scores:
        return Scores(accuracy=float(candidate.index == reference.gold_index))


def judge_for(judge_id: str) -> IndexMatchJudge:
    if judge_id != IndexMatchJudge.judge_id:
        raise ValueError(f"unknown judge {judge_id!r}")
    return IndexMatchJudge()
