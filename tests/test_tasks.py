from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from conftest import make_item, make_row
from dr_hf import SourceRow

if TYPE_CHECKING:
    from po_effect.tasks import ArcTask, Item, ItemIndex, ItemSet


@pytest.mark.parametrize(("key", "gold"), [("C", 2), ("3", 2), ("5", 4)])
def test_numeric_answer_mapping(
    task: ArcTask,
    key: str,
    gold: int,
) -> None:
    row = SourceRow(
        pin=task.source,
        origin_split="test",
        native_id="x",
        fields={
            "id": "x",
            "question": "q",
            "choices": {
                "text": ["a", "b", "c", "d", "e"],
                "label": ["1", "2", "3", "4", "5"],
            },
            "answerKey": key,
        },
    )
    assert task.build(row).reference.gold_index == gold


@pytest.mark.parametrize("key", ["Z", "5"])
def test_bad_answer_mapping(task: ArcTask, key: str) -> None:
    row = make_row("test", "x", "q", ("a", "b"), key)
    with pytest.raises(ValueError, match=r"unknown answer|beyond the choices"):
        task.build(row)


def test_item_id_is_stable() -> None:
    assert make_item().item_id == make_item().item_id
    assert make_item().native_id == "x"


@pytest.mark.parametrize(
    "changed",
    [
        make_item(question="q2"),
        make_item(choices=("a", "c")),
        make_item(gold=1),
        make_item(version=2),
        make_item(native_id="y"),
    ],
)
def test_item_id_tracks_content_and_version(changed: Item) -> None:
    assert changed.item_id != make_item().item_id


def test_index_covers_every_split_and_item_set_narrows(
    items: ItemIndex, item_set: ItemSet
) -> None:
    assert sorted(items.splits) == ["test", "train"]
    assert len(items.all()) == 4
    assert [items.get(i).native_id for i in item_set.item_ids] == ["x", "y"]
