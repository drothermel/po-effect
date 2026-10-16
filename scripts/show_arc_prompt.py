from __future__ import annotations

import argparse
from pathlib import Path

from dr_hf import read_rows
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from po_effect.config import DEFAULT_CONFIG_DIR
from po_effect.prompts import (
    MultipleChoiceRenderer,
    load_prompt_config,
    resolve_prompt,
)
from po_effect.tasks import (
    ArcChallengeTask,
    Item,
    ItemIndex,
    ItemSet,
    ItemSetDerivation,
    ItemSetRole,
    build_item_index,
    load_arc_challenge,
)

DEFAULT_PROMPT = "olmes_rc"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print the exact context, candidate continuations, and "
            "unconditional context sent to the model for one "
            "ARC-Challenge item under the selected prompt configuration."
        )
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--native-id")
    selection.add_argument("--index", type=int)
    parser.add_argument("--format", default=DEFAULT_PROMPT)
    parser.add_argument("--show-whitespace", action="store_true")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    return parser.parse_args(argv)


def evaluation_item_set(task: ArcChallengeTask, items: ItemIndex) -> ItemSet:
    origin = items.splits.get(task.config.split)
    if origin is None:
        raise ValueError(f"unknown evaluation split {task.config.split!r}")
    return ItemSet(
        name=origin.key.name,
        task_id=task.task_id,
        task_version=task.version,
        role=ItemSetRole.REPORT,
        item_ids=tuple(items.item_ids[key] for key in origin.native_ids),
        derivation=ItemSetDerivation(
            method="origin_split", parent=origin.content_hash
        ),
    )


def select_item(
    args: argparse.Namespace, item_set: ItemSet, items: ItemIndex
) -> Item:
    if args.native_id is not None:
        item_id = items.item_ids.get(args.native_id)
        if item_id is None or item_id not in item_set.item_ids:
            raise SystemExit(
                f"native id {args.native_id!r} is not in the evaluation set"
            )
        return items.get(item_id)
    index = 0 if args.index is None else args.index
    if not 0 <= index < len(item_set.item_ids):
        raise SystemExit(
            f"index {index} is outside the evaluation set of "
            f"{len(item_set.item_ids)} items"
        )
    return items.get(item_set.item_ids[index])


def exact(value: str, *, show_whitespace: bool) -> Text:
    shown = repr(value) if show_whitespace else value
    return Text(shown, overflow="fold", end="")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    task = load_arc_challenge(args.config_dir)
    prompt_config = load_prompt_config(args.config_dir, args.format)
    items = build_item_index(task, tuple(read_rows(task.source)))
    item_set = evaluation_item_set(task, items)
    prompt = resolve_prompt(prompt_config, task, items)
    item = select_item(args, item_set, items)
    rendered = MultipleChoiceRenderer().render(
        prompt,
        item.problem,
        tuple(items.get(i) for i in prompt.shot_item_ids),
    )
    show = args.show_whitespace
    console = Console(highlight=False)
    console.print(
        Panel(
            exact(rendered.context, show_whitespace=show),
            title=Text(item.native_id),
            title_align="left",
        )
    )
    table = Table(title=Text("candidates"), highlight=False)
    table.add_column("index", justify="right")
    table.add_column("continuation", overflow="fold")
    table.add_column("gold")
    for position, candidate in enumerate(rendered.candidates):
        table.add_row(
            Text(str(position)),
            exact(candidate, show_whitespace=show),
            Text("*" if position == item.reference.gold_index else ""),
        )
    console.print(table)
    unconditional = rendered.unconditional_context
    console.print(
        Panel(
            Text("none")
            if unconditional is None
            else exact(unconditional, show_whitespace=show),
            title=Text("unconditional context"),
            title_align="left",
        )
    )


if __name__ == "__main__":
    main()
