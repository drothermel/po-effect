from __future__ import annotations

import argparse
from pathlib import Path

from dr_hf import read_rows
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from po_effect.config import DEFAULT_CONFIG_DIR, load_config
from po_effect.measurement import (
    Abstention,
    DecisionRule,
    GenerationAttempt,
    GenerationSolver,
    GenerationSpec,
    IndexMatchJudge,
    LikelihoodArgmaxReadout,
    ScoredCandidatesAttempt,
    ScoredCandidatesSolver,
    SingleCallSpec,
    readout_for,
    readout_ids_for,
    truncated_text,
)
from po_effect.models import (
    Device,
    ModelsConfig,
    ModelSession,
    RunConditions,
    load_model_handle,
    load_models_config,
)
from po_effect.prompts import (
    MultipleChoiceRenderer,
    PromptsConfig,
    PromptSpec,
    load_prompt_config,
    resolve_prompt,
)
from po_effect.tasks import (
    ArcChallengeTask,
    AttemptKind,
    Item,
    ItemIndex,
    ItemSet,
    ItemSetDerivation,
    ItemSetRole,
    build_item_index,
    load_arc_challenge,
)

DEFAULT_PROMPT = "olmes_rc"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_NEW_TOKENS = 16


def prompt_names(config_dir: Path) -> list[str]:
    return sorted(
        load_config(config_dir / "prompts.toml", PromptsConfig).prompts
    )


def model_help(models: ModelsConfig) -> str:
    return (
        f"model entry from models.toml for --score or --generate: "
        f"{', '.join(sorted(models.models))} (default {models.default})"
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    config = argparse.ArgumentParser(add_help=False)
    config.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    config_dir = config.parse_known_args(argv)[0].config_dir
    names = prompt_names(config_dir)
    models = load_models_config(config_dir)
    parser = argparse.ArgumentParser(
        parents=[config],
        description=(
            "Print the exact context, candidate continuations, and "
            "unconditional context sent to the model for one "
            "ARC-Challenge item under the selected prompt configuration."
        ),
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--native-id")
    selection.add_argument("--index", type=int)
    parser.add_argument(
        "--format",
        default=DEFAULT_PROMPT,
        choices=names,
        help=f"prompt entry from prompts.toml: {', '.join(names)}",
    )
    parser.add_argument("--show-whitespace", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--score",
        action="store_true",
        help=(
            "score the item through ScoredCandidatesSolver and print the "
            "candidate scores and per-decision-rule readouts; requires "
            "--device"
        ),
    )
    action.add_argument(
        "--generate",
        action="store_true",
        help=(
            "generate greedily through GenerationSolver and print the raw "
            "and truncated text, the applicable readout, and the judge's "
            "scores; requires --device"
        ),
    )
    parser.add_argument(
        "--device",
        choices=[device.value for device in Device],
        help="device for --score or --generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "batch size for --score or --generate "
            f"(default {DEFAULT_BATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=(
            f"token limit for --generate (default {DEFAULT_MAX_NEW_TOKENS})"
        ),
    )
    parser.add_argument(
        "--model",
        choices=sorted(models.models),
        help=model_help(models),
    )
    args = parser.parse_args(argv)
    if (args.score or args.generate) and args.device is None:
        parser.error("--score and --generate require --device")
    return args


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


def optional_float(value: float | None) -> Text:
    return Text("" if value is None else f"{value:.4f}")


def score_table(
    attempt: ScoredCandidatesAttempt, item: Item, *, show_whitespace: bool
) -> Table:
    table = Table(title=Text("candidate scores"), highlight=False)
    table.add_column("index", justify="right")
    table.add_column("continuation", overflow="fold")
    table.add_column("gold")
    table.add_column("loglik", justify="right", no_wrap=True)
    table.add_column("tokens", justify="right")
    table.add_column("chars", justify="right")
    table.add_column("bytes", justify="right")
    table.add_column("uncond loglik", justify="right", no_wrap=True)
    for position, (candidate, score) in enumerate(
        zip(attempt.candidates, attempt.scores, strict=True)
    ):
        table.add_row(
            Text(str(position)),
            exact(candidate, show_whitespace=show_whitespace),
            Text("*" if position == item.reference.gold_index else ""),
            optional_float(score.log_likelihood),
            Text(str(score.token_count)),
            Text(str(score.char_count)),
            Text(str(score.byte_count)),
            optional_float(score.unconditional_log_likelihood),
        )
    return table


def readout_table(
    attempt: ScoredCandidatesAttempt,
    item: Item,
    prompt: PromptSpec,
    *,
    show_whitespace: bool,
) -> Table:
    table = Table(title=Text("decision rules"), highlight=False)
    table.add_column("rule")
    table.add_column("chosen", justify="right")
    table.add_column("continuation", overflow="fold")
    table.add_column("matches gold")
    judge = IndexMatchJudge()
    for rule in DecisionRule:
        readout = LikelihoodArgmaxReadout(rule)
        reason = readout.inapplicable_reason(
            attempt.kind, prompt.item_format.candidates, (attempt,)
        )
        if reason is not None:
            table.add_row(
                Text(rule.value),
                Text("not applicable"),
                Text(reason),
                Text(""),
            )
            continue
        try:
            candidate = readout.read(attempt, item.problem)
        except (ValueError, ZeroDivisionError) as error:
            table.add_row(
                Text(rule.value),
                Text(""),
                Text(f"readout cannot apply: {error}"),
                Text(""),
            )
            continue
        verdict = judge.judge(candidate, item.reference)
        table.add_row(
            Text(rule.value),
            Text(str(candidate.index)),
            exact(
                attempt.candidates[candidate.index],
                show_whitespace=show_whitespace,
            ),
            Text("yes" if verdict.accuracy == 1 else "no"),
        )
    return table


def score_item(
    console: Console,
    args: argparse.Namespace,
    prompt: PromptSpec,
    item: Item,
    items: ItemIndex,
) -> None:
    solver = ScoredCandidatesSolver(MultipleChoiceRenderer())
    spec = SingleCallSpec(prompt=prompt)
    with open_session(args) as session:
        attempt = solver.solve(item, session, spec, items)
    show = args.show_whitespace
    console.print(score_table(attempt, item, show_whitespace=show))
    console.print(readout_table(attempt, item, prompt, show_whitespace=show))


def open_session(args: argparse.Namespace) -> ModelSession:
    return ModelSession(
        load_model_handle(args.config_dir, args.model),
        RunConditions(device=Device(args.device), batch_size=args.batch_size),
    )


def generation_panels(
    attempt: GenerationAttempt, *, show_whitespace: bool
) -> tuple[Panel, Panel]:
    output = attempt.output
    raw = Panel(
        exact(output.text, show_whitespace=show_whitespace),
        title=Text("generated text"),
        title_align="left",
        subtitle=Text(
            f"stop reason {output.stop_reason}, "
            f"prompt tokens {output.prompt_tokens}, "
            f"completion tokens {output.completion_tokens}"
        ),
        subtitle_align="left",
    )
    truncated = Panel(
        exact(truncated_text(output), show_whitespace=show_whitespace),
        title=Text("truncated at the first newline"),
        title_align="left",
    )
    return raw, truncated


def generation_table(
    attempt: GenerationAttempt,
    item: Item,
    prompt: PromptSpec,
    *,
    show_whitespace: bool,
) -> Table:
    table = Table(title=Text("generation readout"), highlight=False)
    table.add_column("readout")
    table.add_column("candidate", justify="right")
    table.add_column("choice or abstention reason", overflow="fold")
    table.add_column("accuracy", justify="right")
    table.add_column("abstained", justify="right")
    judge = IndexMatchJudge()
    candidates = prompt.item_format.candidates
    for readout_id in readout_ids_for(AttemptKind.GENERATION):
        readout = readout_for(readout_id)
        reason = readout.inapplicable_reason(
            attempt.kind, candidates, (attempt,)
        )
        if reason is not None:
            continue
        candidate = readout.read(attempt, item.problem)
        scores = judge.judge(candidate, item.reference)
        if isinstance(candidate, Abstention):
            chosen = Text("abstained")
            detail = Text(candidate.reason)
        else:
            chosen = Text(str(candidate.index))
            detail = exact(
                item.problem.choices[candidate.index],
                show_whitespace=show_whitespace,
            )
        table.add_row(
            Text(readout_id.value),
            chosen,
            detail,
            Text(f"{scores.accuracy:g}"),
            Text(str(scores.abstained)),
        )
    return table


def generate_item(
    console: Console,
    args: argparse.Namespace,
    prompt: PromptSpec,
    item: Item,
    items: ItemIndex,
) -> None:
    solver = GenerationSolver(MultipleChoiceRenderer())
    spec = GenerationSpec(prompt=prompt, max_new_tokens=args.max_new_tokens)
    with open_session(args) as session:
        attempt = solver.solve(item, session, spec, items)
    show = args.show_whitespace
    for panel in generation_panels(attempt, show_whitespace=show):
        console.print(panel)
    console.print(
        generation_table(attempt, item, prompt, show_whitespace=show)
    )


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
    if args.score:
        score_item(console, args, prompt, item, items)
    if args.generate:
        generate_item(console, args, prompt, item, items)


if __name__ == "__main__":
    main()
