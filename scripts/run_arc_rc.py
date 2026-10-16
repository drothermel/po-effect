from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from dr_hf import read_rows

from po_effect.config import DEFAULT_CONFIG_DIR, load_config
from po_effect.measurement import (
    GenerationSolver,
    GenerationSpec,
    IndexMatchJudge,
    MeasurementResult,
    MeasurementRuntime,
    MeasurementSpec,
    ReadoutSpec,
    RecordStore,
    ScoredCandidatesSolver,
    SingleCallSpec,
    Solver,
    SolverSpec,
    apply_readout,
    inapplicable_reason,
    readout_ids_for,
    run_measurement,
)
from po_effect.models import (
    Device,
    ModelHandle,
    ModelsConfig,
    ModelSession,
    RunConditions,
    load_model_handle,
    load_models_config,
)
from po_effect.prompts import (
    MultipleChoiceRenderer,
    PromptsConfig,
    load_prompt_config,
    resolve_prompt,
)
from po_effect.records import FileRecordStore, export_wide
from po_effect.tasks import (
    ArcChallengeTask,
    AttemptKind,
    ItemIndex,
    ItemSet,
    ItemSetDerivation,
    ItemSetRole,
    build_item_index,
    load_arc_challenge,
    prefix_item_set,
)

if TYPE_CHECKING:
    from collections.abc import Callable


DEFAULT_BATCH_SIZE = 8
DEFAULT_PROMPT = "olmes_rc"
DEFAULT_MAX_NEW_TOKENS = 16


def prompt_names(config_dir: Path) -> list[str]:
    return sorted(
        load_config(config_dir / "prompts.toml", PromptsConfig).prompts
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
            "Measure ARC-Challenge with the pinned model under the selected "
            "prompt, either scoring candidates or generating greedily, apply "
            "the applicable readouts, and export a wide parquet table."
        ),
    )
    parser.add_argument(
        "--device", choices=[device.value for device in Device], required=True
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--model", choices=sorted(models.models), help=model_help(models)
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        choices=names,
        help=(
            f"prompt entry from prompts.toml: {', '.join(names)} "
            f"(default {DEFAULT_PROMPT})"
        ),
    )
    parser.add_argument(
        "--attempt",
        choices=[kind.value for kind in AttemptKind],
        help=(
            "score candidates or generate (default the prompt entry's "
            "declared attempt)"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=(
            "generation token limit for --attempt generate "
            f"(default {DEFAULT_MAX_NEW_TOKENS})"
        ),
    )
    return parser.parse_args(argv)


def model_help(models: ModelsConfig) -> str:
    return (
        f"model entry from models.toml: {', '.join(sorted(models.models))} "
        f"(default {models.default})"
    )


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


def measure(  # noqa: PLR0913, PLR0917
    spec: MeasurementSpec,
    conditions: RunConditions,
    task: ArcChallengeTask,
    items: ItemIndex,
    solver: Solver,
    store: RecordStore,
    open_session: Callable[
        [ModelHandle, RunConditions], ModelSession
    ] = ModelSession,
) -> tuple[MeasurementResult, float]:
    store.put_measurement(spec, conditions)
    with open_session(spec.model, conditions) as session:
        runtime = MeasurementRuntime(task, items, session, solver, store)
        start = perf_counter()
        result = run_measurement(spec, runtime)
        return result, perf_counter() - start


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    task = load_arc_challenge(args.config_dir)
    prompt_config = load_prompt_config(args.config_dir, args.prompt)
    items = build_item_index(task, tuple(read_rows(task.source)))
    full_set = evaluation_item_set(task, items)
    prompt = resolve_prompt(args.prompt, prompt_config, task, items)
    kind = (
        prompt.attempt if args.attempt is None else AttemptKind(args.attempt)
    )
    try:
        prompt.require_admitted(kind)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    item_set = (
        full_set
        if args.limit is None
        else prefix_item_set(full_set, args.limit)
    )
    handle = load_model_handle(args.config_dir, args.model)
    conditions = RunConditions(
        device=Device(args.device), batch_size=args.batch_size
    )
    renderer = MultipleChoiceRenderer()
    solver: Solver
    solver_spec: SolverSpec
    if kind is AttemptKind.GENERATE:
        solver = GenerationSolver(renderer)
        solver_spec = GenerationSpec(
            prompt=prompt, max_new_tokens=args.max_new_tokens
        )
    else:
        solver = ScoredCandidatesSolver(renderer)
        solver_spec = SingleCallSpec(prompt=prompt)
    spec = MeasurementSpec(
        task_id=task.task_id,
        task_version=task.version,
        item_set=item_set,
        model=handle,
        solver_id=solver.solver_id,
        solver_spec=solver_spec,
        attempt_kind=solver.kind,
    )
    store = FileRecordStore(args.output_dir / "records")
    result, elapsed = measure(spec, conditions, task, items, solver, store)
    measurement_hash = spec.identity_hash
    skipped = []
    for rule in readout_ids_for(spec.attempt_kind):
        readout_spec = ReadoutSpec(
            measurement_hash=measurement_hash,
            readout_id=rule,
            judge_id=IndexMatchJudge.judge_id,
        )
        reason = inapplicable_reason(readout_spec, store)
        if reason is None:
            apply_readout(readout_spec, store, items)
        else:
            skipped.append((rule, reason))
    export = args.output_dir / f"{measurement_hash}.parquet"
    export_wide(store, measurement_hash, items, export)
    rate = result.measured / elapsed if elapsed > 0 else 0.0
    print(f"measurement {measurement_hash}")
    print(f"attempt {spec.attempt_kind.value}")
    print(f"items {len(result.attempts)} ({result.measured} measured)")
    print(f"items per second {rate:.2f}")
    print(f"export {export}")
    for rule, reason in skipped:
        print(f"readout {rule} not applicable: {reason}")


if __name__ == "__main__":
    main()
