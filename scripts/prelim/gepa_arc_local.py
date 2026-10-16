from __future__ import annotations

import argparse
from time import perf_counter
from typing import TYPE_CHECKING

from dr_hf import read_rows
from prelim_arc import (
    Harness,
    ItemResult,
    OpenRouter,
    ReflectionModel,
    RunRecorder,
    Split,
    Splits,
    Status,
    add_arm_arguments,
    build_splits,
    instruction_id,
    preamble_for,
    run_arm,
)

from po_effect.measurement import (
    AttemptRecord,
    AttemptTiming,
    DecisionRule,
    IndexMatchJudge,
    LikelihoodArgmaxReadout,
    MeasurementSpec,
    ReadoutSpec,
    RecordStore,
    ScoredCandidatesAttempt,
    ScoredCandidatesSolver,
    SingleCallSpec,
    apply_readout,
    decision_value,
)
from po_effect.models import (
    Device,
    ModelHandle,
    ModelSession,
    NamedResponseTemplate,
    RunConditions,
    load_model_handle,
    load_response_template,
)
from po_effect.prompts import (
    MultipleChoiceRenderer,
    PromptSpec,
    load_prompt_config,
    resolve_prompt,
)
from po_effect.records import open_record_store
from po_effect.tasks import (
    ArcChallengeTask,
    AttemptKind,
    ItemIndex,
    ItemSet,
    ItemSetDerivation,
    ItemSetRole,
    build_item_index,
    load_arc_challenge,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_PROMPT = "olmes_mc"
DEFAULT_BATCH_SIZE = 8
ITEM_SET_METHOD = "prelim_sorted_prefix"
SPLIT_SOURCES = {
    Split.TRAIN: "train",
    Split.VAL: "validation",
    Split.HELDOUT: "test",
}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize the preamble of a po-effect prompt entry for "
            "ARC-Challenge with a local po-effect model scored by "
            "likelihood over the answer candidates, under one arm (gepa, "
            "naive, format_fixed, or null), with an OpenRouter reflection "
            "model; attempts go to a dr-store record store under "
            "--out/records and per-item responses, candidate instructions, "
            "results.json, and progress.log go under --out."
        )
    )
    add_arm_arguments(parser)
    parser.add_argument("--model", required=True, help="models.toml entry")
    parser.add_argument(
        "--device", choices=[device.value for device in Device], required=True
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--rule",
        choices=[rule.value for rule in DecisionRule],
        default=DecisionRule.RAW.value,
        help="likelihood decision rule (default raw)",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"prompt entry from prompts.toml (default {DEFAULT_PROMPT})",
    )
    return parser.parse_args(argv)


def split_item_sets(
    task: ArcChallengeTask, items: ItemIndex, splits: Splits
) -> dict[Split, ItemSet]:
    return {
        split: ItemSet(
            name=f"prelim_{split.value}_{len(splits.ids(split))}",
            task_id=task.task_id,
            task_version=task.version,
            role=ItemSetRole.REPORT,
            item_ids=splits.ids(split),
            derivation=ItemSetDerivation(
                method=ITEM_SET_METHOD,
                parent=items.splits[source].content_hash,
            ),
        )
        for split, source in SPLIT_SOURCES.items()
    }


class LikelihoodEvaluator:
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        handle: ModelHandle,
        prompt: PromptSpec,
        items: ItemIndex,
        item_sets: dict[Split, ItemSet],
        session: ModelSession,
        store: RecordStore,
        rule: DecisionRule,
    ) -> None:
        self.handle = handle
        self.prompt = prompt
        self.items = items
        self.item_sets = item_sets
        self.session = session
        self.store = store
        self.rule = rule
        self.renderer = MultipleChoiceRenderer()
        self.solver = ScoredCandidatesSolver(self.renderer)
        self.readout = LikelihoodArgmaxReadout(rule)
        self.judge = IndexMatchJudge()
        self.shots = tuple(items.get(i) for i in prompt.shot_item_ids)
        self.split_of = {
            item_id: split
            for split, item_set in item_sets.items()
            for item_id in item_set.item_ids
        }
        self.measurements: dict[tuple[str, Split], MeasurementSpec] = {}
        self.measured = 0
        self.reused = 0

    def measurement(self, instruction: str, split: Split) -> MeasurementSpec:
        key = (instruction_id(instruction), split)
        spec = self.measurements.get(key)
        if spec is None:
            prompt = self.prompt.model_copy(
                update={"preamble": preamble_for(instruction)}
            )
            spec = MeasurementSpec(
                task_id=prompt.task_id,
                task_version=prompt.task_version,
                item_set=self.item_sets[split],
                model=self.handle,
                solver_id=self.solver.solver_id,
                solver_spec=SingleCallSpec(prompt=prompt),
                attempt_kind=self.solver.kind,
            )
            self.solver.validate(spec.solver_spec, spec.item_set)
            self.store.put_measurement(spec, self.session.conditions)
            self.measurements[key] = spec
        return spec

    def evaluate(
        self, item_ids: Sequence[str], instruction: str
    ) -> list[ItemResult]:
        groups: dict[Split, list[str]] = {}
        for item_id in item_ids:
            groups.setdefault(self.split_of[item_id], []).append(item_id)
        results: dict[str, ItemResult] = {}
        for split, ids in groups.items():
            spec = self.measurement(instruction, split)
            attempts = self.attempts(spec, ids)
            for item_id in ids:
                results[item_id] = self.result(
                    spec, item_id, attempts[item_id]
                )
        return [results[i] for i in item_ids]

    def attempts(
        self, spec: MeasurementSpec, ids: Sequence[str]
    ) -> dict[str, ScoredCandidatesAttempt]:
        measurement_hash = spec.identity_hash
        found: dict[str, ScoredCandidatesAttempt] = {}
        reused = [
            i for i in ids if self.store.has_attempt(measurement_hash, i)
        ]
        if reused:
            stored = self.store.attempts(measurement_hash)
            for item_id in reused:
                found[item_id] = scored(stored[item_id])
            self.reused += len(reused)
        for item_id in ids:
            if item_id in found:
                continue
            start = perf_counter()
            try:
                attempt = self.solver.solve(
                    self.items.get(item_id),
                    self.session,
                    spec.solver_spec,
                    self.items,
                )
            finally:
                for document in self.session.drain_evidence():
                    self.store.put_evidence(document)
            self.store.put_attempt(
                AttemptRecord(
                    measurement_hash=measurement_hash,
                    item_id=item_id,
                    attempt=attempt,
                ),
                AttemptTiming(
                    measurement_hash=measurement_hash,
                    item_id=item_id,
                    wall_time_seconds=perf_counter() - start,
                ),
            )
            found[item_id] = attempt
            self.measured += 1
        return found

    def result(
        self,
        spec: MeasurementSpec,
        item_id: str,
        attempt: ScoredCandidatesAttempt,
    ) -> ItemResult:
        item = self.items.get(item_id)
        context = self.renderer.render(
            spec.solver_spec.prompt, item.problem, self.shots
        ).context
        candidate = self.readout.read(attempt, item.problem)
        score = self.judge.judge(candidate, item.reference).accuracy
        chosen = attempt.candidates[candidate.index].strip()
        gold = attempt.candidates[item.reference.gold_index].strip()
        return ItemResult(
            item_id=item_id,
            native_id=item.native_id,
            prompt=context,
            response={
                "rule": self.rule.value,
                "decision_values": [
                    decision_value(self.rule, value)
                    for value in attempt.scores
                ],
            },
            output=chosen,
            gold=gold,
            candidate=chosen,
            score=score,
            feedback=(
                f"correct; the highest-{self.rule.value}-likelihood answer "
                f"was {chosen}"
                if score
                else (
                    f"incorrect; the highest-{self.rule.value}-likelihood "
                    f"answer was {chosen}, the right answer is {gold}"
                )
            ),
        )


def scored(record: AttemptRecord) -> ScoredCandidatesAttempt:
    attempt = record.attempt
    if not isinstance(attempt, ScoredCandidatesAttempt):
        raise TypeError("stored attempt is not a scored-candidates attempt")
    return attempt


def heldout_readouts(
    results: dict[str, object],
    evaluator: LikelihoodEvaluator,
    store: RecordStore,
    response_template: NamedResponseTemplate | None,
) -> dict[str, str]:
    applied = {}
    for key in ("seed_instruction_id", "best_instruction_id"):
        identifier = results.get(key)
        if not isinstance(identifier, str):
            continue
        spec = evaluator.measurements.get((identifier, Split.HELDOUT))
        if spec is None:
            continue
        readout = ReadoutSpec(
            measurement_hash=spec.identity_hash,
            readout_id=evaluator.rule,
            judge_id=IndexMatchJudge.judge_id,
            response_template=response_template,
        )
        apply_readout(readout, store, evaluator.items)
        applied[key] = readout.identity_hash
    return applied


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    task = load_arc_challenge(args.config_dir)
    items = build_item_index(task, tuple(read_rows(task.source)))
    prompt = resolve_prompt(
        args.prompt,
        load_prompt_config(args.config_dir, args.prompt),
        task,
        items,
    )
    try:
        prompt.require_admitted(AttemptKind.SCORE)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    rule = DecisionRule(args.rule)
    if (
        rule is DecisionRule.PMI
        and prompt.item_format.unconditional_context is None
    ):
        raise SystemExit(
            f"rule pmi needs an unconditional context, which prompt "
            f"{args.prompt!r} lacks"
        )
    handle = load_model_handle(args.config_dir, args.model)
    splits = build_splits(items, prompt, args)
    recorder = RunRecorder(args.out)
    recorder.log(
        f"start: arm {args.arm}, model {args.model}, rule {rule.value}, "
        f"reflection {args.reflection_model}, prompt {args.prompt}, splits "
        f"{splits.sizes()}"
    )
    conditions = RunConditions(
        device=Device(args.device), batch_size=args.batch_size
    )
    router = OpenRouter(recorder.log)
    try:
        with (
            open_record_store(args.out / "records") as store,
            ModelSession(handle, conditions, prompt.chat) as session,
        ):
            evaluator = LikelihoodEvaluator(
                handle,
                prompt,
                items,
                split_item_sets(task, items, splits),
                session,
                store,
                rule,
            )
            harness = Harness(evaluator, recorder, splits)
            results = run_arm(
                args,
                harness,
                prompt.preamble or "",
                ReflectionModel(
                    router, args.reflection_model, args.reflection_reasoning
                ),
                {
                    "model": args.model,
                    "device": args.device,
                    "rule": rule.value,
                    "reflection_model": args.reflection_model,
                    "reflection_reasoning": args.reflection_reasoning,
                    "prompt": args.prompt,
                },
                lambda: {
                    "api": router.summary(),
                    "attempts_measured": evaluator.measured,
                    "attempts_reused": evaluator.reused,
                },
            )
            if results.get("status") != Status.OUT_OF_BAND.value:
                results["heldout_readouts"] = heldout_readouts(
                    results,
                    evaluator,
                    store,
                    load_response_template(
                        args.config_dir, handle.response_template
                    ),
                )
                results["store"] = str(store.path)
                recorder.write_results(results)
    finally:
        router.close()


if __name__ == "__main__":
    main()
