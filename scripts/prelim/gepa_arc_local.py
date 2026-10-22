from __future__ import annotations

import argparse
from time import perf_counter
from typing import TYPE_CHECKING

from dr_hf import read_rows
from prelim_arc import (
    OBJECTIVE_READOUTS,
    Harness,
    ItemResult,
    Objective,
    OpenRouter,
    ReflectionModel,
    RunRecorder,
    Split,
    Splits,
    Status,
    Verdict,
    add_arm_arguments,
    build_splits,
    instruction_id,
    preamble_for,
    run_arm,
)

from po_effect.measurement import (
    Abstention,
    Attempt,
    AttemptRecord,
    AttemptTiming,
    Candidate,
    DecisionRule,
    GenerationAttempt,
    GenerationRule,
    GenerationSolver,
    GenerationSpec,
    IndexMatchJudge,
    LikelihoodArgmaxReadout,
    MeasurementSpec,
    ReadoutId,
    ReadoutSpec,
    RecordStore,
    ScoredCandidatesAttempt,
    ScoredCandidatesSolver,
    SingleCallSpec,
    apply_readout,
    decision_value,
    readout_for,
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
    CandidateKind,
    ChatSettings,
    MultipleChoiceRenderer,
    PromptSpec,
    choice_labels,
    load_prompt_config,
    resolve_prompt,
)
from po_effect.records import open_record_store
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

if TYPE_CHECKING:
    from collections.abc import Sequence

    from po_effect.measurement import SolverSpec

DEFAULT_PROMPTS = {AttemptKind.SCORE: "olmes_mc", AttemptKind.GENERATE: "cfpo"}
DEFAULT_MAX_NEW_TOKENS = 64
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
            "ARC-Challenge with a local po-effect model, either scored by "
            "likelihood over the answer candidates or generating greedily "
            "and read out by the --objective readout, under one arm "
            "(gepa, naive, format_fixed, null, or eval), with an OpenRouter "
            "reflection model; "
            "attempts go to a dr-store record store under --out/records and "
            "per-item responses, candidate instructions, results.json, and "
            "progress.log go under --out."
        )
    )
    add_arm_arguments(parser)
    parser.add_argument("--model", required=True, help="models.toml entry")
    parser.add_argument(
        "--device", choices=[device.value for device in Device], required=True
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--attempt",
        choices=[kind.value for kind in AttemptKind],
        default=AttemptKind.SCORE.value,
        help="score candidates by likelihood or generate (default score)",
    )
    parser.add_argument(
        "--rule",
        choices=[rule.value for rule in DecisionRule],
        default=DecisionRule.RAW.value,
        help="likelihood decision rule for --attempt score (default raw)",
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
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="turn thinking on in the chat template for --attempt generate",
    )
    parser.add_argument(
        "--prompt",
        help=(
            "prompt entry from prompts.toml (default "
            + ", ".join(
                f"{name} for {kind.value}"
                for kind, name in DEFAULT_PROMPTS.items()
            )
            + ")"
        ),
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


def judged(candidate: Candidate, item: Item) -> float:
    return IndexMatchJudge().judge(candidate, item.reference).accuracy


def label(candidate: Candidate, item: Item) -> str:
    if isinstance(candidate, Abstention):
        return f"abstained: {candidate.reason}"
    return choice_labels(item.problem)[candidate.index]


class LikelihoodMode:
    def __init__(
        self, renderer: MultipleChoiceRenderer, rule: DecisionRule
    ) -> None:
        self.rule = rule
        self.solver = ScoredCandidatesSolver(renderer)
        self.readout = LikelihoodArgmaxReadout(rule)
        self.readout_ids: tuple[ReadoutId, ...] = (rule,)
        self.describe = {
            "attempt": AttemptKind.SCORE.value,
            "rule": rule.value,
        }

    def solver_spec(self, prompt: PromptSpec) -> SolverSpec:
        return SingleCallSpec(prompt=prompt)

    def result(self, item: Item, context: str, attempt: Attempt) -> ItemResult:
        if not isinstance(attempt, ScoredCandidatesAttempt):
            raise TypeError(
                "stored attempt is not a scored-candidates attempt"
            )
        candidate = self.readout.read(attempt, item.problem)
        score = judged(candidate, item)
        chosen = attempt.candidates[candidate.index].strip()
        gold = attempt.candidates[item.reference.gold_index].strip()
        return ItemResult(
            item_id=item.item_id,
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


class GenerationMode:
    def __init__(
        self,
        renderer: MultipleChoiceRenderer,
        max_new_tokens: int,
        response_template: NamedResponseTemplate | None,
        objective: Objective,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.solver = GenerationSolver(renderer)
        template = (
            None if response_template is None else response_template.template
        )
        self.objective = OBJECTIVE_READOUTS[objective]
        self.readout_ids: tuple[ReadoutId, ...] = (
            GenerationRule.ANSWER_LETTER,
            GenerationRule.FIRST_LETTER,
            GenerationRule.LETTER_ANYWHERE,
            GenerationRule.CHOICE_TEXT,
        )
        self.readouts = {
            rule: readout_for(rule, template) for rule in self.readout_ids
        }
        self.describe = {
            "attempt": AttemptKind.GENERATE.value,
            "objective": objective.value,
            "readout": self.objective.value,
            "max_new_tokens": max_new_tokens,
        }

    def solver_spec(self, prompt: PromptSpec) -> SolverSpec:
        return GenerationSpec(
            prompt=prompt, max_new_tokens=self.max_new_tokens
        )

    def result(self, item: Item, context: str, attempt: Attempt) -> ItemResult:
        if not isinstance(attempt, GenerationAttempt):
            raise TypeError("stored attempt is not a generation attempt")
        candidates = {
            rule.value: readout.read(attempt, item.problem)
            for rule, readout in self.readouts.items()
        }
        verdicts = {
            name: Verdict(
                candidate=label(found, item), score=judged(found, item)
            )
            for name, found in candidates.items()
        }
        chosen = verdicts[self.objective.value].candidate
        score = verdicts[self.objective.value].score
        gold = choice_labels(item.problem)[item.reference.gold_index]
        text = attempt.output.text
        return ItemResult(
            item_id=item.item_id,
            native_id=item.native_id,
            prompt=context,
            response={
                "text": text,
                "stop_reason": attempt.output.stop_reason,
            },
            output=text,
            gold=gold,
            candidate=chosen,
            score=score,
            feedback=(
                f"correct; the answer read from the response was {chosen}"
                if score
                else (
                    f"incorrect; the answer read from the response was "
                    f"{chosen}, the right answer is {gold}"
                )
            ),
            readouts=verdicts,
        )


Mode = LikelihoodMode | GenerationMode


class LocalEvaluator:
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        handle: ModelHandle,
        prompt: PromptSpec,
        items: ItemIndex,
        item_sets: dict[Split, ItemSet],
        session: ModelSession,
        store: RecordStore,
        mode: Mode,
    ) -> None:
        self.handle = handle
        self.prompt = prompt
        self.items = items
        self.item_sets = item_sets
        self.session = session
        self.store = store
        self.mode = mode
        self.renderer = MultipleChoiceRenderer()
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
            solver = self.mode.solver
            spec = MeasurementSpec(
                task_id=prompt.task_id,
                task_version=prompt.task_version,
                item_set=self.item_sets[split],
                model=self.handle,
                solver_id=solver.solver_id,
                solver_spec=self.mode.solver_spec(prompt),
                attempt_kind=solver.kind,
            )
            solver.validate(spec.solver_spec, spec.item_set)
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
                item = self.items.get(item_id)
                context = self.renderer.render(
                    spec.solver_spec.prompt, item.problem, self.shots
                ).context
                results[item_id] = self.mode.result(
                    item, context, attempts[item_id]
                )
        return [results[i] for i in item_ids]

    def attempts(
        self, spec: MeasurementSpec, ids: Sequence[str]
    ) -> dict[str, Attempt]:
        measurement_hash = spec.identity_hash
        found: dict[str, Attempt] = {}
        reused = [
            i for i in ids if self.store.has_attempt(measurement_hash, i)
        ]
        if reused:
            stored = self.store.attempts(measurement_hash)
            for item_id in reused:
                found[item_id] = stored[item_id].attempt
            self.reused += len(reused)
        for item_id in ids:
            if item_id in found:
                continue
            start = perf_counter()
            try:
                attempt = self.mode.solver.solve(
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


def heldout_readouts(
    results: dict[str, object],
    evaluator: LocalEvaluator,
    store: RecordStore,
    response_template: NamedResponseTemplate | None,
) -> dict[str, dict[str, str]]:
    applied: dict[str, dict[str, str]] = {}
    for key in ("seed_instruction_id", "best_instruction_id"):
        identifier = results.get(key)
        if not isinstance(identifier, str):
            continue
        spec = evaluator.measurements.get((identifier, Split.HELDOUT))
        if spec is None:
            continue
        for readout_id in evaluator.mode.readout_ids:
            readout = ReadoutSpec(
                measurement_hash=spec.identity_hash,
                readout_id=readout_id,
                judge_id=IndexMatchJudge.judge_id,
                response_template=response_template,
            )
            apply_readout(readout, store, evaluator.items)
            applied.setdefault(key, {})[readout_id.value] = (
                readout.identity_hash
            )
    return applied


def generation_chat(
    prompt: PromptSpec, *, enable_thinking: bool
) -> ChatSettings:
    if prompt.chat is None:
        return ChatSettings(
            enable_thinking=enable_thinking,
            shots_as_turns=False,
            preamble_as_system=False,
        )
    return prompt.chat.model_copy(update={"enable_thinking": enable_thinking})


def resolve_mode(
    args: argparse.Namespace,
    kind: AttemptKind,
    prompt: PromptSpec,
    response_template: NamedResponseTemplate | None,
) -> tuple[PromptSpec, Mode]:
    renderer = MultipleChoiceRenderer()
    if kind is AttemptKind.GENERATE:
        if prompt.item_format.candidates is not CandidateKind.CHOICE_LABEL:
            raise SystemExit(
                f"generation reads answer letters, which prompt "
                f"{args.prompt!r} does not list"
            )
        prompt = prompt.model_copy(
            update={
                "chat": generation_chat(
                    prompt, enable_thinking=args.enable_thinking
                )
            }
        )
        return prompt, GenerationMode(
            renderer,
            args.max_new_tokens,
            response_template,
            Objective(args.objective),
        )
    if args.enable_thinking:
        raise SystemExit("--enable-thinking applies to --attempt generate")
    if Objective(args.objective) is not Objective.STRICT:
        raise SystemExit("--objective applies to --attempt generate")
    rule = DecisionRule(args.rule)
    if (
        rule is DecisionRule.PMI
        and prompt.item_format.unconditional_context is None
    ):
        raise SystemExit(
            f"rule pmi needs an unconditional context, which prompt "
            f"{args.prompt!r} lacks"
        )
    return prompt, LikelihoodMode(renderer, rule)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    kind = AttemptKind(args.attempt)
    if args.prompt is None:
        args.prompt = DEFAULT_PROMPTS[kind]
    task = load_arc_challenge(args.config_dir)
    items = build_item_index(task, tuple(read_rows(task.source)))
    prompt = resolve_prompt(
        args.prompt,
        load_prompt_config(args.config_dir, args.prompt),
        task,
        items,
    )
    try:
        prompt.require_admitted(kind)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    handle = load_model_handle(args.config_dir, args.model)
    response_template = load_response_template(
        args.config_dir, handle.response_template
    )
    prompt, mode = resolve_mode(args, kind, prompt, response_template)
    splits = build_splits(items, prompt, args)
    recorder = RunRecorder(args.out)
    recorder.log(
        f"start: arm {args.arm}, model {args.model}, {mode.describe}, "
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
            evaluator = LocalEvaluator(
                handle,
                prompt,
                items,
                split_item_sets(task, items, splits),
                session,
                store,
                mode,
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
                    **mode.describe,
                    "enable_thinking": (
                        None
                        if prompt.chat is None
                        else prompt.chat.enable_thinking
                    ),
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
                    results, evaluator, store, response_template
                )
                results["store"] = str(store.path)
                recorder.write_results(results)
    finally:
        router.close()


if __name__ == "__main__":
    main()
