from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from dr_hf import read_rows
from prelim_arc import (
    Completion,
    Harness,
    ItemResult,
    OpenRouter,
    ReflectionModel,
    RunRecorder,
    add_arm_arguments,
    build_splits,
    preamble_for,
    run_arm,
)

from po_effect.measurement import (
    Abstention,
    AnswerLetterReadout,
    IndexCandidate,
    IndexMatchJudge,
    truncated_text,
)
from po_effect.prompts import (
    MultipleChoiceRenderer,
    PromptSpec,
    choice_labels,
    load_prompt_config,
    resolve_prompt,
)
from po_effect.tasks import (
    AttemptKind,
    Item,
    ItemIndex,
    build_item_index,
    load_arc_challenge,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from po_effect.measurement import Candidate

DEFAULT_PROMPT = "cfpo"
DEFAULT_MAX_TOKENS = 64
DEFAULT_CONCURRENCY = 8
THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
EMPTY_ANSWER = "truncated text is empty"
EMPTY_RESPONSE = "empty_response"
EMPTY_RETRIES = 3
EMPTY_BACKOFF_S = 5.0
EMPTY_BODIES_LOGGED = 5
EMPTY_BODIES_FILE = "empty_responses.jsonl"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize the preamble of a po-effect prompt entry for "
            "ARC-Challenge with an OpenRouter task model, under one arm "
            "(gepa, naive, format_fixed, or null), scoring generations with "
            "the answer-letter readout on the text before the first newline, "
            "and write per-item responses, candidate instructions, "
            "results.json, and progress.log under --out."
        )
    )
    add_arm_arguments(parser)
    parser.add_argument("--task-model", required=True)
    parser.add_argument(
        "--reasoning",
        choices=("off", "minimal", "low", "medium", "high"),
        help=(
            "reasoning setting for the task model, sent only when OpenRouter "
            "lists reasoning as supported (default: not sent)"
        ),
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"prompt entry from prompts.toml (default {DEFAULT_PROMPT})",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="concurrent task-model requests within one batch",
    )
    return parser.parse_args(argv)


def answer_text(response: str) -> str:
    return truncated_text(THINK_BLOCK.sub("", response).lstrip()).strip()


def candidate_label(candidate: Candidate, item: Item) -> str:
    if isinstance(candidate, IndexCandidate):
        return choice_labels(item.problem)[candidate.index]
    return f"abstained: {candidate.reason}"


class EmptyTracker:
    def __init__(self, out: Path, log: Callable[[str], None]) -> None:
        self.path = out / EMPTY_BODIES_FILE
        self.log = log
        self.lock = threading.Lock()
        self.empty_responses = 0
        self.retried_empty = 0
        self.bodies_logged = 0

    def record(self, item_id: str, completion: Completion | None) -> None:
        if completion is None:
            return
        with self.lock:
            if self.bodies_logged < EMPTY_BODIES_LOGGED:
                self.bodies_logged += 1
                with self.path.open("a") as stream:
                    stream.write(
                        json.dumps(
                            {"item_id": item_id, "body": completion.body}
                        )
                        + "\n"
                    )

    def retried(self) -> None:
        with self.lock:
            self.retried_empty += 1

    def exhausted(self, item_id: str, completion: Completion | None) -> None:
        with self.lock:
            self.empty_responses += 1
        cause = (
            "call failed"
            if completion is None
            else f"provider {completion.provider}, finish_reason "
            f"{completion.finish_reason}"
        )
        self.log(
            f"{EMPTY_RESPONSE}: {item_id} empty after {EMPTY_RETRIES} "
            f"retries ({cause})"
        )

    def summary(self) -> dict[str, int]:
        with self.lock:
            return {
                "empty_responses": self.empty_responses,
                "retried_empty": self.retried_empty,
            }


class OpenRouterEvaluator:
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        router: OpenRouter,
        args: argparse.Namespace,
        prompt: PromptSpec,
        items: ItemIndex,
        pool: ThreadPoolExecutor,
        empties: EmptyTracker,
    ) -> None:
        self.router = router
        self.empties = empties
        self.model = args.task_model
        self.max_tokens = args.max_tokens
        self.reasoning = router.reasoning(args.task_model, args.reasoning)
        self.prompt = prompt
        self.items = items
        self.shots = tuple(items.get(i) for i in prompt.shot_item_ids)
        self.renderer = MultipleChoiceRenderer()
        self.readout = AnswerLetterReadout(None)
        self.judge = IndexMatchJudge()
        self.pool = pool

    def evaluate(
        self, item_ids: Sequence[str], instruction: str
    ) -> list[ItemResult]:
        spec = self.prompt.model_copy(
            update={"preamble": preamble_for(instruction)}
        )
        return list(self.pool.map(lambda i: self.one(spec, i), item_ids))

    def one(self, spec: PromptSpec, item_id: str) -> ItemResult:
        item = self.items.get(item_id)
        context = self.renderer.render(spec, item.problem, self.shots).context
        completion, attempts = self.complete(item_id, context)
        response = completion.content if completion else ""
        empty = not response.strip()
        text = answer_text(response)
        candidate: Candidate
        if empty:
            candidate = Abstention(reason=EMPTY_RESPONSE)
        elif not text:
            candidate = Abstention(reason=EMPTY_ANSWER)
        else:
            candidate = self.readout.choose(text, item.problem)
        score = self.judge.judge(candidate, item.reference).accuracy
        gold = choice_labels(item.problem)[item.reference.gold_index]
        return ItemResult(
            item_id=item_id,
            native_id=item.native_id,
            prompt=context,
            response=response,
            output=response,
            gold=gold,
            candidate=candidate_label(candidate, item),
            score=score,
            feedback=(
                "correct"
                if score
                else f"incorrect; the right answer is {gold}"
            ),
            details={
                "provider": completion.provider if completion else None,
                "finish_reason": (
                    completion.finish_reason if completion else None
                ),
                "attempts": attempts,
                "call_failed": completion is None,
                EMPTY_RESPONSE: empty,
            },
        )

    def complete(
        self, item_id: str, context: str
    ) -> tuple[Completion | None, int]:
        body: Mapping[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": context}],
            "max_tokens": self.max_tokens,
            "temperature": 0,
            **self.reasoning,
        }
        attempt = 0
        while True:
            attempt += 1
            completion = self.router.complete("task", body)
            if completion is not None and completion.content.strip():
                return completion, attempt
            self.empties.record(item_id, completion)
            if attempt > EMPTY_RETRIES:
                self.empties.exhausted(item_id, completion)
                return completion, attempt
            self.empties.retried()
            time.sleep(EMPTY_BACKOFF_S * 2 ** (attempt - 1))


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
        prompt.require_admitted(AttemptKind.GENERATE)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    splits = build_splits(items, prompt, args)
    recorder = RunRecorder(args.out)
    recorder.log(
        f"start: arm {args.arm}, task {args.task_model}, reflection "
        f"{args.reflection_model}, prompt {args.prompt}, splits "
        f"{splits.sizes()}"
    )
    router = OpenRouter(recorder.log)
    empties = EmptyTracker(args.out, recorder.log)
    try:
        with ThreadPoolExecutor(args.concurrency) as pool:
            evaluator = OpenRouterEvaluator(
                router, args, prompt, items, pool, empties
            )
            harness = Harness(evaluator, recorder, splits)
            run_arm(
                args,
                harness,
                prompt.preamble or "",
                ReflectionModel(
                    router, args.reflection_model, args.reflection_reasoning
                ),
                {
                    "task_model": args.task_model,
                    "reasoning": args.reasoning,
                    "task_reasoning_sent": bool(evaluator.reasoning),
                    "reflection_model": args.reflection_model,
                    "reflection_reasoning": args.reflection_reasoning,
                    "prompt": args.prompt,
                    "max_tokens": args.max_tokens,
                },
                lambda: {"api": router.summary(), **empties.summary()},
            )
        counts = empties.summary()
        recorder.log(
            f"empty_responses {counts['empty_responses']}, retried_empty "
            f"{counts['retried_empty']}"
        )
    finally:
        router.close()


if __name__ == "__main__":
    main()
