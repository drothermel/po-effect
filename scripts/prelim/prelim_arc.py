from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import UNIQUE, StrEnum, auto, verify
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import gepa
import httpx
from gepa.utils.stop_condition import TimeoutStopCondition

from po_effect.config import DEFAULT_CONFIG_DIR
from po_effect.measurement import GenerationRule
from po_effect.tasks import TaskName

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Mapping, Sequence

    from gepa.core.adapter import ProposalFn
    from gepa.core.state import GEPAState

    from po_effect.prompts import PromptSpec
    from po_effect.tasks import ItemIndex, Task

OPENROUTER_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
REQUEST_TIMEOUT_S = 120.0
RETRIES = 6
MAX_BACKOFF_S = 30.0
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
REFLECTION_MAX_TOKENS = 4096
COMPONENT = "instruction"
INSTRUCTION_ID_LENGTH = 12
MAX_EMPTY_PROPOSALS = 3
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
DEFAULT_REFLECTION_MODEL = "openai/gpt-5-nano"
DEFAULT_REFLECTION_REASONING = "low"
DEFAULT_TRAIN_N = 60
DEFAULT_VAL_N = 40
DEFAULT_HELDOUT_N = 120
DEFAULT_MAX_METRIC_CALLS = 300
DEFAULT_MINIBATCH = 5
DEFAULT_WALL_CAP_S = 3600.0
DEFAULT_BAND_LOW = 0.15
DEFAULT_BAND_HIGH = 0.85
DEFAULT_FORMAT_SENTENCE = (
    "Respond with only the letter of the correct answer and nothing else."
)
PARAPHRASE_PROMPT = (
    "Paraphrase the instruction below. Keep its meaning and every "
    "requirement it states, but change the wording. Reply with the new "
    "instruction only, with no preamble or quotation marks.\n\n"
    "Instruction:\n{instruction}"
)
REASONING_CHOICES = ("off", "minimal", "low", "medium", "high")
REASONING_PARAMETER = "reasoning"
OFF = "off"


@verify(UNIQUE)
class Arm(StrEnum):
    GEPA = auto()
    NAIVE = auto()
    FORMAT_FIXED = auto()
    NULL = auto()
    EVAL = auto()


OPTIMIZING_ARMS = frozenset({Arm.GEPA, Arm.NULL})


@verify(UNIQUE)
class Objective(StrEnum):
    STRICT = auto()
    ANYWHERE = auto()
    CHOICE = auto()


OBJECTIVE_READOUTS = {
    Objective.STRICT: GenerationRule.ANSWER_LETTER,
    Objective.ANYWHERE: GenerationRule.LETTER_ANYWHERE,
    Objective.CHOICE: GenerationRule.CHOICE_TEXT,
}


@verify(UNIQUE)
class Split(StrEnum):
    TRAIN = auto()
    VAL = auto()
    HELDOUT = auto()


@verify(UNIQUE)
class Origin(StrEnum):
    SEED = auto()
    FORMAT_FIXED = auto()
    PARAPHRASE = auto()
    GEPA = auto()
    GIVEN = auto()


@verify(UNIQUE)
class Status(StrEnum):
    COMPLETE = auto()
    OUT_OF_BAND = auto()
    SEARCH_FAILED = auto()


@dataclass(frozen=True, slots=True)
class Verdict:
    candidate: str
    score: float


@dataclass(frozen=True, slots=True)
class ItemResult:
    item_id: str
    native_id: str
    prompt: str
    response: object
    output: str
    gold: str
    candidate: str
    score: float
    feedback: str
    details: Mapping[str, object] = field(default_factory=dict)
    readouts: Mapping[str, Verdict] = field(default_factory=dict)


class ItemEvaluator(Protocol):
    def evaluate(
        self, item_ids: Sequence[str], instruction: str
    ) -> list[ItemResult]: ...


def add_arm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument(
        "--task",
        choices=[name.value for name in TaskName],
        default=TaskName.ARC_CHALLENGE.value,
        help=(
            "task from tasks.toml (default arc_challenge); held-out items "
            "come from its evaluation split, validation items from its "
            "validation split when that is not the evaluation split and "
            "otherwise from the train items after the train prefix"
        ),
    )
    parser.add_argument(
        "--arm",
        choices=[arm.value for arm in Arm],
        required=True,
        help=(
            "gepa: reflective search; naive: seed only; format_fixed: seed "
            "plus --format-sentence; null: paraphrases of the seed with no "
            "feedback, best by validation, same budget; eval: no search, "
            "the --instruction-file text in place of the best instruction"
        ),
    )
    parser.add_argument(
        "--objective",
        choices=[objective.value for objective in Objective],
        default=Objective.STRICT.value,
        help=(
            "generate-attempt readout the arm optimizes and reports as its "
            "score: strict reads the answer letter on the first line, "
            "anywhere the last standalone letter, choice the last letter or "
            "choice-text mention (default strict); every run records all three"
        ),
    )
    parser.add_argument(
        "--instruction-file",
        type=Path,
        help="file whose exact text the eval arm scores on held-out",
    )
    parser.add_argument("--reflection-model", default=DEFAULT_REFLECTION_MODEL)
    parser.add_argument(
        "--reflection-reasoning",
        choices=REASONING_CHOICES,
        default=DEFAULT_REFLECTION_REASONING,
        help="reasoning effort for the reflection model when it supports it",
    )
    parser.add_argument("--train-n", type=int, default=DEFAULT_TRAIN_N)
    parser.add_argument("--val-n", type=int, default=DEFAULT_VAL_N)
    parser.add_argument("--heldout-n", type=int, default=DEFAULT_HELDOUT_N)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-metric-calls", type=int, default=DEFAULT_MAX_METRIC_CALLS
    )
    parser.add_argument("--minibatch", type=int, default=DEFAULT_MINIBATCH)
    parser.add_argument(
        "--wall-cap-s",
        type=float,
        default=DEFAULT_WALL_CAP_S,
        help="wall-time cap on the search phase in seconds",
    )
    parser.add_argument("--band-low", type=float, default=DEFAULT_BAND_LOW)
    parser.add_argument("--band-high", type=float, default=DEFAULT_BAND_HIGH)
    parser.add_argument(
        "--format-sentence",
        default=DEFAULT_FORMAT_SENTENCE,
        help="output-format sentence appended to the seed by format_fixed",
    )
    parser.add_argument("--out", type=Path, required=True)


@dataclass(frozen=True, slots=True)
class Splits:
    train: tuple[str, ...]
    val: tuple[str, ...]
    heldout: tuple[str, ...]
    sources: Mapping[Split, str]

    def split_of(self, item_id: str) -> Split:
        if item_id in self.train:
            return Split.TRAIN
        if item_id in self.val:
            return Split.VAL
        if item_id in self.heldout:
            return Split.HELDOUT
        raise KeyError(item_id)

    def sizes(self) -> dict[str, int]:
        return {
            Split.TRAIN.value: len(self.train),
            Split.VAL.value: len(self.val),
            Split.HELDOUT.value: len(self.heldout),
        }

    def ids(self, split: Split) -> tuple[str, ...]:
        match split:
            case Split.TRAIN:
                return self.train
            case Split.VAL:
                return self.val
            case Split.HELDOUT:
                return self.heldout


def build_splits(
    task: Task, items: ItemIndex, prompt: PromptSpec, args: argparse.Namespace
) -> Splits:
    shots = set(prompt.shot_item_ids)
    heldout_split = task.config.split
    val_split = (
        VAL_SPLIT
        if VAL_SPLIT in task.splits and heldout_split != VAL_SPLIT
        else TRAIN_SPLIT
    )

    def ordered(split: str) -> list[str]:
        return [
            items.item_ids[native]
            for native in sorted(items.splits[split].native_ids)
            if items.item_ids[native] not in shots
        ]

    def take(split: str, ids: list[str], count: int) -> tuple[str, ...]:
        if not 0 < count <= len(ids):
            raise SystemExit(f"{split} has {len(ids)} items, asked {count}")
        return tuple(ids[:count])

    train_ids = ordered(TRAIN_SPLIT)
    train = take(TRAIN_SPLIT, train_ids, args.train_n)
    val = take(
        val_split,
        train_ids[args.train_n :]
        if val_split == TRAIN_SPLIT
        else ordered(val_split),
        args.val_n,
    )
    return Splits(
        train=train,
        val=val,
        heldout=take(heldout_split, ordered(heldout_split), args.heldout_n),
        sources={
            Split.TRAIN: TRAIN_SPLIT,
            Split.VAL: val_split,
            Split.HELDOUT: heldout_split,
        },
    )


def preamble_for(instruction: str) -> str | None:
    return instruction or None


def instruction_id(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:INSTRUCTION_ID_LENGTH]


class RunRecorder:
    def __init__(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.out = out
        self.lock = threading.Lock()
        self.known: dict[str, str] = {}
        self.progress = out / "progress.log"
        self.responses_path = out / "responses.jsonl"
        self.candidates_path = out / "candidates.jsonl"
        self.results_path = out / "results.json"

    def log(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        with self.lock:
            print(line, flush=True)
            with self.progress.open("a") as stream:
                stream.write(line + "\n")

    def register(self, text: str, origin: Origin) -> str:
        identifier = instruction_id(text)
        if identifier not in self.known:
            self.known[identifier] = text
            _append_json(
                self.candidates_path,
                {
                    "instruction_id": identifier,
                    "origin": origin.value,
                    "instruction": text,
                },
            )
        return identifier

    def responses(
        self, results: Sequence[ItemResult], identifier: str, splits: Splits
    ) -> None:
        with self.responses_path.open("a") as stream:
            for result in results:
                row = {
                    "instruction_id": identifier,
                    "split": splits.split_of(result.item_id).value,
                    "item_id": result.item_id,
                    "native_id": result.native_id,
                    "prompt": result.prompt,
                    "response": result.response,
                    "gold": result.gold,
                    "candidate": result.candidate,
                    "score": result.score,
                    **result.details,
                    "readouts": {
                        name: asdict(verdict)
                        for name, verdict in result.readouts.items()
                    },
                }
                stream.write(json.dumps(row) + "\n")

    def write_results(self, payload: Mapping[str, object]) -> None:
        temporary = self.results_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(self.results_path)


def _append_json(path: Path, row: Mapping[str, object]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def accuracy(results: Sequence[ItemResult]) -> float:
    return sum(result.score for result in results) / len(results)


def readout_accuracies(results: Sequence[ItemResult]) -> dict[str, float]:
    names = results[0].readouts.keys() if results else ()
    return {
        name: sum(result.readouts[name].score for result in results)
        / len(results)
        for name in names
    }


def readout_summaries(
    seed_val: Sequence[ItemResult],
    seed_heldout: Sequence[ItemResult] | None,
    best_heldout: Sequence[ItemResult] | None,
) -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    for phase, results in (
        ("seed_val", seed_val),
        ("seed_heldout", seed_heldout),
        ("best_heldout", best_heldout),
    ):
        for name, value in readout_accuracies(results or ()).items():
            summaries.setdefault(name, {})[phase] = value
    for summary in summaries.values():
        if "seed_heldout" in summary and "best_heldout" in summary:
            summary["delta"] = (
                summary["best_heldout"] - summary["seed_heldout"]
            )
    return summaries


class Harness:
    def __init__(
        self,
        evaluator: ItemEvaluator,
        recorder: RunRecorder,
        splits: Splits,
    ) -> None:
        self.evaluator = evaluator
        self.recorder = recorder
        self.splits = splits
        self.memo: dict[tuple[str, str], ItemResult] = {}
        self.requested = 0
        self.evaluated = 0

    def evaluate(
        self, item_ids: Sequence[str], text: str, origin: Origin
    ) -> list[ItemResult]:
        identifier = self.recorder.register(text, origin)
        missing = [i for i in item_ids if (identifier, i) not in self.memo]
        if missing:
            fresh = self.evaluator.evaluate(missing, text)
            for result in fresh:
                self.memo[identifier, result.item_id] = result
            self.recorder.responses(fresh, identifier, self.splits)
            self.evaluated += len(fresh)
        self.requested += len(item_ids)
        return [self.memo[identifier, i] for i in item_ids]


@dataclass(slots=True)
class ApiStats:
    calls: int = 0
    failures: int = 0
    retries: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class Completion:
    content: str
    provider: str | None
    finish_reason: str | None
    usage: Mapping[str, Any]
    body: Mapping[str, Any]


@dataclass(slots=True)
class OpenRouter:
    log: Callable[[str], None]
    client: httpx.Client = field(init=False)
    lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    stats: dict[str, ApiStats] = field(init=False, default_factory=dict)
    parameters: dict[str, frozenset[str]] | None = field(
        init=False, default=None
    )

    def __post_init__(self) -> None:
        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise SystemExit(f"{API_KEY_ENV} is not set")
        self.client = httpx.Client(
            base_url=OPENROUTER_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=REQUEST_TIMEOUT_S,
        )

    def supported_parameters(self, model: str) -> frozenset[str]:
        if self.parameters is None:
            response = self.client.get("/models")
            response.raise_for_status()
            self.parameters = {
                entry["id"]: frozenset(entry.get("supported_parameters") or ())
                for entry in response.json()["data"]
            }
        if model not in self.parameters:
            raise SystemExit(f"OpenRouter does not list model {model!r}")
        return self.parameters[model]

    def reasoning(self, model: str, effort: str | None) -> dict[str, Any]:
        if effort is None:
            return {}
        if REASONING_PARAMETER not in self.supported_parameters(model):
            self.log(f"{model} does not support reasoning; {effort} omitted")
            return {}
        if effort == OFF:
            return {REASONING_PARAMETER: {"enabled": False}}
        return {REASONING_PARAMETER: {"effort": effort}}

    def chat(self, role: str, body: Mapping[str, Any]) -> str:
        completion = self.complete(role, body)
        return completion.content if completion else ""

    def complete(
        self, role: str, body: Mapping[str, Any]
    ) -> Completion | None:
        payload = {**body, "usage": {"include": True}}
        error = "no attempt"
        for attempt in range(RETRIES):
            try:
                response = self.client.post("/chat/completions", json=payload)
            except httpx.TransportError as failure:
                error = repr(failure)
            else:
                if response.status_code == httpx.codes.OK:
                    reply = response.json()
                    if reply.get("choices"):
                        self._count(role, reply)
                        choice = reply["choices"][0]
                        return Completion(
                            content=choice["message"].get("content") or "",
                            provider=reply.get("provider"),
                            finish_reason=choice.get("finish_reason"),
                            usage=reply.get("usage") or {},
                            body=reply,
                        )
                    error = f"no choices: {str(reply)[:200]}"
                else:
                    error = (
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                    if response.status_code not in RETRY_STATUSES:
                        break
            with self.lock:
                self._role(role).retries += 1
            time.sleep(min(2.0**attempt, MAX_BACKOFF_S))
        with self.lock:
            self._role(role).failures += 1
        self.log(f"{role} call to {body.get('model')} failed: {error}")
        return None

    def _role(self, role: str) -> ApiStats:
        return self.stats.setdefault(role, ApiStats())

    def _count(self, role: str, reply: Mapping[str, Any]) -> None:
        usage = reply.get("usage") or {}
        with self.lock:
            stats = self._role(role)
            stats.calls += 1
            stats.cost_usd += float(usage.get("cost") or 0.0)

    def summary(self) -> dict[str, object]:
        with self.lock:
            return {role: asdict(stats) for role, stats in self.stats.items()}

    def close(self) -> None:
        self.client.close()


class ReflectionModel:
    def __init__(
        self, router: OpenRouter, model: str, effort: str | None
    ) -> None:
        self.router = router
        self.model = model
        self.extra = router.reasoning(model, effort)

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else prompt
        )
        return self.router.chat(
            "reflection",
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": REFLECTION_MAX_TOKENS,
                **self.extra,
            },
        )


class InstructionAdapter:
    propose_new_texts: ProposalFn | None = None

    def __init__(self, harness: Harness) -> None:
        self.harness = harness

    def evaluate(
        self,
        batch: list[str],
        candidate: dict[str, str],
        capture_traces: bool = False,  # noqa: FBT001, FBT002
    ) -> gepa.EvaluationBatch[ItemResult, str]:
        results = self.harness.evaluate(
            batch, candidate[COMPONENT], Origin.GEPA
        )
        return gepa.EvaluationBatch(
            outputs=[result.output for result in results],
            scores=[result.score for result in results],
            trajectories=results if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: gepa.EvaluationBatch[ItemResult, str],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        del candidate
        trajectories = eval_batch.trajectories or []
        rows = [
            {
                "Inputs": result.prompt,
                "Generated Outputs": result.output,
                "Feedback": result.feedback,
            }
            for result in trajectories
        ]
        return dict.fromkeys(components_to_update, rows)


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    best: str
    metric_calls: int
    candidates: list[dict[str, object]]
    error: str | None = None


class CandidateWatcher:
    def __init__(self, recorder: RunRecorder) -> None:
        self.recorder = recorder
        self.seen = 1

    def __call__(self, gepa_state: GEPAState) -> bool:
        count = len(gepa_state.program_candidates)
        if count > self.seen:
            self.recorder.log(
                f"gepa accepted candidate {count - 1}; metric calls "
                f"{gepa_state.total_num_evals}"
            )
            self.seen = count
        return False


def gepa_search(
    args: argparse.Namespace,
    harness: Harness,
    seed_text: str,
    propose: ReflectionModel,
) -> SearchOutcome:
    result = gepa.optimize(
        seed_candidate={COMPONENT: seed_text},
        trainset=list(harness.splits.train),
        valset=list(harness.splits.val),
        adapter=InstructionAdapter(harness),
        reflection_lm=propose,
        reflection_minibatch_size=args.minibatch,
        max_metric_calls=args.max_metric_calls,
        seed=args.seed,
        run_dir=str(args.out / "gepa_run"),
        display_progress_bar=False,
        raise_on_exception=False,
        stop_callbacks=[
            TimeoutStopCondition(args.wall_cap_s),
            CandidateWatcher(harness.recorder),
        ],
    )
    best = result.best_candidate
    best_text = best[COMPONENT] if isinstance(best, dict) else best
    candidates: list[dict[str, object]] = [
        {
            "instruction_id": instruction_id(candidate[COMPONENT]),
            "val": score,
            "parents": parents,
        }
        for candidate, score, parents in zip(
            result.candidates,
            result.val_aggregate_scores,
            result.parents,
            strict=True,
        )
    ]
    return SearchOutcome(
        best=best_text,
        metric_calls=result.total_metric_calls or 0,
        candidates=candidates,
    )


def null_search(
    args: argparse.Namespace,
    harness: Harness,
    seed_text: str,
    propose: ReflectionModel,
    seed_val: float,
) -> SearchOutcome:
    start = time.monotonic()
    val = harness.splits.val
    calls = len(val)
    best, best_val = seed_text, seed_val
    candidates: list[dict[str, object]] = [
        {"instruction_id": instruction_id(seed_text), "val": seed_val}
    ]
    empty = 0
    while (
        calls + len(val) <= args.max_metric_calls
        and time.monotonic() - start < args.wall_cap_s
    ):
        text = propose(PARAPHRASE_PROMPT.format(instruction=seed_text)).strip()
        if not text:
            empty += 1
            if empty >= MAX_EMPTY_PROPOSALS:
                return SearchOutcome(
                    best=best,
                    metric_calls=calls,
                    candidates=candidates,
                    error="proposer returned empty paraphrases",
                )
            continue
        empty = 0
        score = accuracy(harness.evaluate(val, text, Origin.PARAPHRASE))
        calls += len(val)
        candidates.append(
            {"instruction_id": instruction_id(text), "val": score}
        )
        harness.recorder.log(
            f"null candidate {len(candidates) - 1} val {score:.3f}; "
            f"metric calls {calls}"
        )
        if score > best_val:
            best, best_val = text, score
    return SearchOutcome(best=best, metric_calls=calls, candidates=candidates)


def with_sentence(seed_text: str, sentence: str) -> str:
    return " ".join(part for part in (seed_text, sentence) if part)


def run_arm(  # noqa: PLR0913, PLR0917
    args: argparse.Namespace,
    harness: Harness,
    seed_text: str,
    propose: ReflectionModel,
    describe: Mapping[str, object],
    stats: Callable[[], Mapping[str, object]],
) -> dict[str, object]:
    start = time.monotonic()
    recorder = harness.recorder
    arm = Arm(args.arm)
    seed_id = recorder.register(seed_text, Origin.SEED)
    results: dict[str, object] = {
        "arm": arm.value,
        **describe,
        "seed": args.seed,
        "max_metric_calls": args.max_metric_calls,
        "minibatch": args.minibatch,
        "splits": harness.splits.sizes(),
        "seed_instruction_id": seed_id,
    }

    def finish(status: Status) -> dict[str, object]:
        results.update(
            status=status.value,
            requested_evaluations=harness.requested,
            fresh_evaluations=harness.evaluated,
            wall_s=round(time.monotonic() - start, 1),
            **stats(),
        )
        recorder.write_results(results)
        recorder.log(f"done: {status.value}")
        return results

    seed_val_results = harness.evaluate(
        harness.splits.val, seed_text, Origin.SEED
    )
    seed_val = accuracy(seed_val_results)
    results["readouts"] = readout_summaries(seed_val_results, None, None)
    in_band = args.band_low <= seed_val <= args.band_high
    results["screen"] = {
        "seed_val": seed_val,
        "band": [args.band_low, args.band_high],
        "in_band": in_band,
    }
    recorder.log(f"screen: seed val {seed_val:.3f}, in band {in_band}")
    if arm in OPTIMIZING_ARMS and not in_band:
        return finish(Status.OUT_OF_BAND)
    outcome = search(args, harness, seed_text, propose, seed_val)
    results.update(
        best_instruction_id=instruction_id(outcome.best),
        search_metric_calls=outcome.metric_calls,
        candidates=outcome.candidates,
        search_error=outcome.error,
    )
    seed_heldout_results = harness.evaluate(
        harness.splits.heldout, seed_text, Origin.SEED
    )
    best_heldout_results = harness.evaluate(
        harness.splits.heldout, outcome.best, Origin.GEPA
    )
    seed_heldout = accuracy(seed_heldout_results)
    best_heldout = accuracy(best_heldout_results)
    results.update(
        seed_heldout=seed_heldout,
        best_heldout=best_heldout,
        delta=best_heldout - seed_heldout,
        readouts=readout_summaries(
            seed_val_results, seed_heldout_results, best_heldout_results
        ),
    )
    recorder.log(
        f"heldout: seed {seed_heldout:.3f}, best {best_heldout:.3f} "
        f"({outcome.metric_calls} search metric calls)"
    )
    return finish(
        Status.COMPLETE if outcome.error is None else Status.SEARCH_FAILED
    )


def search(
    args: argparse.Namespace,
    harness: Harness,
    seed_text: str,
    propose: ReflectionModel,
    seed_val: float,
) -> SearchOutcome:
    arm = Arm(args.arm)
    match arm:
        case Arm.NAIVE:
            return SearchOutcome(
                best=seed_text,
                metric_calls=len(harness.splits.val),
                candidates=[
                    {
                        "instruction_id": instruction_id(seed_text),
                        "val": seed_val,
                    }
                ],
            )
        case Arm.FORMAT_FIXED:
            text = with_sentence(seed_text, args.format_sentence)
            score = accuracy(
                harness.evaluate(harness.splits.val, text, Origin.FORMAT_FIXED)
            )
            return SearchOutcome(
                best=text,
                metric_calls=2 * len(harness.splits.val),
                candidates=[
                    {
                        "instruction_id": instruction_id(seed_text),
                        "val": seed_val,
                    },
                    {"instruction_id": instruction_id(text), "val": score},
                ],
            )
        case Arm.EVAL:
            if args.instruction_file is None:
                msg = "--arm eval requires --instruction-file"
                raise SystemExit(msg)
            text = args.instruction_file.read_text(encoding="utf-8")
            harness.recorder.register(text, Origin.GIVEN)
            return SearchOutcome(best=text, metric_calls=0, candidates=[])
        case Arm.NULL:
            return null_search(args, harness, seed_text, propose, seed_val)
        case Arm.GEPA:
            try:
                return gepa_search(args, harness, seed_text, propose)
            except Exception as error:  # noqa: BLE001
                harness.recorder.log(f"gepa failed: {error!r}")
                return SearchOutcome(
                    best=seed_text,
                    metric_calls=harness.requested,
                    candidates=[],
                    error=repr(error)[:500],
                )
