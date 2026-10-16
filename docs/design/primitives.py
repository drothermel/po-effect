from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Protocol

from po_effect.identity import FrozenContract, Identified
from po_effect.measurement import (
    IndexCandidate,
    MeasurementSpec,
    RecordStore,
    ScoredCandidatesAttempt,
    Scores,
    SingleCallSpec,
)
from po_effect.models import CandidateScore, EvidenceRef, ModelHandle
from po_effect.prompts import PromptSpec
from po_effect.tasks import (
    IndexReference,
    Item,
    ItemIndex,
    ItemSet,
    MultipleChoiceProblem,
)

# Sketch of the primitives not yet built, written against the types that
# exist in po_effect. Built primitives are described by the code and README.


# ------------------------------------------------------------------- tasks


class Paragraph(FrozenContract):
    title: str
    sentences: tuple[str, ...]


class OpenQAProblem(FrozenContract):
    question: str
    paragraphs: tuple[Paragraph, ...]


class TextReference(FrozenContract):
    answer: str
    aliases: tuple[str, ...] = ()
    supporting_facts: tuple[tuple[str, int], ...] = ()


class CodeProblem(FrozenContract):
    prompt: str
    entry_point: str
    language: str


class TestSuiteReference(FrozenContract):
    tests: str
    timeout_seconds: float


class HotpotQATask:
    task_id = "hotpot_qa"
    version = 1

    def build(self, row: object) -> Item: ...


class HumanEvalTask:
    task_id = "human_eval"
    version = 1

    def build(self, row: object) -> Item: ...


# --------------------------------------------------------------- item sets


class ItemSetRoleAdditions(StrEnum):
    OPTIMIZE = auto()
    SELECT = auto()
    HOLDOUT = auto()


class SplitPlan(Identified):
    task_id: str
    task_version: int
    sets: tuple[ItemSet, ...]

    def by_role(self, role: str) -> ItemSet: ...


def seeded_partition(
    items: Sequence[Item],
    *,
    fractions: Mapping[str, float],
    seed: int,
    exclude: Sequence[str] = (),
) -> SplitPlan: ...


# ------------------------------------------------------------------ models


class ModelCapabilityAdditions(StrEnum):
    GENERATE = auto()
    CHAT_TEMPLATE = auto()
    REASONING_CONTROL = auto()
    REASONING_VISIBLE = auto()


class ReasoningVisibility(StrEnum):
    NOT_REQUESTED = auto()
    INLINE = auto()
    PROVIDER_CHANNEL = auto()
    HIDDEN = auto()


class ResponseTemplate(FrozenContract):
    template: Mapping[str, object]


class GenerationControls(FrozenContract):
    max_new_tokens: int
    temperature: float | None = None
    stop: tuple[str, ...] = ()


class GenerationOutput(FrozenContract):
    text: str
    stop_reason: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_text: str | None = None


class GeneratingSession(Protocol):
    handle: ModelHandle

    def generate(
        self, prompt: RenderedMessages | str, controls: GenerationControls
    ) -> tuple[GenerationOutput, tuple[EvidenceRef, ...]]: ...


# ----------------------------------------------------------------- prompts


class LabelStyle(StrEnum):
    LETTER_DOT = auto()
    LETTER_COLON = auto()
    LETTER_PAREN = auto()
    NUMBER_DOT = auto()


class ChoiceListing(FrozenContract):
    choices_header: str | None
    label_style: LabelStyle
    label_prefix: str
    label_text_separator: str


class Message(FrozenContract):
    role: str
    content: str


class RenderedMessages(FrozenContract):
    messages: tuple[Message, ...]
    candidates: tuple[str, ...] | None


class OpenQARenderer:
    def render(
        self, spec: PromptSpec, problem: OpenQAProblem, shots: Sequence[Item]
    ) -> RenderedMessages: ...


class CodeRenderer:
    def render(
        self, spec: PromptSpec, problem: CodeProblem, shots: Sequence[Item]
    ) -> RenderedMessages: ...


# ---------------------------------------------------------------- attempts


class AttemptKindAdditions(StrEnum):
    GENERATION = auto()
    TRAJECTORY = auto()


class ResponseSplit(FrozenContract):
    reasoning: tuple[str, ...]
    answer: str
    visibility: ReasoningVisibility


class GenerationAttempt(FrozenContract):
    output: GenerationOutput
    evidence: tuple[EvidenceRef, ...]


class TrajectoryStep(FrozenContract):
    name: str
    inputs: Mapping[str, str]
    attempt: ScoredCandidatesAttempt | GenerationAttempt


class TrajectoryAttempt(FrozenContract):
    steps: tuple[TrajectoryStep, ...]
    final: ScoredCandidatesAttempt | GenerationAttempt


Attempt = ScoredCandidatesAttempt | GenerationAttempt | TrajectoryAttempt


def split_response(
    output: GenerationOutput, template: ResponseTemplate
) -> ResponseSplit: ...


# ----------------------------------------------------------------- solvers


class RetrieveThenAnswerSpec(FrozenContract):
    retrieve: PromptSpec
    answer: PromptSpec
    top_k: int


class GenerateExecuteRepairSpec(FrozenContract):
    generate: PromptSpec
    repair: PromptSpec
    max_rounds: int


SolverSpec = SingleCallSpec | RetrieveThenAnswerSpec | GenerateExecuteRepairSpec


class SpecSlot(FrozenContract):
    path: tuple[str, ...]


def prompt_slots(spec: SolverSpec) -> tuple[SpecSlot, ...]: ...


def replace_slot(
    spec: SolverSpec, slot: SpecSlot, prompt: PromptSpec
) -> SolverSpec: ...


class DirectGenerationSolver:
    solver_id = "direct_generation"

    def solve(
        self,
        item: Item,
        session: GeneratingSession,
        spec: SingleCallSpec,
        items: ItemIndex,
    ) -> GenerationAttempt: ...


class ReasonedGenerationSolver:
    solver_id = "reasoned_generation"

    def solve(
        self,
        item: Item,
        session: GeneratingSession,
        spec: SingleCallSpec,
        items: ItemIndex,
    ) -> GenerationAttempt: ...


class RetrieveThenAnswerSolver:
    solver_id = "retrieve_then_answer"

    def solve(
        self,
        item: Item,
        session: GeneratingSession,
        spec: RetrieveThenAnswerSpec,
        items: ItemIndex,
    ) -> TrajectoryAttempt: ...


class GenerateExecuteRepairSolver:
    solver_id = "generate_execute_repair"

    def __init__(self, sandbox: Sandbox) -> None: ...

    def solve(
        self,
        item: Item,
        session: GeneratingSession,
        spec: GenerateExecuteRepairSpec,
        items: ItemIndex,
    ) -> TrajectoryAttempt: ...


# ---------------------------------------------------------------- readouts


class TextCandidate(FrozenContract):
    text: str


class CodeCandidate(FrozenContract):
    source: str


class Abstention(FrozenContract):
    reason: str


Candidate = IndexCandidate | TextCandidate | CodeCandidate | Abstention


class FirstLetterReadout:
    readout_id = "first_letter"

    def __init__(self, labels: tuple[str, ...], *, digit_fallback: bool) -> None: ...

    def read(
        self, attempt: GenerationAttempt, problem: MultipleChoiceProblem
    ) -> IndexCandidate | Abstention: ...


class ChoiceTextPrefixReadout:
    readout_id = "choice_text_prefix"

    def read(
        self, attempt: GenerationAttempt, problem: MultipleChoiceProblem
    ) -> IndexCandidate | Abstention: ...


class AnswerSpanReadout:
    readout_id = "answer_span"

    def __init__(self, template: ResponseTemplate) -> None: ...

    def read(
        self, attempt: GenerationAttempt, problem: object
    ) -> TextCandidate | Abstention: ...


class CodeBlockReadout:
    readout_id = "code_block"

    def read(
        self, attempt: GenerationAttempt, problem: CodeProblem
    ) -> CodeCandidate | Abstention: ...


class FinalStepReadout:
    readout_id = "final_step"

    def __init__(self, inner: object) -> None: ...

    def read(self, attempt: TrajectoryAttempt, problem: object) -> Candidate: ...


# ------------------------------------------------------------------ judges


class ScoresAdditions(FrozenContract):
    correct_prob: float | None = None
    margin: float | None = None
    norm_correct_prob: float | None = None
    total_prob: float | None = None
    exact_match: float | None = None
    f1: float | None = None
    passed: float | None = None
    abstained: float | None = None


class LikelihoodProxyJudge:
    judge_id = "likelihood_proxy"

    def judge(
        self,
        candidate: Candidate,
        reference: IndexReference,
        attempt: ScoredCandidatesAttempt,
    ) -> Scores: ...


class ExactMatchF1Judge:
    judge_id = "exact_match_f1"

    def judge(
        self,
        candidate: TextCandidate | Abstention,
        reference: TextReference,
        attempt: Attempt,
    ) -> Scores: ...


class SandboxResult(FrozenContract):
    passed: bool
    exit_code: int
    stdout: str
    stderr: str


class Sandbox(Protocol):
    def run(
        self, source: str, tests: str, *, timeout_seconds: float
    ) -> SandboxResult: ...


class TestExecutionJudge:
    judge_id = "test_execution"

    def __init__(self, sandbox: Sandbox) -> None: ...

    def judge(
        self,
        candidate: CodeCandidate | Abstention,
        reference: TestSuiteReference,
        attempt: Attempt,
    ) -> Scores: ...


# ------------------------------------------------------------- measurement


class Summary(FrozenContract):
    n_items: int
    means: Mapping[str, float]
    intervals: Mapping[str, tuple[float, float]]


class PairedContrast(FrozenContract):
    a: str
    b: str
    deltas: Mapping[str, float]
    intervals: Mapping[str, tuple[float, float]]
    n_items: int


def summarize(
    store: RecordStore, readout_hash: str, *, seed: int, resamples: int
) -> Summary: ...


def contrast(
    store: RecordStore,
    a_readout_hash: str,
    b_readout_hash: str,
    *,
    seed: int,
    resamples: int,
) -> PairedContrast: ...


# -------------------------------------------------------------- optimizers


class Budget(FrozenContract):
    max_solver_calls: int | None
    max_proposer_calls: int | None
    max_rounds: int | None


class ObjectiveValue(FrozenContract):
    score: float
    n_items: int
    readout_hash: str


class Objective(Protocol):
    def evaluate(
        self, spec: SolverSpec, item_set: ItemSet
    ) -> ObjectiveValue: ...


class Proposal(Identified):
    round: int
    parent: str | None
    spec: SolverSpec
    slot: SpecSlot | None
    proposer_evidence: tuple[EvidenceRef, ...]
    objective: ObjectiveValue | None


class OptimizationResult(Identified):
    optimizer_id: str
    initial: SolverSpec
    proposals: tuple[Proposal, ...]
    best: SolverSpec
    budget_used: Budget


class Optimizer(Protocol):
    optimizer_id: str

    def optimize(
        self,
        objective: Objective,
        initial: SolverSpec,
        plan: SplitPlan,
        budget: Budget,
        proposer: GeneratingSession,
        seed: int,
    ) -> OptimizationResult: ...


class ApeOptimizer: ...


class OproOptimizer: ...


class MiproOptimizer: ...


class GepaOptimizer: ...


class CfpoOptimizer: ...


# ----------------------------------------------------------------- storage


class DrStoreRecordStore:
    def put_measurement(self, spec: MeasurementSpec, conditions: object) -> None: ...


# ----------------------------------------------------------------- studies


class Study(Identified):
    name: str
    measurements: tuple[MeasurementSpec, ...]
    optimizations: tuple[tuple[str, SolverSpec, Budget], ...]
    null_optimizers: tuple[str, ...]
    minimum_detectable_effect: float


class StudyRunner(Protocol):
    def run(self, study: Study) -> None: ...


@dataclass(frozen=True, slots=True)
class LocalStudyRunner:
    def run(self, study: Study) -> None: ...


@dataclass(frozen=True, slots=True)
class PoolStudyRunner:
    def run(self, study: Study) -> None: ...
