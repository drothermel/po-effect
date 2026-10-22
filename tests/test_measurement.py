from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from conftest import (
    CPU_CONDITIONS,
    example_format,
    fake_backend,
    make_item,
    readout_spec,
    scored_attempt,
)
from dr_providers import ProviderScoreRequest
from dr_providers.surfaces.testing.fake_backend import ScoreCall

from po_effect.measurement import (
    AttemptRecord,
    AttemptTiming,
    DecisionRule,
    IndexCandidate,
    LikelihoodArgmaxReadout,
    MeasurementRuntime,
    MeasurementSpec,
    ReadoutSpec,
    ScoredCandidatesAttempt,
    ScoredCandidatesSolver,
    SingleCallSpec,
    apply_readout,
    inapplicable_reason,
    run_measurement,
)
from po_effect.models import (
    CandidateScore,
    Device,
    ModelSession,
    RunConditions,
    load_model_handle,
)
from po_effect.prompts import (
    MultipleChoiceRenderer,
    PromptConfig,
    PromptSpec,
    resolve_prompt,
)
from po_effect.records import open_record_store
from po_effect.records.store import (
    attempt_key,
    evidence_key,
    readout_spec_key,
    timing_key,
)
from po_effect.tasks import (
    AttemptKind,
    Item,
    ItemSet,
    ItemSetDerivation,
    ItemSetRole,
    MultipleChoiceProblem,
)

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import ExampleMeasurement
    from dr_providers import (
        LocalExecutionEvidence,
        ProviderCallRequest,
        ProviderInvocationEvidence,
    )

    from po_effect.measurement import MeasurementResult
    from po_effect.records import DrStoreRecordStore
    from po_effect.tasks import ArcTask, ItemIndex


def test_solver_rejects_shots_in_evaluated_set() -> None:
    shot = make_item(native_id="s")
    target = make_item(native_id="t")
    prompt = PromptSpec(
        name="example",
        attempt=AttemptKind.SCORE,
        admits=(AttemptKind.SCORE,),
        task_id="arc_challenge",
        task_version=1,
        shot_item_ids=(shot.item_id,),
        item_format=example_format(),
    )
    solver = ScoredCandidatesSolver(MultipleChoiceRenderer())

    def item_set(*items: Item) -> ItemSet:
        return ItemSet(
            name="example",
            task_id="arc_challenge",
            task_version=1,
            role=ItemSetRole.REPORT,
            item_ids=tuple(item.item_id for item in items),
            derivation=ItemSetDerivation(method="explicit"),
        )

    solver.validate(SingleCallSpec(prompt=prompt), item_set(target))
    with pytest.raises(ValueError, match=shot.item_id):
        solver.validate(SingleCallSpec(prompt=prompt), item_set(target, shot))


def test_measurement_rejects_prompt_for_other_task_version(
    example: ExampleMeasurement,
) -> None:
    prompt = example.spec.solver_spec.prompt.model_copy(
        update={"task_version": 2}
    )
    with pytest.raises(ValueError, match="different task"):
        MeasurementSpec.model_validate(
            {
                **dict(example.spec),
                "solver_spec": SingleCallSpec(prompt=prompt),
            }
        )


@pytest.mark.parametrize("admits", [(AttemptKind.SCORE,), tuple(AttemptKind)])
def test_measurement_accepts_admitted_attempt_kind(
    example: ExampleMeasurement, admits: tuple[AttemptKind, ...]
) -> None:
    prompt = example.spec.solver_spec.prompt.model_copy(
        update={"admits": admits}
    )
    spec = MeasurementSpec.model_validate(
        {**dict(example.spec), "solver_spec": SingleCallSpec(prompt=prompt)}
    )
    assert spec.attempt_kind is AttemptKind.SCORE


def test_measurement_rejects_attempt_kind_the_prompt_does_not_admit(
    example: ExampleMeasurement,
) -> None:
    prompt = example.spec.solver_spec.prompt.model_copy(
        update={
            "attempt": AttemptKind.GENERATE,
            "admits": (AttemptKind.GENERATE,),
        }
    )
    with pytest.raises(
        ValueError,
        match="prompt 'example' does not admit attempt kind 'score'",
    ):
        MeasurementSpec.model_validate(
            {
                **dict(example.spec),
                "solver_spec": SingleCallSpec(prompt=prompt),
            }
        )


def score(
    log_likelihood: float,
    unconditional: float,
    *,
    tokens: int = 1,
    chars: int = 2,
    size: int = 2,
) -> CandidateScore:
    return CandidateScore(
        log_likelihood=log_likelihood,
        token_count=tokens,
        char_count=chars,
        byte_count=size,
        unconditional_log_likelihood=unconditional,
    )


@pytest.mark.parametrize(
    ("rule", "index"),
    [
        (DecisionRule.RAW, 1),
        (DecisionRule.PER_TOKEN, 1),
        (DecisionRule.PER_CHAR, 0),
        (DecisionRule.PER_BYTE, 1),
        (DecisionRule.PMI, 1),
    ],
)
def test_five_decision_rules(rule: DecisionRule, index: int) -> None:
    problem = MultipleChoiceProblem(question="q", choices=("abcd", "éé"))
    attempt = ScoredCandidatesAttempt(
        candidates=problem.choices,
        scores=(
            score(-3, -4, tokens=1, chars=4, size=4),
            score(-2, -8, tokens=3, chars=2, size=4),
        ),
        evidence=(),
        chat_template=None,
    )
    assert LikelihoodArgmaxReadout(rule).read(attempt, problem).index == index


def test_ties_select_first() -> None:
    attempt = ScoredCandidatesAttempt(
        candidates=(" a", " b"),
        scores=(score(-2, -1), score(-2, -1)),
        evidence=(),
        chat_template=None,
    )
    problem = MultipleChoiceProblem(question="q", choices=("a", "b"))
    assert (
        LikelihoodArgmaxReadout(DecisionRule.RAW).read(attempt, problem).index
        == 0
    )


def test_attempts_convert_scores_and_log_evidence(
    example: ExampleMeasurement,
) -> None:
    result = run_measurement(example.spec, example.runtime)
    assert result.measured == 2
    first = scored_attempt(result.attempts[0])
    assert [s.log_likelihood for s in first.scores] == [-2, -3]
    assert [s.unconditional_log_likelihood for s in first.scores] == [-3, -8]
    assert first.scores[1].char_count == 3
    assert first.scores[1].byte_count == 5
    session = example.runtime.session
    assert session.execution == example.backend.execution()
    assert session.score_requests == 4
    items = example.runtime.items
    prompt = example.spec.solver_spec.prompt
    rendered = example.runtime.solver.renderer.render(
        prompt,
        items.get(example.spec.item_set.item_ids[0]).problem,
        tuple(items.get(i) for i in prompt.shot_item_ids),
    )
    assert rendered.unconditional_context is not None
    for ref, context in zip(
        first.evidence,
        (rendered.context, rendered.unconditional_context),
        strict=True,
    ):
        expected = ProviderScoreRequest(
            config=session.config,
            context=context,
            continuations=rendered.candidates,
        )
        assert ref.request_hash == expected.identity_hash
    objects = example.store.objects
    for record in result.attempts:
        for ref in record.attempt.evidence:
            reference = objects.resolve(evidence_key(ref.evidence_hash))
            assert reference is not None
            assert reference.content_hash == ref.evidence_hash
            stored = objects.get(reference)
            assert isinstance(stored, dict)
            assert isinstance(stored["payload"], dict)
            assert stored["payload"]["request_hash"] == ref.request_hash


def test_resume_skips_stored_attempts(example: ExampleMeasurement) -> None:
    first_id, second_id = example.spec.item_set.item_ids
    stored = AttemptRecord(
        measurement_hash=example.spec.identity_hash,
        item_id=first_id,
        attempt=ScoredCandidatesAttempt(
            candidates=(" a", " éé"),
            scores=(score(-1, -1), score(-5, -1)),
            evidence=(),
            chat_template=None,
        ),
    )
    example.runtime.store.put_attempt(
        stored,
        AttemptTiming(
            measurement_hash=example.spec.identity_hash,
            item_id=first_id,
            wall_time_seconds=0.5,
        ),
    )
    result = run_measurement(example.spec, example.runtime)
    assert result.measured == 1
    assert result.attempts[0] == stored
    assert result.attempts[1].item_id == second_id
    assert len(example.backend.calls) == 2
    second = example.runtime.items.get(second_id)
    assert all(
        isinstance(call, ScoreCall) and call.continuations == (" c", " d")
        for call in example.backend.calls
    )
    assert second.problem.choices == ("c", "d")
    run_measurement(example.spec, example.runtime)
    assert len(example.backend.calls) == 2


def test_readouts_derive_from_attempts_without_model_calls(
    example: ExampleMeasurement,
) -> None:
    run_measurement(example.spec, example.runtime)
    calls = len(example.backend.calls)
    store = example.runtime.store
    expected = {
        DecisionRule.RAW: ((0, 0.0), (0, 1.0)),
        DecisionRule.PER_TOKEN: ((0, 0.0), (0, 1.0)),
        DecisionRule.PER_CHAR: ((0, 0.0), (0, 1.0)),
        DecisionRule.PER_BYTE: ((1, 1.0), (0, 1.0)),
        DecisionRule.PMI: ((1, 1.0), (0, 1.0)),
    }
    for rule, values in expected.items():
        records = apply_readout(
            readout_spec(example, rule), store, example.runtime.items
        )
        assert [(r.candidate, r.scores.accuracy) for r in records] == [
            (IndexCandidate(index=index), accuracy)
            for index, accuracy in values
        ]
        assert store.readout_records(readout_spec(example, rule)) == {
            r.item_id: r for r in records
        }
    assert len(example.backend.calls) == calls
    assert {
        store.readout(
            example.spec.identity_hash,
            readout_spec(example, rule).identity_hash,
        ).readout_id
        for rule in DecisionRule
    } == set(DecisionRule)
    unknown = readout_spec(example, DecisionRule.RAW).model_copy(
        update={"judge_id": "other"}
    )
    with pytest.raises(ValueError, match="unknown judge"):
        apply_readout(unknown, store, example.runtime.items)


@pytest.mark.parametrize("after_calls", [1, 2])
def test_execution_drift_aborts_run(
    example: ExampleMeasurement,
    after_calls: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = example.backend
    original = backend.execution()

    def execution() -> LocalExecutionEvidence:
        if len(backend.calls) > after_calls:
            return original.model_copy(update={"torch_version": "changed"})
        return original

    monkeypatch.setattr(backend, "execution", execution)
    with pytest.raises(RuntimeError, match="execution conditions changed"):
        run_measurement(example.spec, example.runtime)


@pytest.mark.parametrize("failure", ["request", "outcome"])
def test_scoring_failure_leaves_no_attempt(
    example: ExampleMeasurement,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    from dr_providers import ProviderTransportFailure, RecoverabilityClass

    provider = example.runtime.session.provider
    invoke_original = provider.invoke
    returned: list[str] = []

    def altered(request: ProviderCallRequest) -> ProviderInvocationEvidence:
        evidence = invoke_original(request)
        if len(example.backend.calls) < 3:
            return evidence
        if failure == "request":
            return evidence.model_copy(update={"request_hash": "f" * 64})
        return evidence.model_copy(
            update={
                "score_response": None,
                "failure": ProviderTransportFailure(
                    code="local_execution_error",
                    message="fixture failure",
                    recoverability=RecoverabilityClass.PERMANENT,
                ),
            }
        )

    def invoke(request: ProviderCallRequest) -> ProviderInvocationEvidence:
        evidence = altered(request)
        returned.append(evidence.identity_hash)
        return evidence

    monkeypatch.setattr(provider, "invoke", invoke)
    with pytest.raises(RuntimeError, match=r"another request|score failed"):
        run_measurement(example.spec, example.runtime)
    store = example.store
    first, second = example.spec.item_set.item_ids
    assert store.has_attempt(example.spec.identity_hash, first)
    assert not store.has_attempt(example.spec.identity_hash, second)
    measurement_hash = example.spec.identity_hash
    assert store.objects.resolve(attempt_key(measurement_hash, second)) is None
    assert store.objects.resolve(timing_key(measurement_hash, second)) is None
    assert len(set(returned)) == 3
    assert all(
        store.objects.resolve(evidence_key(evidence_hash)) is not None
        for evidence_hash in returned
    )


def run_with(
    example: ExampleMeasurement, conditions: RunConditions, root: Path
) -> tuple[int, int]:
    handle = example.spec.model
    backend = fake_backend(handle, conditions)
    runtime = example.runtime
    with (
        open_record_store(root) as store,
        ModelSession(handle, conditions, None, backend) as session,
    ):
        result = run_measurement(
            example.spec,
            MeasurementRuntime(
                runtime.task,
                runtime.items,
                session,
                runtime.solver,
                store,
            ),
        )
    return result.measured, len(backend.calls)


def recorded_conditions(
    root: Path, measurement_hash: str
) -> RunConditions | None:
    with open_record_store(root) as store:
        return store.run_conditions(measurement_hash)


def test_identity_ignores_run_conditions(
    example: ExampleMeasurement, tmp_path: Path
) -> None:
    measurement_hash = example.spec.identity_hash
    other = RunConditions(device=Device.CUDA, batch_size=2)
    for name, conditions in (("a", CPU_CONDITIONS), ("b", other)):
        assert run_with(example, conditions, tmp_path / name) == (2, 4)
        assert recorded_conditions(tmp_path / name, measurement_hash) == (
            conditions
        )
    assert not {"device", "batch_size"} & set(example.spec.model.model_dump())


@pytest.mark.parametrize(
    ("conditions", "field"),
    [
        (RunConditions(device=Device.MPS, batch_size=8), "device"),
        (RunConditions(device=Device.CPU, batch_size=4), "batch_size"),
    ],
)
def test_resume_refuses_different_run_conditions(
    example: ExampleMeasurement,
    tmp_path: Path,
    conditions: RunConditions,
    field: str,
) -> None:
    run_with(example, CPU_CONDITIONS, tmp_path)
    with pytest.raises(ValueError, match=f"run condition {field!r}"):
        run_with(example, conditions, tmp_path)
    assert recorded_conditions(tmp_path, example.spec.identity_hash) == (
        CPU_CONDITIONS
    )


def test_resume_proceeds_with_identical_run_conditions(
    example: ExampleMeasurement, tmp_path: Path
) -> None:
    run_with(example, CPU_CONDITIONS, tmp_path)
    assert run_with(example, CPU_CONDITIONS, tmp_path) == (0, 0)


def test_format_without_unconditional_context_scores_conditionally(
    task: ArcTask,
    items: ItemIndex,
    item_set: ItemSet,
    prompt_config: PromptConfig,
    tmp_path: Path,
) -> None:
    config = prompt_config.model_copy(
        update={
            "item_format": prompt_config.item_format.model_copy(
                update={"unconditional_context": None}
            )
        }
    )
    prompt = resolve_prompt("example", config, task, items)
    handle = load_model_handle()
    backend = fake_backend(handle, CPU_CONDITIONS)
    solver = ScoredCandidatesSolver(MultipleChoiceRenderer())
    spec = MeasurementSpec(
        task_id=task.task_id,
        task_version=task.version,
        item_set=item_set,
        model=handle,
        solver_id=solver.solver_id,
        solver_spec=SingleCallSpec(prompt=prompt),
        attempt_kind=solver.kind,
    )
    with (
        open_record_store(tmp_path / "records") as store,
        ModelSession(handle, CPU_CONDITIONS, None, backend) as session,
    ):
        result = run_measurement(
            spec, MeasurementRuntime(task, items, session, solver, store)
        )
        assert session.score_requests == len(item_set.item_ids)
        check_conditional_readouts(spec, store, items, result)
    assert len(backend.calls) == len(item_set.item_ids)


def check_conditional_readouts(
    spec: MeasurementSpec,
    store: DrStoreRecordStore,
    items: ItemIndex,
    result: MeasurementResult,
) -> None:
    item_set = spec.item_set
    for record in result.attempts:
        assert len(record.attempt.evidence) == 1
        assert all(
            score.unconditional_log_likelihood is None
            for score in scored_attempt(record).scores
        )
    assert [
        s.log_likelihood for s in scored_attempt(result.attempts[1]).scores
    ] == [
        -3,
        -8,
    ]

    def rule_spec(rule: DecisionRule) -> ReadoutSpec:
        return ReadoutSpec(
            measurement_hash=spec.identity_hash,
            readout_id=rule,
            judge_id="index_match",
            response_template=None,
        )

    pmi = rule_spec(DecisionRule.PMI)
    assert inapplicable_reason(pmi, store) is not None
    with pytest.raises(ValueError, match="'pmi' does not apply"):
        apply_readout(pmi, store, items)
    assert (
        store.objects.resolve(
            readout_spec_key(spec.identity_hash, pmi.identity_hash)
        )
        is None
    )
    applicable = [
        rule for rule in DecisionRule if rule is not DecisionRule.PMI
    ]
    for rule in applicable:
        assert inapplicable_reason(rule_spec(rule), store) is None
        records = apply_readout(rule_spec(rule), store, items)
        assert len(records) == len(item_set.item_ids)
    assert {
        store.readout(
            spec.identity_hash, rule_spec(rule).identity_hash
        ).readout_id
        for rule in applicable
    } == set(applicable)
    assert len(store.attempts(spec.identity_hash)) == len(item_set.item_ids)


def test_readouts_apply_to_attempts_with_unconditional_scores(
    example: ExampleMeasurement,
) -> None:
    run_measurement(example.spec, example.runtime)
    for rule in DecisionRule:
        assert (
            inapplicable_reason(
                readout_spec(example, rule), example.runtime.store
            )
            is None
        )
