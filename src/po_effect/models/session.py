from __future__ import annotations

from typing import TYPE_CHECKING, Self, assert_never

from dr_providers import (
    ContinuationTokenization as ProviderContinuationTokenization,
)
from dr_providers import (
    LocalDevice,
    LocalDtype,
    LocalModelProvider,
    PromptRendering,
    ProviderScoreRequest,
    huggingface_config,
)
from pydantic import Field

from po_effect.identity import FrozenContract
from po_effect.models.handle import (
    ContinuationTokenization,
    ModelCapability,
    ModelHandle,
    RunConditions,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dr_providers import (
        LocalExecutionEvidence,
        ProviderCallConfig,
        ProviderScoreResponse,
    )
    from dr_providers.local.backend import LocalBackend


class EvidenceRef(FrozenContract):
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class CandidateScore(FrozenContract):
    log_likelihood: float
    token_count: int = Field(gt=0)
    char_count: int = Field(gt=0)
    byte_count: int = Field(gt=0)
    unconditional_log_likelihood: float | None


class ScoringResult(FrozenContract):
    scores: tuple[CandidateScore, ...]
    evidence: tuple[EvidenceRef, ...]


def provider_tokenization(
    rule: ContinuationTokenization,
) -> ProviderContinuationTokenization:
    match rule:
        case ContinuationTokenization.LM_EVAL_ENCODE_PAIR:
            return ProviderContinuationTokenization.LM_EVAL_ENCODE_PAIR
        case _:
            assert_never(rule)


def provider_config(
    handle: ModelHandle, conditions: RunConditions
) -> ProviderCallConfig:
    return huggingface_config(
        model=handle.pin.repo_id,
        revision=handle.pin.revision,
        device=LocalDevice(conditions.device.value),
        dtype=LocalDtype(handle.dtype.value),
        batch_size=conditions.batch_size,
        max_sequence_length=handle.max_sequence_length,
        continuation_tokenization=provider_tokenization(
            handle.continuation_tokenization
        ),
        add_bos_token=handle.add_bos_token,
        prompt_rendering=PromptRendering.FLAT_TEXT,
    )


class ModelSession:
    def __init__(
        self,
        handle: ModelHandle,
        conditions: RunConditions,
        backend: LocalBackend | None = None,
    ) -> None:
        if ModelCapability.SCORE not in handle.capabilities:
            raise ValueError("model handle lacks the score capability")
        self.handle = handle
        self.conditions = conditions
        self.config = provider_config(handle, conditions)
        self.provider = LocalModelProvider(config=self.config, backend=backend)
        self.execution: LocalExecutionEvidence | None = None
        self.evidence: list[str] = []
        self.input_tokens = 0
        self.score_requests = 0
        self.warnings = 0
        self.scoring_seconds = 0.0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.provider.close()

    def drain_evidence(self) -> tuple[str, ...]:
        documents = tuple(self.evidence)
        self.evidence.clear()
        return documents

    def _score(
        self, context: str, continuations: tuple[str, ...]
    ) -> tuple[ProviderScoreResponse, EvidenceRef]:
        request = ProviderScoreRequest(
            config=self.config,
            context=context,
            continuations=continuations,
        )
        evidence = self.provider.invoke(request)
        self.evidence.append(evidence.model_dump_json())
        response = evidence.score_response
        execution = evidence.local_execution
        if response is None or execution is None:
            raise RuntimeError(f"scoring failed: {evidence.failure}")
        if evidence.request_hash != request.identity_hash:
            raise RuntimeError("scoring evidence belongs to another request")
        if self.execution is None:
            self.execution = execution
        elif execution != self.execution:
            raise RuntimeError("local execution conditions changed during run")
        if len(response.scores) != len(continuations):
            raise RuntimeError("provider returned the wrong score count")
        self.score_requests += 1
        self.warnings += len(response.warnings)
        self.scoring_seconds += evidence.wall_time_seconds or 0.0
        if response.usage is not None:
            self.input_tokens += response.usage.prompt_tokens or 0
        return response, EvidenceRef(
            evidence_hash=evidence.identity_hash,
            request_hash=evidence.request_hash,
        )

    def score(
        self,
        context: str,
        continuations: Sequence[str],
        *,
        unconditional_context: str | None,
    ) -> ScoringResult:
        continuations = tuple(continuations)
        conditional, conditional_ref = self._score(context, continuations)
        if unconditional_context is None:
            baselines: tuple[float | None, ...] = (None,) * len(continuations)
            evidence = (conditional_ref,)
        else:
            unconditional, unconditional_ref = self._score(
                unconditional_context, continuations
            )
            baselines = tuple(s.log_likelihood for s in unconditional.scores)
            evidence = (conditional_ref, unconditional_ref)
        scores = tuple(
            CandidateScore(
                log_likelihood=score.log_likelihood,
                token_count=score.token_count,
                char_count=len(text),
                byte_count=len(text.encode("utf-8")),
                unconditional_log_likelihood=baseline,
            )
            for text, score, baseline in zip(
                continuations,
                conditional.scores,
                baselines,
                strict=True,
            )
        )
        return ScoringResult(scores=scores, evidence=evidence)
