from __future__ import annotations

from typing import TYPE_CHECKING, Self, assert_never

from dr_providers import (
    ContinuationTokenization as ProviderContinuationTokenization,
)
from dr_providers import (
    GenerationControls,
    LocalDevice,
    LocalDtype,
    LocalModelProvider,
    MessageRole,
    PromptMessage,
    PromptRendering,
    ProviderCallConfig,
    ProviderGenerateRequest,
    ProviderScoreRequest,
    Transcript,
    huggingface_config,
)
from pydantic import Field

from po_effect.identity import FrozenContract
from po_effect.models.chat import AppliedTemplate, chat_template_for
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
        ProviderCallRequest,
        ProviderInvocationEvidence,
    )
    from dr_providers.local.backend import LocalBackend
    from dr_serialize import IdentityDocument
    from transformers import PreTrainedTokenizerBase

    from po_effect.prompts import ChatSettings


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
    chat_template: AppliedTemplate | None


class GenerationOutput(FrozenContract):
    text: str
    stop_reason: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class GenerationResult(FrozenContract):
    output: GenerationOutput
    evidence: tuple[EvidenceRef, ...]
    chat_template: AppliedTemplate | None


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
        chat: ChatSettings | None,
        backend: LocalBackend | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.handle = handle
        self.conditions = conditions
        self.chat = chat
        self.template = chat_template_for(handle, chat, tokenizer)
        self.config = provider_config(handle, conditions)
        self.provider = LocalModelProvider(config=self.config, backend=backend)
        self.execution: LocalExecutionEvidence | None = None
        self.evidence: list[IdentityDocument] = []
        self.input_tokens = 0
        self.score_requests = 0
        self.generate_requests = 0
        self.warnings = 0
        self.model_seconds = 0.0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.provider.close()

    def drain_evidence(self) -> tuple[IdentityDocument, ...]:
        documents = tuple(self.evidence)
        self.evidence.clear()
        return documents

    @property
    def applied_template(self) -> AppliedTemplate | None:
        return None if self.template is None else self.template.applied

    def templated(self, context: str) -> str:
        return (
            context if self.template is None else self.template.apply(context)
        )

    def _require(self, capability: ModelCapability) -> None:
        if capability not in self.handle.capabilities:
            raise ValueError(
                f"model handle lacks the {capability.value} capability"
            )

    def _invoke(
        self, request: ProviderCallRequest
    ) -> tuple[ProviderInvocationEvidence, EvidenceRef]:
        evidence = self.provider.invoke(request)
        self.evidence.append(evidence.identity_document())
        execution = evidence.local_execution
        if execution is None or evidence.failure is not None:
            raise RuntimeError(
                f"{request.kind.value} failed: {evidence.failure}"
            )
        if evidence.request_hash != request.identity_hash:
            raise RuntimeError("evidence belongs to another request")
        if self.execution is None:
            self.execution = execution
        elif execution != self.execution:
            raise RuntimeError("local execution conditions changed during run")
        self.model_seconds += evidence.wall_time_seconds or 0.0
        return evidence, EvidenceRef(
            evidence_hash=evidence.identity_hash,
            request_hash=evidence.request_hash,
        )

    def _score(
        self, context: str, continuations: tuple[str, ...]
    ) -> tuple[tuple[float, ...], tuple[int, ...], EvidenceRef]:
        evidence, ref = self._invoke(
            ProviderScoreRequest(
                config=self.config,
                context=context,
                continuations=continuations,
            )
        )
        response = evidence.score_response
        if response is None:
            raise RuntimeError("scoring returned no score response")
        if len(response.scores) != len(continuations):
            raise RuntimeError("provider returned the wrong score count")
        self.score_requests += 1
        self.warnings += len(response.warnings)
        if response.usage is not None:
            self.input_tokens += response.usage.prompt_tokens or 0
        return (
            tuple(score.log_likelihood for score in response.scores),
            tuple(score.token_count for score in response.scores),
            ref,
        )

    def score(
        self,
        context: str,
        continuations: Sequence[str],
        *,
        unconditional_context: str | None,
    ) -> ScoringResult:
        self._require(ModelCapability.SCORE)
        continuations = tuple(continuations)
        conditional, token_counts, conditional_ref = self._score(
            self.templated(context), continuations
        )
        if unconditional_context is None:
            baselines: tuple[float | None, ...] = (None,) * len(continuations)
            evidence = (conditional_ref,)
        else:
            unconditional, _, unconditional_ref = self._score(
                self.templated(unconditional_context), continuations
            )
            baselines = unconditional
            evidence = (conditional_ref, unconditional_ref)
        scores = tuple(
            CandidateScore(
                log_likelihood=log_likelihood,
                token_count=token_count,
                char_count=len(text),
                byte_count=len(text.encode("utf-8")),
                unconditional_log_likelihood=baseline,
            )
            for text, log_likelihood, token_count, baseline in zip(
                continuations,
                conditional,
                token_counts,
                baselines,
                strict=True,
            )
        )
        return ScoringResult(
            scores=scores,
            evidence=evidence,
            chat_template=self.applied_template,
        )

    def generate(self, context: str, max_new_tokens: int) -> GenerationResult:
        self._require(ModelCapability.GENERATE)
        config = ProviderCallConfig(
            definition=self.config.definition,
            controls=GenerationControls(token_limit=max_new_tokens),
        )
        evidence, ref = self._invoke(
            ProviderGenerateRequest(
                config=config,
                transcript=Transcript(
                    messages=(
                        PromptMessage(
                            role=MessageRole.USER,
                            content=self.templated(context),
                        ),
                    )
                ),
            )
        )
        response = evidence.response
        if response is None or response.stop_reason is None:
            raise RuntimeError("generation returned no complete response")
        usage = response.usage
        if (
            usage is None
            or usage.prompt_tokens is None
            or usage.completion_tokens is None
        ):
            raise RuntimeError("generation returned no token usage")
        self.generate_requests += 1
        self.warnings += len(response.warnings)
        self.input_tokens += usage.prompt_tokens
        return GenerationResult(
            output=GenerationOutput(
                text=response.text,
                stop_reason=response.stop_reason.value,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            ),
            evidence=(ref,),
            chat_template=self.applied_template,
        )
