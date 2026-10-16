from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING, Any

from dr_providers import MessageRole
from pydantic import Field

from po_effect.identity import FrozenContract
from po_effect.models.handle import ModelCapability, ModelHandle

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from po_effect.prompts import ChatSettings


class AppliedTemplate(FrozenContract):
    template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_kwargs: dict[str, bool]


class ChatTemplate:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, kwargs: dict[str, bool]
    ) -> None:
        template = tokenizer.chat_template
        if not isinstance(template, str) or not template:
            raise ValueError("tokenizer has no single chat template")
        self.tokenizer = tokenizer
        self.applied = AppliedTemplate(
            template_sha256=sha256(template.encode("utf-8")).hexdigest(),
            template_kwargs=kwargs,
        )

    def apply(self, context: str) -> str:
        kwargs: dict[str, Any] = dict(self.applied.template_kwargs)
        text = self.tokenizer.apply_chat_template(
            [{"role": MessageRole.USER.value, "content": context}],
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        if not isinstance(text, str):
            raise TypeError("chat template did not return text")
        return text


def has_chat_template(handle: ModelHandle) -> bool:
    return ModelCapability.CHAT_TEMPLATE in handle.capabilities


def load_tokenizer(handle: ModelHandle) -> PreTrainedTokenizerBase:
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(
        handle.pin.repo_id, revision=handle.pin.revision
    )
    if tokenizer is None:
        raise ValueError(f"no tokenizer for {handle.pin.repo_id!r}")
    return tokenizer


def chat_template_for(
    handle: ModelHandle,
    chat: ChatSettings | None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> ChatTemplate | None:
    if not has_chat_template(handle):
        if chat is not None and chat.enable_thinking:
            raise ValueError(
                "enable_thinking requires the chat_template capability, "
                "which the model handle lacks"
            )
        return None
    return ChatTemplate(
        load_tokenizer(handle) if tokenizer is None else tokenizer,
        {} if chat is None else chat.template_kwargs(),
    )
