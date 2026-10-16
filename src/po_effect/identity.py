from typing import ClassVar

from dr_serialize import (
    IdentityDocument,
    build_identity_document,
    identity_document_hash,
)
from pydantic import BaseModel, ConfigDict


class FrozenContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Identified(FrozenContract):
    identity_schema: ClassVar[str]
    identity_schema_version: ClassVar[int]

    def identity_document(self) -> IdentityDocument:
        return build_identity_document(
            schema=self.identity_schema,
            schema_version=self.identity_schema_version,
            payload=self.model_dump(mode="json"),
        )

    @property
    def identity_hash(self) -> str:
        return identity_document_hash(self.identity_document())
