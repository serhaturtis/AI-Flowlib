from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar('T')

class ProviderBase(BaseModel, Generic[T]):
    """
    Strict pydantic v2 contract for all providers.
    All providers must inherit from this class and declare their schema.
    """
    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True, strict=True)

    name: str = Field(..., min_length=1, description="Provider name must not be empty")
    provider_type: str = Field(..., min_length=1, description="Provider type must not be empty")
    settings: T

    # Optionally, add validation or helper methods here if needed
