"""Strict Pydantic base model enforcing CLAUDE.md architectural principles.

This module provides the single source of truth for all Pydantic models
in the AI-Flowlib codebase, enforcing strict validation patterns.
"""

from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    """Base model with strict validation enforcing CLAUDE.md principles.
    
    This is THE base class for all Pydantic models in AI-Flowlib.
    It enforces:
    - strict=True: Type coercion is disabled, inputs must match exact types
    - extra="forbid": No additional fields allowed, prevents config drift
    - validate_assignment=True: Validation on all field assignments
    - frozen=True: Immutable by default, preventing accidental mutations
    
    These settings ensure fail-fast behavior and eliminate silent errors.
    """

    model_config = ConfigDict(
        strict=True,              # No type coercion - fail fast on wrong types
        extra="forbid",           # No extra fields - fail fast on unknown config
        validate_assignment=True, # Always validate - fail fast on invalid updates
        frozen=True,              # Immutable by default - explicit mutability required
        # Additional strict settings
        validate_default=True,    # Validate even default values
        use_enum_values=False,    # Preserve enum objects for their methods
        arbitrary_types_allowed=False,  # Only standard types by default
    )


class MutableStrictBaseModel(BaseModel):
    """Mutable version of StrictBaseModel for cases requiring updates.
    
    Use this ONLY when mutability is explicitly required.
    Still enforces all validation rules except frozen=False.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
        frozen=False,             # Allow mutations when explicitly needed
        validate_default=True,
        use_enum_values=False,    # Preserve enum objects for their methods
        arbitrary_types_allowed=False,
    )


# For backwards compatibility during migration
# TODO: Remove these aliases after full migration
StrictModel = StrictBaseModel
MutableStrictModel = MutableStrictBaseModel


__all__ = [
    "StrictBaseModel",
    "MutableStrictBaseModel",
    "StrictModel",
    "MutableStrictModel",
]
