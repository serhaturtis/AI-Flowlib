"""Core foundational models and utilities.

Provides strict Pydantic models following CLAUDE.md principles.
"""

# Strict base models - enforce CLAUDE.md principles
from .models import (
    MutableStrictBaseModel,
    StrictBaseModel,
)

__all__ = [
    # Strict Models - CLAUDE.md compliance
    "StrictBaseModel",
    "MutableStrictBaseModel",
]
