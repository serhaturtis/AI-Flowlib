"""Core foundational models and utilities.

Provides strict Pydantic models following CLAUDE.md principles.
"""

# Strict base models - enforce CLAUDE.md principles
from .models import (
    StrictBaseModel,
    MutableStrictBaseModel,
    StrictModel,
    MutableStrictModel
)

__all__ = [
    # Strict Models - CLAUDE.md compliance
    'StrictBaseModel', 'MutableStrictBaseModel', 'StrictModel', 'MutableStrictModel'
]