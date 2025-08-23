"""Clean provider access module.

This module provides clean access to the provider registry without circular dependencies.
"""

from .registry import provider_registry

__all__ = ["provider_registry"]