"""LLM provider implementations for local models with structured generation.

This package provides specialized providers for local language models with
structured generation using LlamaGrammar for enforced output format.
"""

import logging

from .base import LLMProvider, PromptConfigOverride
from .google_ai.provider import GoogleAIProvider, GoogleAISettings
from .llama_cpp.provider import LlamaCppProvider, LlamaCppSettings

__all__ = [
    "LLMProvider",
    "PromptConfigOverride",
    "LlamaCppProvider",
    "LlamaCppSettings",
    "GoogleAIProvider",
    "GoogleAISettings",
]

"""
LLM Providers Package
"""

logger = logging.getLogger(__name__)

# All configuration should be loaded from ~/.flowlib
# No hardcoded configurations in the code!
