"""LLM provider implementations for local models with structured generation.

This package provides specialized providers for local language models with
structured generation using LlamaGrammar for enforced output format.
"""

from .base import LLMProvider, PromptConfigOverride
from .llama_cpp.provider import LlamaCppProvider, LlamaCppSettings
from .google_ai.provider import GoogleAIProvider, GoogleAISettings

__all__ = [
    "LLMProvider",
    "PromptConfigOverride",
    "LlamaCppProvider",
    "LlamaCppSettings",
    "GoogleAIProvider",
    "GoogleAISettings"
]

"""
LLM Providers Package
"""

import logging
from pydantic import Field, BaseModel, ConfigDict
from typing import Optional

# Import the decorator
from flowlib.resources.decorators.decorators import model_config
from flowlib.core.errors.errors import ConfigurationError

logger = logging.getLogger(__name__)

# All configuration should be loaded from ~/.flowlib
# No hardcoded configurations in the code!
