"""Embedding provider package.

This package contains providers for text embedding generation,
offering a common interface for working with different embedding models.
"""

import logging

from .base import EmbeddingProvider
from .llama_cpp.provider import LlamaCppEmbeddingProvider

__all__ = ["EmbeddingProvider", "LlamaCppEmbeddingProvider"]

logger = logging.getLogger(__name__)


# Do not register a default embedding provider with hardcoded paths
# Users must explicitly configure embedding models in agent_config.yaml
# or set up appropriate environment variables

logger.info("Embedding providers initialized - explicit configuration required")
