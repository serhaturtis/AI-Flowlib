"""
Decorators for registering embedding models and providers.
"""

import inspect
import logging
from typing import Any, Dict, Type, Callable

from flowlib.providers.core.registry import provider_registry
# Removed ProviderType import - using config-driven provider access
from .base import EmbeddingProvider

from flowlib.core.errors.errors import ConfigurationError

logger = logging.getLogger(__name__)

# Mapping from a implementation name (if specified) to provider class
# This might need expansion if more embedding providers are added



# TODO: Add @embedding_provider decorator if needed for direct class registration