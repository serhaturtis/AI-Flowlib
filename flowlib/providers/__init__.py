"""Provider package for interacting with various services.

This package contains providers for various services, organized by type:
- LLM providers (llm): Large Language Model providers
- Database providers (db): SQL and NoSQL database providers
- Message queue providers (mq): Message queue and event bus providers
- Cache providers (cache): In-memory and distributed caching providers
- Vector database providers (vector): Vector embedding storage and search
- Storage providers (storage): Object storage and file management
- API providers (api): External API integration providers
"""

from .core.base import Provider
from .core.clean_provider_access import provider_registry
from .core.constants import PROVIDER_CATEGORIES, DEFAULT_CONFIGS, PROVIDER_TYPE_MAP
from .core import decorators
from .core import factory

# Optionally expose main provider interfaces if needed
try:
    from .llm import LLMProvider
except ImportError:
    pass

try:
    from .db import DBProvider, DBProviderSettings
except ImportError:
    pass

try:
    from .cache import CacheProvider, CacheProviderSettings
except ImportError:
    pass

try:
    from .vector import VectorDBProvider, VectorDBProviderSettings
except ImportError:
    pass

__all__ = [
    "Provider",
    "provider_registry",
    "PROVIDER_CATEGORIES",
    "DEFAULT_CONFIGS", 
    "PROVIDER_TYPE_MAP",
    "decorators",
    "factory",
    "LLMProvider",
    "DBProvider", 
    "DBProviderSettings",
    "CacheProvider",
    "CacheProviderSettings", 
    "VectorDBProvider",
    "VectorDBProviderSettings",
]
