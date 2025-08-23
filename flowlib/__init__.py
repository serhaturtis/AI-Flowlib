"""Flowlib Framework.

This package provides a unified, type-safe framework for building and executing flows,
using providers, and creating agents with clean error handling and validation.

Key features:
1. Declarative resource management with decorators for models, providers, and prompts
2. Lazy initialization of LLM providers with automatic model loading
3. Type-safe flow execution with validation
4. Structured error handling across all components
"""

from flowlib.core.context.context import Context

from .flows.base import FlowStatus, FlowSettings, Flow
from .flows.decorators import flow, pipeline
from .flows.registry import flow_registry

from .flows.models.results import FlowResult

from .core.errors.errors import BaseError, ValidationError, ExecutionError, ResourceError

# Use clean architecture imports (no circular dependencies)
from .core import container, get_container

# Clean provider access (no circular dependencies)
from .providers.core.clean_provider_access import provider_registry

# Clean constants (no enum dependencies)
from flowlib.resources.models.constants import ResourceType
from flowlib.providers.core.constants import PROVIDER_CATEGORIES, DEFAULT_CONFIGS

# Core interfaces and decorators (no circular dependencies)
from .core.decorators import resource, flow, llm_config, vector_db_config, config
from .providers import Provider
from flowlib.providers.core.factory import create_provider
# Note: Provider decorators will be loaded dynamically to avoid circular dependencies


__version__ = "0.1.0"

__all__ = [
    # Core models
    "FlowResult",
    "FlowStatus",
    "Context",
    "FlowSettings",
    
    # Errors
    "BaseError",
    "ValidationError",
    "ExecutionError",
    "ResourceError",
    
    # Clean architecture components
    "container",
    "get_container", 
    "provider_registry",
    
    # Clean constants (no ProviderType enum)
    "ResourceType",
    "PROVIDER_CATEGORIES",
    "DEFAULT_CONFIGS",
    
    # Clean decorators (no circular dependencies)
    "resource",
    "flow",
    "llm_config", 
    "vector_db_config",
    "config",
    
    # Flows
    "Flow",
    "flow",
    "pipeline",
    "flow_registry",
    
    # Providers
    "Provider",
    "create_provider",
    
    # Version
    "__version__"
]
