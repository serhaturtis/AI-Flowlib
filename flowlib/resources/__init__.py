"""Flowlib resource management system public API."""

from flowlib.resources.decorators.decorators import (
    agent_profile_config,
    cache_config,
    config,
    database_config,
    embedding_config,
    graph_db_config,
    llm_config,
    message_queue_config,
    model_config,
    prompt,
    resource,
    storage_config,
    vector_db_config,
)
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.models.constants import ResourceType
from flowlib.resources.models.model_resource import ModelResource
from flowlib.resources.models.template_resource import (
    TemplateResource,
    TemplateVariableConfig,
)
from flowlib.resources.registry.registry import ResourceRegistry, resource_registry

# Templates are now managed directly in GUI template system

__all__ = [
    "ResourceBase",
    "ResourceRegistry",
    "resource_registry",
    "ResourceType",
    "resource",
    "model_config",
    "prompt",
    "config",
    "llm_config",
    "database_config",
    "vector_db_config",
    "cache_config",
    "storage_config",
    "embedding_config",
    "graph_db_config",
    "message_queue_config",
    "agent_profile_config",
    "ModelResource",
    "TemplateResource",
    "TemplateVariableConfig",
]
