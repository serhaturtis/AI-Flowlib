"""Flowlib resource management system public API."""

from .models.base import ResourceBase
from .registry.registry import ResourceRegistry, resource_registry
from .models.constants import ResourceType
from .decorators.decorators import (
    resource, model_config, prompt, config,
    llm_config, database_config, vector_db_config,
    cache_config, storage_config, embedding_config,
    graph_db_config, message_queue_config
)
from .models.model_resource import ModelResource
from .models.template_resource import TemplateResource, TemplateVariableConfig

# Templates are now managed directly in GUI template system

__all__ = [
    'ResourceBase',
    'ResourceRegistry',
    'resource_registry', 
    'ResourceType',
    'resource',
    'model_config',
    'prompt',
    'config',
    'llm_config',
    'database_config',
    'vector_db_config',
    'cache_config',
    'storage_config',
    'embedding_config',
    'graph_db_config',
    'message_queue_config',
    'ModelResource',
    'TemplateResource',
    'TemplateVariableConfig',
]