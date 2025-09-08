"""Flowlib resource management system public API."""

from flowlib.resources.models.base import ResourceBase
from flowlib.resources.registry.registry import ResourceRegistry, resource_registry
from flowlib.resources.models.constants import ResourceType
from flowlib.resources.decorators.decorators import (
    resource, model_config, prompt, config,
    llm_config, database_config, vector_db_config,
    cache_config, storage_config, embedding_config,
    graph_db_config, message_queue_config
)
from flowlib.resources.models.model_resource import ModelResource
from flowlib.resources.models.template_resource import TemplateResource, TemplateVariableConfig

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