"""Core dependency management system eliminating circular dependencies.

This module provides the foundational dependency management system that
completely eliminates circular dependencies through:

1. Central dependency container
2. Event-driven registration 
3. Interface-based dependencies
4. Dynamic loading
5. Clean decorators

All imports are at the top of files, no circular dependencies possible.
"""

# Core interfaces - no dependencies
from .interfaces.interfaces import (
    Provider, LLMProvider, VectorProvider, GraphProvider, 
    DatabaseProvider, CacheProvider, ToolProvider, Resource, Configuration,
    PromptResource, ModelResource, Flow, AgentFlow, Stage,
    Memory, Planning, Factory, Container
)

# Event system - no circular dependencies
from .events.events import (
    EventBus, RegistrationEvent, ConfigurationEvent,
    get_event_bus, emit_registration, emit_configuration,
    set_global_container, get_event_bus_stats
)

# Dynamic loader - no circular dependencies
from .loader.loader import DynamicLoader

# Central container - no circular dependencies
from .container.container import DependencyContainer, get_container, set_global_container

# Clean decorators - no circular dependencies
from .decorators.decorators import (
    provider, resource, flow, config, tool,
    llm_config, database_config, vector_db_config, graph_db_config,
    cache_config, embedding_config, model, prompt, template,
    singleton, lazy_init, inject
)

# Strict base models - enforce CLAUDE.md principles
from .models import (
    StrictBaseModel, 
    MutableStrictBaseModel,
    StrictModel,
    MutableStrictModel
)

# Initialize the system
def initialize_dependency_system():
    """Initialize the dependency management system.
    
    This function sets up the event bus and container connection,
    enabling event-driven registration without circular dependencies.
    """
    container = get_container()
    event_bus = get_event_bus()
    
    # Connect event bus to container
    set_global_container(container)
    
    return container, event_bus

# Auto-initialize when module is imported
_container, _event_bus = initialize_dependency_system()

# Export the initialized instances (renamed to avoid conflicts with submodules)
dependency_container = _container
event_bus = _event_bus

__all__ = [
    # Interfaces
    'Provider', 'LLMProvider', 'VectorProvider', 'GraphProvider',
    'DatabaseProvider', 'CacheProvider', 'ToolProvider', 'Resource', 'Configuration',
    'PromptResource', 'ModelResource', 'Flow', 'AgentFlow', 'Stage',
    'Memory', 'Planning', 'Factory', 'Container',
    
    # Events
    'EventBus', 'RegistrationEvent', 'ConfigurationEvent',
    'get_event_bus', 'emit_registration', 'emit_configuration',
    'get_event_bus_stats',
    
    # Dynamic loading
    'DynamicLoader',
    
    # Container
    'DependencyContainer', 'get_container', 'dependency_container',
    
    # Decorators
    'provider', 'resource', 'flow', 'config', 'tool',
    'llm_config', 'database_config', 'vector_db_config', 'graph_db_config',
    'cache_config', 'embedding_config', 'model', 'prompt', 'template',
    'singleton', 'lazy_init', 'inject',
    
    # Strict Models - CLAUDE.md compliance
    'StrictBaseModel', 'MutableStrictBaseModel', 'StrictModel', 'MutableStrictModel',
    
    # System
    'initialize_dependency_system', 'event_bus'
]