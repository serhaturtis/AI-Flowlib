"""
Simplified decorator API for agent system.

This module provides streamlined decorators for creating agent classes and flows,
with reduced complexity and better alignment with flowlib patterns.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

from ....flows.decorators import flow
from ....flows.base import Flow
from ....flows.registry import flow_registry

from ...registry import agent_registry
from ...core.agent import AgentCore
from ...models.config import AgentConfig

logger = logging.getLogger(__name__)

def agent(
    name: str = None,
    provider_name: str = "llamacpp",
    model_name: str = None,
    **kwargs
) -> Callable:
    """Decorator for creating agent classes with minimal complexity.
    
    Args:
        name: Agent name
        provider_name: Name of LLM provider to use
        model_name: Model to use (for planning, reflection, etc.)
        **kwargs: Additional configuration options
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        # Create a wrapper class that extends AgentCore for core functionality
        class AgentWrapper(AgentCore):
            """Agent class created with @agent decorator."""
            
            def __init__(
                self,
                flows: List[Flow] = None,
                config: Optional[Union[Dict[str, Any], AgentConfig]] = None,
                task_description: str = "",
                **init_kwargs
            ):
                """Initialize the agent."""
                # Build simplified configuration
                config_dict = {}
                
                # Set name
                config_dict["name"] = name or cls.__name__
                
                # Set provider
                if provider_name:
                    config_dict["provider_name"] = provider_name
                
                # Set model name for all components
                if model_name:
                    config_dict["model_name"] = model_name
                
                # Set default persona if not provided
                if "persona" not in kwargs:
                    config_dict["persona"] = f"A helpful AI agent named {config_dict['name']}"
                
                # Separate known AgentConfig fields from additional kwargs
                known_fields = {
                    "name", "persona", "provider_name", "model_name", "description", "system_prompt",
                    "max_iterations", "timeout", "temperature", "conversation", "memory", "reflection",
                    "tasks", "performance", "resilience", "integration", "flows", "persistence",
                    "mcp_integration", "dual_path", "activity", "batch_operations", "task_id",
                    "task_description", "engine_config", "planner_config", "reflection_config",
                    "memory_config", "state_config", "enable_memory", "provider_config", "components",
                    "resource_config"
                }
                
                additional_settings = {}
                for key, value in kwargs.items():
                    if key in known_fields:
                        config_dict[key] = value
                    else:
                        additional_settings[key] = value
                
                # Add additional settings to the config
                if additional_settings:
                    config_dict["additional_settings"] = additional_settings
                
                # If config is provided, merge with our config
                if config:
                    if not isinstance(config, (dict, AgentConfig)):
                        raise TypeError(f"Config must be a dict or AgentConfig instance, got {type(config)}")
                    
                    if isinstance(config, dict):
                        # Separate known and unknown fields from provided config
                        for key, value in config.items():
                            if key not in config_dict:  # Don't override existing values
                                if key in known_fields:
                                    config_dict[key] = value
                                else:
                                    # Add to additional_settings
                                    if "additional_settings" not in config_dict:
                                        config_dict["additional_settings"] = {}
                                    config_dict["additional_settings"][key] = value
                    elif isinstance(config, AgentConfig):
                        # Convert to dict and merge
                        config_data = config.model_dump()
                        for key, value in config_data.items():
                            if key not in config_dict:
                                if key == "additional_settings" and isinstance(value, dict):
                                    # Merge additional_settings
                                    if "additional_settings" not in config_dict:
                                        config_dict["additional_settings"] = {}
                                    config_dict["additional_settings"].update(value)
                                else:
                                    config_dict[key] = value
                
                # Create final config
                effective_config = AgentConfig(**config_dict)
                
                # Initialize AgentCore
                super().__init__(
                    config=effective_config,
                    task_description=task_description
                )
                
                # Create implementation instance
                self._impl = cls(**init_kwargs)
                
                # Give implementation access to the agent
                if hasattr(self._impl, "set_agent"):
                    self._impl.set_agent(self)
                else:
                    self._impl.agent = self
                
                # Register flows if provided
                if flows:
                    for flow in flows:
                        self.register_flow(flow)
            
            def get_flows(self) -> Dict[str, Any]:
                """Get the dictionary of registered flows.
                
                Returns:
                    Dictionary of flow name to flow instance
                """
                return self.flows
            
            # Forward method calls to the implementation class
            def __getattr__(self, name):
                """Forward attribute access to the implementation class.
                
                This allows methods defined on the implementation class to be 
                called directly on the agent instance.
                
                Args:
                    name: Name of the attribute to retrieve
                    
                Returns:
                    Attribute from the implementation class
                    
                Raises:
                    AttributeError: If attribute not found
                """
                if hasattr(self._impl, name):
                    return getattr(self._impl, name)
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Add the create class method to the class
        @classmethod
        def create(cls, *args, **kwargs):
            """Create an agent instance."""
            return AgentWrapper(*args, **kwargs)
        
        # Set the create method on the class
        cls.create = create
        
        # Store the wrapper class
        cls.__agent_class__ = AgentWrapper

        # === Register the AgentWrapper class ===
        agent_name = name or cls.__name__
        # Construct metadata for registration
        agent_metadata = {
            "provider_name": provider_name,
            "model_name": model_name,
            "agent_type": "core", # Indicate it uses AgentCore
            **kwargs
        }
        agent_registry.register(name=agent_name, agent_class=AgentWrapper, metadata=agent_metadata)
        # ========================================

        return cls
    
    return decorator

def agent_flow(
    name: str = None,
    description: str = None,
    category: str = "agent",
    is_infrastructure: bool = False
) -> Callable:
    """Decorator for creating agent-specific flows.
    
    This extends the standard @flow decorator with agent-specific metadata
    and registers the flow with flow_registry for discovery.
    
    Args:
        name: Flow name
        description: Human-readable description of the flow
        category: Category for the flow (conversation, tool, etc.)
        is_infrastructure: Whether this is an internal flow
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        # Add agent-specific metadata without full flow decorator for now
        flow_name = name or cls.__name__
        
        # Add flow metadata
        if not hasattr(cls, "__flow_metadata__"):
            cls.__flow_metadata__ = {}
            
        cls.__flow_metadata__.update({
            "name": flow_name,
            "category": category,
            "agent_flow": True
        })
        
        # Add infrastructure flag if specified
        if is_infrastructure:
            cls.__is_infrastructure__ = is_infrastructure
        
        # Add get_description method if description is provided and class doesn't have one
        if description and not hasattr(cls, 'get_description'):
            def get_description(self):
                return description
            cls.get_description = get_description
        elif not hasattr(cls, 'get_description'):
            # Use docstring or empty string
            def get_description(self):
                return self.__class__.__doc__ or ""
            cls.get_description = get_description
        
        # Register with flow_registry for better discovery
        try:
            if flow_registry:
                logger.debug(f"Registering agent flow '{flow_name}' with flow_registry")
                flow_registry.register_flow(flow_name, cls)
        except Exception as e:
            logger.warning(f"Failed to register agent flow with flow_registry: {str(e)}")
        
        return cls
        
    return decorator

def dual_path_agent(
    name: str,
    description: str,
    **kwargs
) -> Callable:
    """Decorator for registering DualPathAgent classes.
    
    Args:
        name: Agent name
        description: Human-readable description of the agent
        **kwargs: Additional metadata for the agent
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        # Basic validation (optional, uncomment if needed)
        # from ..core.base import DualPathAgent 
        # if not issubclass(cls, DualPathAgent):
        #     raise TypeError(f"Class {cls.__name__} must inherit from DualPathAgent to use @dual_path_agent")

        # Assign metadata attributes
        setattr(cls, '__agent_name__', name)
        setattr(cls, '__agent_description__', description)
        
        # Combine base metadata with kwargs
        metadata = {'agent_type': 'dual_path', **kwargs}
        setattr(cls, '__agent_metadata__', metadata)

        # Register the decorated class directly with the agent_registry
        try:
            agent_registry.register(name=name, agent_class=cls, metadata=metadata)
        except Exception as e:
            logger.warning(f"Failed to register dual path agent '{name}': {str(e)}")

        logger.debug(f"Decorated class {cls.__name__} as DualPathAgent with name='{name}'")
        return cls
    return decorator 