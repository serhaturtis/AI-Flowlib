"""Model configuration utilities for standardized agent models.

This module provides utilities for managing the standardized agent-model-small
and agent-model-large configurations throughout the agent system.
"""

import logging
from typing import Dict, Any, Optional
from flowlib.resources.registry.registry import resource_registry
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.decorators.decorators import model_config
from flowlib.resources.models.constants import ResourceType

logger = logging.getLogger(__name__)


def ensure_standard_models_registered():
    """Ensure both agent-model-small and agent-model-large are registered.
    
    This function checks if the standard models are registered. If only one
    is registered, it creates the missing one using the same configuration.
    If neither is registered, it logs a warning.
    
    This supports the pattern where users can configure just one model
    and have both standard names point to it.
    """
    try:
        # Check what's already registered
        has_small = resource_registry.contains("agent-model-small")
        has_large = resource_registry.contains("agent-model-large")
        
        if has_small and has_large:
            # Both registered, nothing to do
            logger.debug("Both agent-model-small and agent-model-large are already registered")
            return
        
        if not has_small and not has_large:
            # Neither registered - this is handled by the REPL or app initialization
            logger.warning("Neither agent-model-small nor agent-model-large are registered. "
                         "Please configure at least one model in your initialization code.")
            return
        
        # One is registered, create the other with the same config
        if has_small and not has_large:
            # Copy small to large
            small_model = resource_registry.get("agent-model-small")
            if small_model:
                logger.info("Creating agent-model-large from agent-model-small configuration")
                
                # Create a simple copy by copying the small model's data
                # and changing the name
                small_data = small_model.model_dump()
                small_data['name'] = "agent-model-large"
                
                # Create a new instance of the same class as small_model
                # but with the large name
                large_instance = small_model.__class__(**small_data)
                
                # Register manually since decorator already fired
                resource_registry.register(
                    name="agent-model-large",
                    obj=large_instance,
                    resource_type=ResourceType.MODEL_CONFIG
                )
                
        elif has_large and not has_small:
            # Copy large to small
            large_model = resource_registry.get("agent-model-large")
            if large_model:
                logger.info("Creating agent-model-small from agent-model-large configuration")
                
                # Create a simple copy by copying the large model's data
                # and changing the name
                large_data = large_model.model_dump()
                large_data['name'] = "agent-model-small"
                
                # Create a new instance of the same class as large_model
                # but with the small name
                small_instance = large_model.__class__(**large_data)
                
                # Register manually since decorator already fired
                resource_registry.register(
                    name="agent-model-small",
                    obj=small_instance,
                    resource_type=ResourceType.MODEL_CONFIG
                )
                
    except Exception as e:
        logger.error(f"Error ensuring standard models are registered: {e}")


def get_model_config(model_name: str = "agent-model-small") -> Optional[Dict[str, Any]]:
    """Get configuration for a standard model.
    
    Args:
        model_name: Either "agent-model-small" or "agent-model-large"
        
    Returns:
        Model configuration dictionary or None if not found
    """
    try:
        model_resource = resource_registry.get(model_name)
        if model_resource:
            # Convert resource to config dict
            config = {}
            for attr in dir(model_resource):
                if not attr.startswith('_') and attr not in ['name', 'type']:
                    value = getattr(model_resource, attr, None)
                    if value is not None and not callable(value):
                        config[attr] = value
            return config
        return None
    except Exception as e:
        logger.error(f"Error getting model config for {model_name}: {e}")
        return None


# Model usage guidelines
MODEL_USAGE_GUIDE = """
Agent Model Usage Guidelines:

1. agent-model-small: Use for quick, simple tasks
   - Conversations and chat responses
   - Simple shell commands
   - Quick classifications
   - Fast iterations where speed matters

2. agent-model-large: Use for complex, thoughtful tasks
   - Planning multi-step operations
   - Deep reflection and analysis
   - Complex reasoning tasks
   - Quality over speed scenarios

The system automatically ensures both models are available.
If you configure only one, both names will use that configuration.
"""