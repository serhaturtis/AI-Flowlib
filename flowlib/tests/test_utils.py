"""Test utilities for loading resources and providers.

This module provides utilities to set up test resources and providers
for consistent testing across the flowlib framework.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List

from flowlib.resources.registry.registry import resource_registry
from flowlib.resources.models.config_resource import (
    EmbeddingConfigResource, LLMConfigResource, VectorDBConfigResource,
    DatabaseConfigResource, CacheConfigResource, StorageConfigResource,
    GraphDBConfigResource, MessageQueueConfigResource
)

logger = logging.getLogger(__name__)


def load_test_resources() -> None:
    """Load test resources from the test_resources.yaml file.
    
    This function loads resources needed for testing, including the
    default-embedding resource that vector providers expect.
    
    It checks for resources in this order:
    1. FLOWLIB_RESOURCE_PATH environment variable
    2. Default test_resources.yaml file in tests directory
    """
    # Check if FLOWLIB_RESOURCE_PATH is set
    resource_path_env = os.getenv('FLOWLIB_RESOURCE_PATH')
    if resource_path_env:
        test_resources_path = Path(resource_path_env)
        logger.info(f"Using resource path from FLOWLIB_RESOURCE_PATH: {test_resources_path}")
    else:
        test_resources_path = Path(__file__).parent / "test_resources.yaml"
    
    if not test_resources_path.exists():
        logger.warning(f"Test resources file not found: {test_resources_path}")
        return
    
    try:
        with open(test_resources_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if not data or 'resources' not in data:
            logger.warning("No resources section found in test_resources.yaml")
            return
        
        for resource_config in data['resources']:
            name = resource_config.get('name')
            resource_type = resource_config.get('type')
            provider_type = resource_config.get('provider_type', 'mock')
            settings = resource_config.get('settings', {})
            
            if not name or not resource_type:
                logger.warning(f"Incomplete resource config: {resource_config}")
                continue
            
            # Create the appropriate config resource based on type
            # All extra fields should be in settings dict due to extra="forbid"
            if resource_type == 'embedding_provider':
                resource = EmbeddingConfigResource(
                    name=name,
                    type="embedding_config",
                    provider_type=provider_type,
                    settings=settings
                )
            elif resource_type == 'llm_provider':
                resource = LLMConfigResource(
                    name=name,
                    type="llm_config",
                    provider_type=provider_type,
                    settings=settings
                )
            elif resource_type == 'vector_db_provider':
                resource = VectorDBConfigResource(
                    name=name,
                    type="vector_db_config",
                    provider_type=provider_type,
                    settings=settings
                )
            else:
                # For other types, use the base config
                from flowlib.resources.models.config_resource import ProviderConfigResource
                resource = ProviderConfigResource(
                    name=name,
                    type="provider_config",
                    provider_type=provider_type,
                    settings=settings
                )
            
            # Register the resource
            resource_registry.register(name, resource)
            logger.info(f"Loaded test resource: {name} ({resource_type})")
            
    except Exception as e:
        logger.error(f"Failed to load test resources: {e}")


def setup_test_environment() -> None:
    """Set up the test environment with required resources.
    
    This is a convenience function to set up all test resources
    and can be called from test setup fixtures.
    """
    load_test_resources()


def create_mock_embedding_resource() -> None:
    """Create a mock embedding resource for testing.
    
    This creates the default-embedding resource that vector providers expect.
    """
    try:
        # Check if it already exists
        if resource_registry.contains("default-embedding"):
            logger.debug("default-embedding resource already exists")
            return
        
        # Create mock embedding config
        mock_settings = {
            "embedding_dim": 384,
            "normalize": True,
            "batch_size": 32
        }
        resource = EmbeddingConfigResource(
            name="default-embedding",
            type="embedding_config",
            provider_type="mock",
            settings=mock_settings,
            dimensions=384,
            batch_size=32,
            normalize=True
        )
        
        # Register it
        resource_registry.register("default-embedding", resource)
        logger.info("Created mock default-embedding resource")
        
    except Exception as e:
        logger.error(f"Failed to create mock embedding resource: {e}")