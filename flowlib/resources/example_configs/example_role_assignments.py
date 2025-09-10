"""Example role assignments for Flowlib.

This file creates role assignments mapping standard semantic roles 
to the example configurations that were copied during initialization.

You can modify these assignments to point to different configurations.
After making changes, restart your Flowlib application.

To use different configurations:
1. Create your own config files with descriptive names
2. Update the role assignments below to point to your config names
3. Restart your Flowlib application
"""

from flowlib.config.role_manager import role_manager
import logging

logger = logging.getLogger(__name__)

# Create role assignments that map to the copied example configurations
# These map semantic role names to the actual configuration names

try:
    # LLM Provider Roles - maps to example_llamacpp_provider.py
    role_manager.assign_role("default-llm", "example-llamacpp-provider")
    logger.info("Assigned role 'default-llm' to 'example-llamacpp-provider'")
    
    # Embedding Provider Roles - maps to example_llamacpp_embedding_provider.py
    role_manager.assign_role("default-embedding", "example-llamacpp-embedding-provider")
    logger.info("Assigned role 'default-embedding' to 'example-llamacpp-embedding-provider'")
    
    # Model Resource Aliases - maps standard role names to internal model names
    # Note: agent-model-small and agent-model-large are deprecated - use default-model instead 
    role_manager.assign_role("agent-embedding-model", "example-llamacpp-embedding-provider")
    logger.info("Assigned internal model aliases to standard providers")
    
    # Database Provider Roles - maps to example_database_provider.py
    # role_manager.assign_role("default-database", "example-database-provider")
    # logger.info("Assigned role 'default-database' to 'example-database-provider'")
    
    # Vector Database Provider Roles - maps to example_vector_db_provider.py
    # role_manager.assign_role("default-vector-db", "example-vector-db-provider")
    # logger.info("Assigned role 'default-vector-db' to 'example-vector-db-provider'")
    
    # Graph Database Provider Roles - maps to example_graph_db_provider.py
    # role_manager.assign_role("default-graph-db", "example-graph-db-provider")
    # logger.info("Assigned role 'default-graph-db' to 'example-graph-db-provider'")
    
    # Cache Provider Roles - maps to example_cache_provider.py
    # role_manager.assign_role("default-cache", "example-cache-provider")
    # logger.info("Assigned role 'default-cache' to 'example-cache-provider'")
    
    logger.info("Role assignments completed successfully")
    
except Exception as e:
    logger.warning(f"Failed to create some role assignments: {e}")
    # Don't raise - this is not critical for basic operation