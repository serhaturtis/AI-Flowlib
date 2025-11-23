"""Example alias bindings for Flowlib projects.

Copy this file to configs/aliases.py and adjust the canonical names to match
your provider and agent configuration resources. Alias bindings MUST exist
for every semantic identifier referenced by agents (e.g., RequiredAlias.DEFAULT_LLM.value).
"""

import logging

from flowlib.config.alias_manager import alias_manager
from flowlib.config.required_resources import RequiredAlias

logger = logging.getLogger(__name__)

try:
    # Provider aliases
    alias_manager.assign_alias(RequiredAlias.DEFAULT_LLM.value, "example-llamacpp-provider")
    alias_manager.assign_alias(RequiredAlias.DEFAULT_EMBEDDING.value, "example-llamacpp-embedding-provider")
    alias_manager.assign_alias(RequiredAlias.DEFAULT_VECTOR_DB.value, "example-vector-db-provider")
    alias_manager.assign_alias(RequiredAlias.DEFAULT_GRAPH_DB.value, "example-graph-db-provider")
    alias_manager.assign_alias(RequiredAlias.DEFAULT_DATABASE.value, "example-database-provider")
    alias_manager.assign_alias(RequiredAlias.DEFAULT_CACHE.value, "example-cache-provider")

    # Model aliases
    alias_manager.assign_alias(RequiredAlias.DEFAULT_MODEL.value, "example-model-config")
    alias_manager.assign_alias(RequiredAlias.DEFAULT_EMBEDDING_MODEL.value, "example-embedding-model-config")
    alias_manager.assign_alias("agent-embedding-model", "example-embedding-model-config")

    # Agent configuration aliases
    alias_manager.assign_alias("default-agent-config", "default-agent-config")
    alias_manager.assign_alias("creative-agent", "creative-agent-config")
    alias_manager.assign_alias("precise-agent", "precise-agent-config")
    alias_manager.assign_alias("devops-agent", "devops-agent-config")

    logger.info("Example alias bindings registered")
except Exception as exc:
    logger.error(f"Failed to register example aliases: {exc}")
    raise

