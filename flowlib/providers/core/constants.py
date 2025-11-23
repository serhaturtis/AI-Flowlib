"""Clean constants for the config-driven provider system.

This module defines only the essential constants needed for the clean,
config-driven provider architecture. ProviderType enum has been removed
as all provider access is now config-driven.
"""

from flowlib.config.required_resources import RequiredAlias

# Provider categories (used internally by the registry)
PROVIDER_CATEGORIES = {
    "llm",
    "multimodal_llm",
    "vector_db",
    "database",
    "cache",
    "storage",
    "message_queue",
    "gpu",
    "api",
    "graph_db",
    "embedding",
    "state_persister",
    "mcp_client",
    "mcp_server",
}

# Default provider configuration names
DEFAULT_CONFIGS = {
    RequiredAlias.DEFAULT_LLM.value: "llm",
    "default-multimodal-llm": "multimodal_llm",
    RequiredAlias.DEFAULT_VECTOR_DB.value: "vector_db",
    RequiredAlias.DEFAULT_GRAPH_DB.value: "graph_db",
    RequiredAlias.DEFAULT_CACHE.value: "cache",
    RequiredAlias.DEFAULT_EMBEDDING.value: "embedding",
    RequiredAlias.DEFAULT_DATABASE.value: "database",
}

# Provider type mapping - only providers that actually exist in the codebase
PROVIDER_TYPE_MAP = {
    "llamacpp": "llm",
    "llamacpp_multimodal": "multimodal_llm",
    "google_ai": "llm",
    "postgres": "database",
    "mongodb": "database",
    "sqlite": "database",
    "chroma": "vector_db",
    "pinecone": "vector_db",
    "qdrant": "vector_db",
    "redis": "cache",
    "memory": "cache",
    "s3": "storage",
    "local": "storage",
    "llamacpp_embedding": "embedding",
    "neo4j": "graph_db",
    "arango": "graph_db",
    "janusgraph": "graph_db",
    "rabbitmq": "message_queue",
    "kafka": "message_queue",
}
