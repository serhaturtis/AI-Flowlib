"""
Required Resource Definitions for Flowlib.

This module defines the canonical set of required aliases that core Flowlib
components depend on, along with their specifications and validation metadata.
"""

from enum import Enum

from pydantic import BaseModel

from flowlib.resources.models.constants import ResourceType


class RequiredAlias(str, Enum):
    """Canonical required aliases in Flowlib.

    These are the semantic names that core library components use.
    Projects must provide bindings for all REQUIRED aliases.
    """

    # Core LLM Resources (REQUIRED)
    DEFAULT_LLM = "default-llm"
    DEFAULT_MODEL = "default-model"

    # Memory System Resources (REQUIRED if memory enabled)
    DEFAULT_EMBEDDING = "default-embedding"
    DEFAULT_EMBEDDING_MODEL = "default-embedding-model"
    DEFAULT_VECTOR_DB = "default-vector-db"
    DEFAULT_GRAPH_DB = "default-graph-db"

    # Optional Performance Aliases
    FAST_LLM = "fast-llm"
    QUALITY_LLM = "quality-llm"

    # Optional Storage Aliases
    DEFAULT_CACHE = "default-cache"
    DEFAULT_DATABASE = "default-database"


class RequiredResourceSpec(BaseModel):
    """Specification for a required resource."""

    alias: RequiredAlias
    resource_type: ResourceType
    required_by: list[str]  # Component names that require this
    required_always: bool = False  # True = always needed, False = conditional
    description: str
    validation_hints: str = ""  # Help text for users


# Registry of required resources
REQUIRED_RESOURCE_SPECS: dict[RequiredAlias, RequiredResourceSpec] = {
    RequiredAlias.DEFAULT_LLM: RequiredResourceSpec(
        alias=RequiredAlias.DEFAULT_LLM,
        resource_type=ResourceType.LLM_CONFIG,
        required_by=["BaseAgent", "TaskPlanning", "TaskEvaluation", "TaskReflection"],
        required_always=True,
        description="Primary language model for agent reasoning and task execution",
        validation_hints="Must be a provider with LLM capabilities (e.g., llamacpp, openai)",
    ),
    RequiredAlias.DEFAULT_MODEL: RequiredResourceSpec(
        alias=RequiredAlias.DEFAULT_MODEL,
        resource_type=ResourceType.MODEL_CONFIG,
        required_by=["BaseAgent"],
        required_always=True,
        description="Model configuration specifying which model to use with default-llm",
        validation_hints="Must reference a model compatible with your default-llm provider",
    ),
    RequiredAlias.DEFAULT_EMBEDDING: RequiredResourceSpec(
        alias=RequiredAlias.DEFAULT_EMBEDDING,
        resource_type=ResourceType.EMBEDDING_CONFIG,
        required_by=["MemoryManager", "VectorMemoryComponent"],
        required_always=False,  # Only if memory enabled
        description="Embedding provider for vector memory and semantic search",
        validation_hints="Required if agent uses vector memory. Can be same as default-llm if multimodal",
    ),
    RequiredAlias.DEFAULT_EMBEDDING_MODEL: RequiredResourceSpec(
        alias=RequiredAlias.DEFAULT_EMBEDDING_MODEL,
        resource_type=ResourceType.MODEL_CONFIG,
        required_by=["MemoryManager", "VectorMemoryComponent"],
        required_always=False,
        description="Model configuration for embedding provider",
        validation_hints="Required if agent uses vector memory. Must be compatible with default-embedding",
    ),
    RequiredAlias.DEFAULT_VECTOR_DB: RequiredResourceSpec(
        alias=RequiredAlias.DEFAULT_VECTOR_DB,
        resource_type=ResourceType.VECTOR_DB_CONFIG,
        required_by=["MemoryManager", "VectorMemoryComponent"],
        required_always=False,
        description="Vector database for storing embeddings and semantic memory",
        validation_hints="Required if agent uses vector memory (e.g., ChromaDB, Qdrant)",
    ),
    RequiredAlias.DEFAULT_GRAPH_DB: RequiredResourceSpec(
        alias=RequiredAlias.DEFAULT_GRAPH_DB,
        resource_type=ResourceType.GRAPH_DB_CONFIG,
        required_by=["MemoryManager", "KnowledgeMemoryComponent"],
        required_always=False,
        description="Graph database for knowledge graph and entity relationships",
        validation_hints="Required if agent uses knowledge memory (e.g., Neo4j, TigerGraph)",
    ),
    RequiredAlias.FAST_LLM: RequiredResourceSpec(
        alias=RequiredAlias.FAST_LLM,
        resource_type=ResourceType.LLM_CONFIG,
        required_by=[],
        required_always=False,
        description="Optional: Fast LLM for quick tasks",
        validation_hints="Optional performance optimization for tasks requiring speed",
    ),
    RequiredAlias.QUALITY_LLM: RequiredResourceSpec(
        alias=RequiredAlias.QUALITY_LLM,
        resource_type=ResourceType.LLM_CONFIG,
        required_by=[],
        required_always=False,
        description="Optional: High-quality LLM for complex reasoning",
        validation_hints="Optional performance optimization for tasks requiring accuracy",
    ),
    RequiredAlias.DEFAULT_CACHE: RequiredResourceSpec(
        alias=RequiredAlias.DEFAULT_CACHE,
        resource_type=ResourceType.CACHE_CONFIG,
        required_by=[],
        required_always=False,
        description="Optional: Cache provider for response caching",
        validation_hints="Optional performance optimization for caching LLM responses",
    ),
    RequiredAlias.DEFAULT_DATABASE: RequiredResourceSpec(
        alias=RequiredAlias.DEFAULT_DATABASE,
        resource_type=ResourceType.DATABASE_CONFIG,
        required_by=[],
        required_always=False,
        description="Optional: Database for persistent storage",
        validation_hints="Optional for storing structured data and application state",
    ),
}


def get_always_required() -> list[RequiredAlias]:
    """Get list of aliases that are always required."""
    return [spec.alias for spec in REQUIRED_RESOURCE_SPECS.values() if spec.required_always]


def get_conditionally_required() -> list[RequiredAlias]:
    """Get list of aliases that are conditionally required."""
    return [spec.alias for spec in REQUIRED_RESOURCE_SPECS.values() if not spec.required_always]


def get_required_for_component(component_name: str) -> list[RequiredAlias]:
    """Get aliases required by a specific component."""
    return [
        spec.alias
        for spec in REQUIRED_RESOURCE_SPECS.values()
        if component_name in spec.required_by
    ]
