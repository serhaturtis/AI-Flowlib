"""
Provider factory for agent system.

This module provides a centralized factory for creating and managing providers
used by the agent system, abstracting away specific provider implementations.
"""

import logging
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field, ConfigDict
# Removed ProviderType import - using config-driven provider access
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.core.factory import create_and_initialize_provider as base_create_provider
from ..core.errors import ProviderError, ConfigurationError

logger = logging.getLogger(__name__)


class VectorMemoryConfig(BaseModel):
    """Vector memory configuration model."""
    model_config = ConfigDict(extra="forbid")
    
    embedding_provider_name: str = Field(default="llamacpp", description="Embedding provider name")
    embedding_settings: Optional[Dict[str, Any]] = Field(default=None, description="Embedding provider settings")
    vector_provider_name: str = Field(default="chroma", description="Vector provider name")
    vector_settings: Optional[Dict[str, Any]] = Field(default=None, description="Vector provider settings")


class KnowledgeMemoryConfig(BaseModel):
    """Knowledge memory configuration model."""
    model_config = ConfigDict(extra="forbid")
    
    graph_provider_name: str = Field(default="memory_graph", description="Graph provider name")
    provider_settings: Optional[Dict[str, Any]] = Field(default=None, description="Provider settings")


class MemoryConfig(BaseModel):
    """Memory configuration model."""
    model_config = ConfigDict(extra="forbid")
    
    vector_memory: Optional[VectorMemoryConfig] = Field(default=None, description="Vector memory configuration")
    knowledge_memory: Optional[KnowledgeMemoryConfig] = Field(default=None, description="Knowledge memory configuration")
    fusion_provider_name: str = Field(default="llamacpp", description="Fusion provider name")
    fusion_settings: Optional[Dict[str, Any]] = Field(default=None, description="Fusion provider settings")


class AgentProviderFactory:
    """Factory for creating providers used by agents."""
    
    # Provider implementation mappings
    PROVIDER_IMPLEMENTATIONS = {
        "llm": {
            "openai": "openai",
            "anthropic": "anthropic",
            "llamacpp": "llamacpp",
            "default": "llamacpp"
        },
        "embedding": {
            "openai": "openai",
            "llamacpp": "llamacpp",
            "default": "llamacpp"
        },
        "vector_db": {
            "chroma": "chroma",
            "pinecone": "pinecone",
            "qdrant": "qdrant",
            "default": "chroma"
        },
        "graph_db": {
            "neo4j": "neo4j",
            "arangodb": "arango",
            "janusgraph": "janus",
            "memory": "memory_graph",
            "default": "memory_graph"
        },
        "cache": {
            "redis": "redis",
            "memory": "memory",
            "default": "memory"
        },
        "database": {
            "mongodb": "mongodb",
            "postgresql": "postgres",
            "sqlite": "sqlite",
            "default": "sqlite"
        }
    }
    
    @classmethod
    async def create_provider(
        cls,
        provider_type: str,
        provider_name: str,
        settings: Optional[Dict[str, Any]] = None,
        register: bool = True
    ) -> Any:
        """Create a provider instance.
        
        Args:
            provider_type: Type of provider to create
            provider_name: Name/implementation of the provider
            settings: Provider-specific settings
            register: Whether to register the provider
            
        Returns:
            Initialized provider instance
            
        Raises:
            ProviderError: If provider creation fails
            ConfigurationError: If configuration is invalid
        """
        try:
            # Get implementation mapping
            implementations = cls.PROVIDER_IMPLEMENTATIONS[provider_type] if provider_type in cls.PROVIDER_IMPLEMENTATIONS else {}
            
            # Resolve implementation name
            implementation = implementations[provider_name] if provider_name in implementations else None
            if not implementation:
                # Try to use provider_name as implementation if not in mapping
                implementation = provider_name
                logger.warning(
                    f"Unknown {provider_type} provider '{provider_name}', "
                    f"attempting to use as implementation name"
                )
            
            # Check if provider already exists using single-argument registry access
            try:
                if provider_registry.contains(provider_name):
                    existing = await provider_registry.get_by_config(provider_name)
                    logger.info(f"Reusing existing provider: {provider_name}")
                    return existing
            except Exception:
                # Provider doesn't exist, continue to create
                pass
            
            # Create provider
            logger.info(f"Creating {provider_type} provider: {provider_name}")
            
            provider = await base_create_provider(
                provider_type=provider_type,
                name=provider_name,
                implementation=implementation,
                register=register,
                **(settings or {})
            )
            
            logger.info(f"Successfully created {provider_type} provider: {provider_name}")
            return provider
            
        except Exception as e:
            raise ProviderError(
                f"Failed to create {provider_type} provider '{provider_name}': {str(e)}",
                provider_name=provider_name,
                operation="create",
                cause=e
            )
    
    @classmethod
    async def create_memory_providers(
        cls,
        memory_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create all providers needed for memory system.
        
        Args:
            memory_config: Memory configuration
            
        Returns:
            Dictionary of created providers
            
        Raises:
            ProviderError: If any provider creation fails
        """
        providers = {}
        
        try:
            # Validate and parse memory configuration
            config = MemoryConfig.model_validate(memory_config)
            
            # Create embedding provider
            if config.vector_memory:
                vec_config = config.vector_memory
                providers["embedding"] = await cls.create_provider(
                    "embedding",
                    vec_config.embedding_provider_name,
                    vec_config.embedding_settings
                )
                
                # Create vector provider
                providers["vector"] = await cls.create_provider(
                    "vector_db",
                    vec_config.vector_provider_name,
                    vec_config.vector_settings
                )
            
            # Create graph provider for knowledge memory
            if config.knowledge_memory:
                kg_config = config.knowledge_memory
                providers["graph"] = await cls.create_provider(
                    "graph_db",
                    kg_config.graph_provider_name,
                    kg_config.provider_settings
                )
            
            # Create LLM provider for fusion
            if config.fusion_provider_name != "llamacpp" or config.fusion_settings:
                providers["fusion_llm"] = await cls.create_provider(
                    "llm",
                    config.fusion_provider_name,
                    config.fusion_settings
                )
            
            return providers
            
        except Exception as e:
            # Clean up any created providers on failure
            for provider in providers.values():
                try:
                    if hasattr(provider, "shutdown"):
                        await provider.shutdown()
                except Exception:
                    pass
            
            raise ProviderError(
                "Failed to create memory providers",
                operation="create_memory_providers",
                cause=e
            )
    
    @classmethod
    def get_default_settings(
        cls,
        provider_type: str,
        provider_name: str
    ) -> Dict[str, Any]:
        """Get default settings for a provider.
        
        Args:
            provider_type: Type of provider
            provider_name: Name of the provider
            
        Returns:
            Default settings dictionary
        """
        # Default settings per provider type and implementation
        defaults = {
            "llm": {
                "llamacpp": {
                    "model_path": "/path/to/model.gguf",
                    "n_ctx": 4096,
                    "n_gpu_layers": -1,
                    "temperature": 0.7
                },
                "openai": {
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            },
            "vector_db": {
                "chroma": {
                    "persist_directory": "./chroma_db",
                    "collection_name": "agent_memory"
                },
                "qdrant": {
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": "agent_memory"
                }
            },
            "graph_db": {
                "neo4j": {
                    "uri": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password"
                },
                "memory_graph": {
                    "max_nodes": 10000,
                    "max_edges": 50000
                }
            }
        }
        
        provider_defaults = defaults[provider_type] if provider_type in defaults else {}
        return provider_defaults[provider_name] if provider_name in provider_defaults else {}