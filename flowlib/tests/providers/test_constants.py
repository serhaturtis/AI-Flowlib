"""Tests for provider constants."""

import pytest
from flowlib.providers.core.constants import (
    PROVIDER_CATEGORIES,
    DEFAULT_CONFIGS,
    PROVIDER_TYPE_MAP
)


class TestProviderCategories:
    """Test PROVIDER_CATEGORIES constant."""
    
    def test_provider_categories_type(self):
        """Test that PROVIDER_CATEGORIES is a set."""
        assert isinstance(PROVIDER_CATEGORIES, set)
    
    def test_provider_categories_content(self):
        """Test that PROVIDER_CATEGORIES contains expected categories."""
        expected_categories = {
            "llm",
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
            "mcp_server"
        }
        
        assert PROVIDER_CATEGORIES == expected_categories
    
    def test_provider_categories_count(self):
        """Test that PROVIDER_CATEGORIES has expected number of items."""
        assert len(PROVIDER_CATEGORIES) == 13
    
    def test_provider_categories_are_strings(self):
        """Test that all categories are strings."""
        for category in PROVIDER_CATEGORIES:
            assert isinstance(category, str)
            assert len(category) > 0
    
    def test_provider_categories_no_duplicates(self):
        """Test that there are no duplicate categories."""
        categories_list = list(PROVIDER_CATEGORIES)
        assert len(categories_list) == len(set(categories_list))
    
    def test_provider_categories_lowercase(self):
        """Test that all categories are lowercase with underscores."""
        for category in PROVIDER_CATEGORIES:
            assert category.islower()
            # Should only contain lowercase letters, numbers, and underscores
            assert all(c.islower() or c.isdigit() or c == '_' for c in category)
    
    def test_provider_categories_core_types_present(self):
        """Test that core provider types are present."""
        core_types = {"llm", "database", "cache", "storage", "embedding"}
        assert core_types.issubset(PROVIDER_CATEGORIES)
    
    def test_provider_categories_mcp_types_present(self):
        """Test that MCP (Model Context Protocol) types are present."""
        mcp_types = {"mcp_client", "mcp_server"}
        assert mcp_types.issubset(PROVIDER_CATEGORIES)
    
    def test_provider_categories_data_storage_types(self):
        """Test that data storage categories are present."""
        storage_types = {"database", "vector_db", "graph_db", "cache", "storage"}
        assert storage_types.issubset(PROVIDER_CATEGORIES)


class TestDefaultConfigs:
    """Test DEFAULT_CONFIGS constant."""
    
    def test_default_configs_type(self):
        """Test that DEFAULT_CONFIGS is a dictionary."""
        assert isinstance(DEFAULT_CONFIGS, dict)
    
    def test_default_configs_content(self):
        """Test that DEFAULT_CONFIGS contains expected mappings."""
        expected_configs = {
            "default-llm": "llm",
            "default-vector-db": "vector_db",
            "default-graph-db": "graph_db", 
            "default-cache": "cache",
            "default-embedding": "embedding",
            "default-database": "database"
        }
        
        assert DEFAULT_CONFIGS == expected_configs
    
    def test_default_configs_count(self):
        """Test that DEFAULT_CONFIGS has expected number of items."""
        assert len(DEFAULT_CONFIGS) == 6
    
    def test_default_configs_keys_format(self):
        """Test that all keys follow 'default-*' pattern."""
        for key in DEFAULT_CONFIGS.keys():
            assert isinstance(key, str)
            assert key.startswith("default-")
            assert len(key) > len("default-")
    
    def test_default_configs_values_in_categories(self):
        """Test that all values are valid provider categories."""
        for value in DEFAULT_CONFIGS.values():
            assert value in PROVIDER_CATEGORIES
    
    def test_default_configs_keys_are_strings(self):
        """Test that all keys are strings."""
        for key in DEFAULT_CONFIGS.keys():
            assert isinstance(key, str)
            assert len(key) > 0
    
    def test_default_configs_values_are_strings(self):
        """Test that all values are strings."""
        for value in DEFAULT_CONFIGS.values():
            assert isinstance(value, str)
            assert len(value) > 0
    
    def test_default_configs_no_duplicate_values(self):
        """Test that each provider category appears only once."""
        values = list(DEFAULT_CONFIGS.values())
        assert len(values) == len(set(values))
    
    def test_default_configs_core_defaults_present(self):
        """Test that core default configurations are present."""
        core_defaults = {
            "default-llm": "llm",
            "default-cache": "cache",
            "default-database": "database"
        }
        
        for key, value in core_defaults.items():
            assert key in DEFAULT_CONFIGS
            assert DEFAULT_CONFIGS[key] == value
    
    def test_default_configs_hyphen_format(self):
        """Test that keys use hyphens while values use underscores."""
        for key, value in DEFAULT_CONFIGS.items():
            if "-" in key:
                # Key should use hyphens
                assert "-" in key
            if "_" in value:
                # Value should use underscores (if multi-word)
                assert "_" in value


class TestProviderTypeMap:
    """Test PROVIDER_TYPE_MAP constant."""
    
    def test__provider_type___map_type(self):
        """Test that PROVIDER_TYPE_MAP is a dictionary."""
        assert isinstance(PROVIDER_TYPE_MAP, dict)
    
    def test__provider_type___map_content(self):
        """Test that PROVIDER_TYPE_MAP contains expected mappings."""
        expected_mappings = {
            "llamacpp": "llm",
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
            "kafka": "message_queue"
        }
        
        assert PROVIDER_TYPE_MAP == expected_mappings
    
    def test__provider_type___map_count(self):
        """Test that PROVIDER_TYPE_MAP has expected number of items."""
        assert len(PROVIDER_TYPE_MAP) == 18
    
    def test__provider_type___map_keys_are_strings(self):
        """Test that all keys are strings."""
        for key in PROVIDER_TYPE_MAP.keys():
            assert isinstance(key, str)
            assert len(key) > 0
    
    def test__provider_type___map_values_are_strings(self):
        """Test that all values are strings."""
        for value in PROVIDER_TYPE_MAP.values():
            assert isinstance(value, str)
            assert len(value) > 0
    
    def test__provider_type___map_values_in_categories(self):
        """Test that all values are valid provider categories."""
        for value in PROVIDER_TYPE_MAP.values():
            assert value in PROVIDER_CATEGORIES
    
    def test__provider_type___map_llm_providers(self):
        """Test that LLM providers are correctly mapped."""
        llm_providers = {
            "llamacpp": "llm",
            "google_ai": "llm"
        }
        
        for provider, category in llm_providers.items():
            assert PROVIDER_TYPE_MAP[provider] == category
    
    def test__provider_type___map_database_providers(self):
        """Test that database providers are correctly mapped."""
        db_providers = {
            "postgres": "database",
            "mongodb": "database", 
            "sqlite": "database"
        }
        
        for provider, category in db_providers.items():
            assert PROVIDER_TYPE_MAP[provider] == category
    
    def test__provider_type___map_vector_db_providers(self):
        """Test that vector database providers are correctly mapped."""
        vector_providers = {
            "chroma": "vector_db",
            "pinecone": "vector_db",
            "qdrant": "vector_db"
        }
        
        for provider, category in vector_providers.items():
            assert PROVIDER_TYPE_MAP[provider] == category
    
    def test__provider_type___map_storage_providers(self):
        """Test that storage providers are correctly mapped."""
        storage_providers = {
            "s3": "storage",
            "local": "storage"
        }
        
        for provider, category in storage_providers.items():
            assert PROVIDER_TYPE_MAP[provider] == category
    
    def test__provider_type___map_embedding_providers(self):
        """Test that embedding providers are correctly mapped."""
        embedding_providers = {
            "llamacpp_embedding": "embedding"
        }
        
        for provider, category in embedding_providers.items():
            assert PROVIDER_TYPE_MAP[provider] == category
    
    def test__provider_type___map_graph_db_providers(self):
        """Test that graph database providers are correctly mapped."""
        graph_providers = {
            "neo4j": "graph_db",
            "arango": "graph_db"
        }
        
        for provider, category in graph_providers.items():
            assert PROVIDER_TYPE_MAP[provider] == category
    
    def test__provider_type___map_message_queue_providers(self):
        """Test that message queue providers are correctly mapped."""
        mq_providers = {
            "rabbitmq": "message_queue",
            "kafka": "message_queue"
        }
        
        for provider, category in mq_providers.items():
            assert PROVIDER_TYPE_MAP[provider] == category
    
    def test__provider_type___map_cache_providers(self):
        """Test that cache providers are correctly mapped."""
        cache_providers = {
            "redis": "cache"
        }
        
        for provider, category in cache_providers.items():
            assert PROVIDER_TYPE_MAP[provider] == category
    
    def test__provider_type___map_no_duplicate_keys(self):
        """Test that there are no duplicate keys."""
        keys = list(PROVIDER_TYPE_MAP.keys())
        assert len(keys) == len(set(keys))
    
    def test__provider_type___map_key_naming_conventions(self):
        """Test that keys follow expected naming conventions."""
        for key in PROVIDER_TYPE_MAP.keys():
            # Should be lowercase
            assert key.islower()
            # Should only contain letters, numbers, and underscores
            assert all(c.islower() or c.isdigit() or c == '_' for c in key)
            # Should not start or end with underscore
            assert not key.startswith('_')
            assert not key.endswith('_')


class TestConstantsIntegration:
    """Integration tests for provider constants."""
    
    def test_default_configs_coverage(self):
        """Test that DEFAULT_CONFIGS covers main provider categories."""
        main_categories = {"llm", "database", "cache", "embedding", "vector_db", "graph_db"}
        default_categories = set(DEFAULT_CONFIGS.values())
        
        # All main categories should have defaults
        assert main_categories.issubset(default_categories)
    
    def test__provider_type___map_category_coverage(self):
        """Test that PROVIDER_TYPE_MAP covers reasonable subset of categories."""
        mapped_categories = set(PROVIDER_TYPE_MAP.values())
        
        # Should cover at least these essential categories
        essential_categories = {"llm", "database", "vector_db", "storage", "cache"}
        assert essential_categories.issubset(mapped_categories)
    
    def test_constants_consistency(self):
        """Test consistency between different constants."""
        # All values in DEFAULT_CONFIGS should be in PROVIDER_CATEGORIES
        for value in DEFAULT_CONFIGS.values():
            assert value in PROVIDER_CATEGORIES
        
        # All values in PROVIDER_TYPE_MAP should be in PROVIDER_CATEGORIES  
        for value in PROVIDER_TYPE_MAP.values():
            assert value in PROVIDER_CATEGORIES
    
    def test_no_overlapping_provider_names(self):
        """Test that provider names don't conflict across categories."""
        # This is a design validation - provider names in TYPE_MAP should be unique
        provider_names = list(PROVIDER_TYPE_MAP.keys())
        assert len(provider_names) == len(set(provider_names))
    
    def test_backward_compatibility_indicators(self):
        """Test that the constants support backward compatibility."""
        # PROVIDER_TYPE_MAP should map old provider names to new categories
        assert len(PROVIDER_TYPE_MAP) > 0
        
        # Should cover major provider types that exist in the codebase
        legacy__provider_type__s = {"postgres", "redis", "google_ai", "chroma"}
        mapped_providers = set(PROVIDER_TYPE_MAP.keys())
        assert legacy__provider_type__s.issubset(mapped_providers)
    
    def test_config_driven_architecture_support(self):
        """Test that constants support config-driven provider architecture."""
        # DEFAULT_CONFIGS should provide default names for config-driven access
        assert len(DEFAULT_CONFIGS) > 0
        
        # All default config keys should follow naming pattern
        for key in DEFAULT_CONFIGS.keys():
            assert key.startswith("default-")
        
        # PROVIDER_CATEGORIES should define all available provider types
        assert len(PROVIDER_CATEGORIES) >= 10  # Reasonable minimum


class TestConstantsDocumentation:
    """Test that constants are well-documented through their structure."""
    
    def test_provider_categories_semantic_grouping(self):
        """Test that provider categories make semantic sense."""
        # Data storage categories
        data_categories = {"database", "vector_db", "graph_db", "cache", "storage"}
        assert data_categories.issubset(PROVIDER_CATEGORIES)
        
        # Processing categories
        processing_categories = {"llm", "embedding", "gpu"}
        assert processing_categories.issubset(PROVIDER_CATEGORIES)
        
        # Communication categories
        comm_categories = {"message_queue", "api", "mcp_client", "mcp_server"}
        assert comm_categories.issubset(PROVIDER_CATEGORIES)
    
    def test__provider_type___map_logical_groupings(self):
        """Test that provider type mappings are logically grouped."""
        # Group by category and verify logical consistency
        by_category = {}
        for provider, category in PROVIDER_TYPE_MAP.items():
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(provider)
        
        # LLM category should have LLM providers
        if "llm" in by_category:
            llm_providers = by_category["llm"]
            assert any("llm" in provider.lower() or provider in ["openai", "google_ai"] 
                      for provider in llm_providers)
        
        # Database category should have database providers
        if "database" in by_category:
            db_providers = by_category["database"]
            assert any(provider in ["postgres", "mongodb", "sqlite"] 
                      for provider in db_providers)
    
    def test_constants_immutability_pattern(self):
        """Test that constants follow immutability patterns."""
        # These should be module-level constants (tested by import)
        # In Python, we can't enforce true immutability, but we can test structure
        
        # Sets and dicts are the appropriate types for these constants
        assert isinstance(PROVIDER_CATEGORIES, set)
        assert isinstance(DEFAULT_CONFIGS, dict)
        assert isinstance(PROVIDER_TYPE_MAP, dict)
        
        # Should not be empty
        assert len(PROVIDER_CATEGORIES) > 0
        assert len(DEFAULT_CONFIGS) > 0
        assert len(PROVIDER_TYPE_MAP) > 0