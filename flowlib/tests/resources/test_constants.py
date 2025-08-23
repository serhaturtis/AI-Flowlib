"""Tests for resource constants."""
import pytest
from flowlib.resources.models.constants import ResourceType


class TestResourceType:
    """Test ResourceType enumeration."""
    
    def test_resource_type_values(self):
        """Test that all resource types have expected values."""
        assert ResourceType.MODEL_CONFIG == "model_config"
        assert ResourceType.PROMPT_CONFIG == "prompt_config"
        assert ResourceType.LLM_CONFIG == "llm_config"
        assert ResourceType.DATABASE_CONFIG == "database_config"
        assert ResourceType.VECTOR_DB_CONFIG == "vector_db_config"
        assert ResourceType.CACHE_CONFIG == "cache_config"
        assert ResourceType.STORAGE_CONFIG == "storage_config"
        assert ResourceType.EMBEDDING_CONFIG == "embedding_config"
        assert ResourceType.GRAPH_DB_CONFIG == "graph_db_config"
        assert ResourceType.MESSAGE_QUEUE_CONFIG == "message_queue_config"
    
    def test_resource_type_is_string_enum(self):
        """Test that ResourceType is a string enum."""
        assert isinstance(ResourceType.MODEL_CONFIG, str)
        assert str(ResourceType.MODEL_CONFIG) == "model_config"
    
    def test_resource_type_enumeration(self):
        """Test ResourceType enumeration behavior."""
        # Test that we can iterate over all values
        all_types = list(ResourceType)
        assert len(all_types) >= 10  # Allow for additional test types
        
        # Test that core values are present
        core_types = {
            "model_config",
            "prompt_config", 
            "llm_config",
            "database_config",
            "vector_db_config",
            "cache_config",
            "storage_config",
            "embedding_config",
            "graph_db_config",
            "message_queue_config"
        }
        
        actual_types = {rt.value for rt in all_types}
        assert core_types.issubset(actual_types)  # Core types must be present
    
    def test_resource_type_comparison(self):
        """Test ResourceType comparison operations."""
        # Test equality with string
        assert ResourceType.MODEL_CONFIG == "model_config"
        assert ResourceType.LLM_CONFIG == "llm_config"
        
        # Test inequality
        assert ResourceType.MODEL_CONFIG != "llm_config"
        assert ResourceType.LLM_CONFIG != ResourceType.MODEL_CONFIG
    
    def test_resource_type_membership(self):
        """Test ResourceType membership testing."""
        assert "model_config" in [member.value for member in ResourceType]
        assert "prompt_config" in [member.value for member in ResourceType]
        assert "invalid_type" not in [member.value for member in ResourceType]
    
    def test_resource_type_immutability(self):
        """Test that ResourceType values are immutable."""
        with pytest.raises(AttributeError):
            ResourceType.MODEL_CONFIG = "modified_value"
    
    def test_resource_type_serialization(self):
        """Test ResourceType serialization behavior."""
        # Test JSON serialization compatibility
        import json
        
        config_type = ResourceType.MODEL_CONFIG
        serialized = json.dumps(config_type)
        assert serialized == '"model_config"'
        
        deserialized = json.loads(serialized)
        assert deserialized == "model_config"
        assert deserialized == config_type
    
    def test_resource_type_hashing(self):
        """Test ResourceType hashing for use in sets and dicts."""
        # Test that enum values can be used as dict keys
        type_dict = {
            ResourceType.MODEL_CONFIG: "model",
            ResourceType.LLM_CONFIG: "llm",
            ResourceType.DATABASE_CONFIG: "db"
        }
        
        assert type_dict[ResourceType.MODEL_CONFIG] == "model"
        assert type_dict[ResourceType.LLM_CONFIG] == "llm"
        
        # Test that enum values can be used in sets
        type_set = {ResourceType.MODEL_CONFIG, ResourceType.LLM_CONFIG}
        assert ResourceType.MODEL_CONFIG in type_set
        assert ResourceType.DATABASE_CONFIG not in type_set
    
    def test_resource_type_string_operations(self):
        """Test string operations on ResourceType values."""
        config_type = ResourceType.MODEL_CONFIG
        
        # Test string formatting
        assert f"Type: {config_type}" == "Type: model_config"
        
        # Test string methods
        assert config_type.startswith("model")
        assert config_type.endswith("config")
        assert "model" in config_type
        
        # Test case operations
        assert config_type.upper() == "MODEL_CONFIG"
        assert config_type.replace("_", "-") == "model-config"