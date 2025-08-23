"""Comprehensive tests for agent persistence adapters module."""

import pytest
import logging
from typing import Optional, Dict, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pydantic import ValidationError

from flowlib.agent.components.persistence.adapters import (
    RedisStatePersister,
    RedisStatePersisterSettings,
    MongoStatePersister,
    MongoStatePersisterSettings,
    PostgresStatePersister,
    PostgresStatePersisterSettings
)
from flowlib.agent.components.persistence.base import BaseStatePersister
from flowlib.agent.models.state import AgentState
from flowlib.providers.cache.base import CacheProvider
from flowlib.providers.db.base import DBProvider
from flowlib.core.errors.errors import ProviderError


class TestRedisStatePersisterSettings:
    """Test RedisStatePersisterSettings model."""
    
    def test_redis_settings_creation_minimal(self):
        """Test creating settings with minimal required fields."""
        settings = RedisStatePersisterSettings(
            redis_provider_name="test_redis"
        )
        assert settings.redis_provider_name == "test_redis"
        assert settings.key_prefix == "agent_state:"
    
    def test_redis_settings_creation_full(self):
        """Test creating settings with all fields."""
        settings = RedisStatePersisterSettings(
            redis_provider_name="custom_redis",
            key_prefix="custom_prefix:"
        )
        assert settings.redis_provider_name == "custom_redis"
        assert settings.key_prefix == "custom_prefix:"
    
    def test_redis_settings_validation_provider_name_required(self):
        """Test that redis_provider_name is required."""
        with pytest.raises(ValidationError) as exc_info:
            RedisStatePersisterSettings()
        
        assert "redis_provider_name" in str(exc_info.value)
    
    def test_redis_settings_validation_provider_name_non_empty(self):
        """Test that redis_provider_name cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            RedisStatePersisterSettings(redis_provider_name="")
        
        assert "String should have at least 1 character" in str(exc_info.value)


class TestRedisStatePersister:
    """Test RedisStatePersister class."""
    
    def test_redis_persister_inheritance(self):
        """Test that RedisStatePersister inherits from BaseStatePersister."""
        settings = RedisStatePersisterSettings(redis_provider_name="test_redis")
        persister = RedisStatePersister(name="test_persister", settings=settings)
        assert isinstance(persister, BaseStatePersister)
    
    def test_redis_persister_creation(self):
        """Test creating RedisStatePersister instance."""
        settings = RedisStatePersisterSettings(redis_provider_name="test_redis")
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        assert persister.name == "test_persister"
        assert persister.settings == settings
        assert persister._redis_provider is None
    
    @pytest.mark.asyncio
    async def test_redis_persister_initialize_success(self):
        """Test successful initialization."""
        settings = RedisStatePersisterSettings(redis_provider_name="test_redis")
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        # Mock the provider registry and cache provider
        mock_cache_provider = Mock(spec=CacheProvider)
        mock_cache_provider.initialized = True
        
        with patch('flowlib.agent.components.persistence.adapters.provider_registry') as mock_registry:
            mock_registry.get_by_config = AsyncMock(return_value=mock_cache_provider)
            
            await persister._initialize()
            
            assert persister._redis_provider == mock_cache_provider
            mock_registry.get_by_config.assert_called_once_with("default-cache")
    
    @pytest.mark.asyncio
    async def test_redis_persister_initialize_provider_not_initialized(self):
        """Test initialization when provider needs initialization."""
        settings = RedisStatePersisterSettings(redis_provider_name="test_redis")
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        mock_cache_provider = Mock(spec=CacheProvider)
        mock_cache_provider.initialized = False
        mock_cache_provider.initialize = AsyncMock()
        
        with patch('flowlib.agent.components.persistence.adapters.provider_registry') as mock_registry:
            mock_registry.get_by_config = AsyncMock(return_value=mock_cache_provider)
            
            await persister._initialize()
            
            assert persister._redis_provider == mock_cache_provider
            mock_cache_provider.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_redis_persister_initialize_wrong_provider_type(self):
        """Test initialization with wrong provider type."""
        settings = RedisStatePersisterSettings(redis_provider_name="test_redis")
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        mock_wrong_provider = Mock()  # Not a CacheProvider
        
        with patch('flowlib.agent.components.persistence.adapters.provider_registry') as mock_registry:
            mock_registry.get_by_config = AsyncMock(return_value=mock_wrong_provider)
            
            with pytest.raises(TypeError) as exc_info:
                await persister._initialize()
            
            assert "not a valid CacheProvider" in str(exc_info.value)
            assert persister._redis_provider is None
    
    @pytest.mark.asyncio
    async def test_redis_persister_initialize_provider_error(self):
        """Test initialization with provider error."""
        settings = RedisStatePersisterSettings(redis_provider_name="test_redis")
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        with patch('flowlib.agent.components.persistence.adapters.provider_registry') as mock_registry:
            mock_registry.get_by_config = AsyncMock(side_effect=Exception("Provider error"))
            
            with pytest.raises(Exception) as exc_info:
                await persister._initialize()
            
            assert "Provider error" in str(exc_info.value)
            assert persister._redis_provider is None
    
    @pytest.mark.asyncio
    async def test_redis_persister_save_state_success(self):
        """Test successful state saving."""
        settings = RedisStatePersisterSettings(
            redis_provider_name="test_redis",
            key_prefix="test:"
        )
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        # Mock initialized state and provider
        mock_cache_provider = Mock(spec=CacheProvider)
        mock_cache_provider.set = AsyncMock()
        persister._redis_provider = mock_cache_provider
        
        # Create test state
        test_state = Mock(spec=AgentState)
        test_state.task_id = "test_task_123"
        test_state.model_dump_json.return_value = '{"test": "data"}'
        test_state.config = None
        
        persister._initialized = True
        result = await persister._save_state_impl(test_state)
        
        assert result is True
        mock_cache_provider.set.assert_called_once_with(
            "test:test_task_123",
            '{"test": "data"}',
            ttl_seconds=None
        )
    
    @pytest.mark.asyncio
    async def test_redis_persister_save_state_with_ttl(self):
        """Test state saving with TTL from config."""
        settings = RedisStatePersisterSettings(redis_provider_name="test_redis")
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        # Mock provider and state
        mock_cache_provider = Mock(spec=CacheProvider)
        mock_cache_provider.set = AsyncMock()
        persister._redis_provider = mock_cache_provider
        
        # Create test state with TTL config
        test_state = Mock(spec=AgentState)
        test_state.task_id = "test_task_123"
        test_state.model_dump_json.return_value = '{"test": "data"}'
        test_state.config = Mock()
        test_state.config.state_config = Mock()
        test_state.config.state_config.ttl_seconds = 3600
        
        persister._initialized = True
        result = await persister._save_state_impl(test_state)
        
        assert result is True
        mock_cache_provider.set.assert_called_once_with(
            "agent_state:test_task_123",
            '{"test": "data"}',
            ttl_seconds=3600
        )
    
    @pytest.mark.asyncio
    async def test_redis_persister_save_state_not_initialized(self):
        """Test saving state when not initialized."""
        settings = RedisStatePersisterSettings(redis_provider_name="test_redis")
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        test_state = Mock(spec=AgentState)
        test_state.task_id = "test_task_123"
        
        with pytest.raises(RuntimeError) as exc_info:
            await persister._save_state_impl(test_state)
        
        assert "not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_redis_persister_save_state_redis_error(self):
        """Test saving state with Redis error."""
        settings = RedisStatePersisterSettings(redis_provider_name="test_redis")
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        # Mock provider and state
        mock_cache_provider = Mock(spec=CacheProvider)
        mock_cache_provider.set = AsyncMock(side_effect=Exception("Redis connection error"))
        persister._redis_provider = mock_cache_provider
        
        test_state = Mock(spec=AgentState)
        test_state.task_id = "test_task_123"
        test_state.model_dump_json.return_value = '{"test": "data"}'
        test_state.config = None
        
        persister._initialized = True
        result = await persister._save_state_impl(test_state)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_redis_persister_load_state_success(self):
        """Test successful state loading."""
        settings = RedisStatePersisterSettings(
            redis_provider_name="test_redis",
            key_prefix="test:"
        )
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        # Mock provider and state
        mock_cache_provider = Mock(spec=CacheProvider)
        mock_cache_provider.get = AsyncMock(return_value='{"task_id": "test_task_123"}')
        persister._redis_provider = mock_cache_provider
        
        persister._initialized = True
        with patch('flowlib.agent.components.persistence.adapters.AgentState') as mock_agent_state:
            mock_state = Mock()
            mock_agent_state.model_validate_json.return_value = mock_state
            
            result = await persister._load_state_impl("test_task_123")
            
            assert result == mock_state
            mock_cache_provider.get.assert_called_once_with("test:test_task_123")
            mock_agent_state.model_validate_json.assert_called_once_with('{"task_id": "test_task_123"}')
    
    @pytest.mark.asyncio
    async def test_redis_persister_load_state_not_found(self):
        """Test loading state when not found."""
        settings = RedisStatePersisterSettings(redis_provider_name="test_redis")
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        # Mock provider
        mock_cache_provider = Mock(spec=CacheProvider)
        mock_cache_provider.get = AsyncMock(return_value=None)
        persister._redis_provider = mock_cache_provider
        
        persister._initialized = True
        result = await persister._load_state_impl("nonexistent_task")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_redis_persister_load_state_not_initialized(self):
        """Test loading state when not initialized."""
        settings = RedisStatePersisterSettings(redis_provider_name="test_redis")
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        with pytest.raises(RuntimeError) as exc_info:
            await persister._load_state_impl("test_task_123")
        
        assert "not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_redis_persister_load_state_redis_error(self):
        """Test loading state with Redis error."""
        settings = RedisStatePersisterSettings(redis_provider_name="test_redis")
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        # Mock provider
        mock_cache_provider = Mock(spec=CacheProvider)
        mock_cache_provider.get = AsyncMock(side_effect=Exception("Redis connection error"))
        persister._redis_provider = mock_cache_provider
        
        persister._initialized = True
        with pytest.raises(Exception) as exc_info:
            await persister._load_state_impl("test_task_123")
        
        assert "Redis connection error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_redis_persister_delete_state_success(self):
        """Test successful state deletion."""
        settings = RedisStatePersisterSettings(
            redis_provider_name="test_redis",
            key_prefix="test:"
        )
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        # Mock provider
        mock_cache_provider = Mock(spec=CacheProvider)
        mock_cache_provider.delete = AsyncMock(return_value=1)
        persister._redis_provider = mock_cache_provider
        
        persister._initialized = True
        result = await persister._delete_state_impl("test_task_123")
        
        assert result is True
        mock_cache_provider.delete.assert_called_once_with("test:test_task_123")
    
    @pytest.mark.asyncio
    async def test_redis_persister_delete_state_not_found(self):
        """Test deleting state when not found."""
        settings = RedisStatePersisterSettings(redis_provider_name="test_redis")
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        # Mock provider
        mock_cache_provider = Mock(spec=CacheProvider)
        mock_cache_provider.delete = AsyncMock(return_value=0)
        persister._redis_provider = mock_cache_provider
        
        persister._initialized = True
        result = await persister._delete_state_impl("nonexistent_task")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_redis_persister_delete_state_error(self):
        """Test deleting state with error."""
        settings = RedisStatePersisterSettings(redis_provider_name="test_redis")
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        # Mock provider
        mock_cache_provider = Mock(spec=CacheProvider)
        mock_cache_provider.delete = AsyncMock(side_effect=Exception("Delete error"))
        persister._redis_provider = mock_cache_provider
        
        persister._initialized = True
        result = await persister._delete_state_impl("test_task_123")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_redis_persister_list_states_not_implemented(self):
        """Test that list_states raises NotImplementedError."""
        settings = RedisStatePersisterSettings(redis_provider_name="test_redis")
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        with pytest.raises(NotImplementedError):
            await persister._list_states_impl()
    
    @pytest.mark.asyncio
    async def test_redis_persister_shutdown(self):
        """Test shutdown functionality."""
        settings = RedisStatePersisterSettings(redis_provider_name="test_redis")
        persister = RedisStatePersister(name="test_persister", settings=settings)
        
        mock_cache_provider = Mock(spec=CacheProvider)
        persister._redis_provider = mock_cache_provider
        
        await persister._shutdown()
        
        assert persister._redis_provider is None


class TestMongoStatePersisterSettings:
    """Test MongoStatePersisterSettings model."""
    
    def test_mongo_settings_creation_minimal(self):
        """Test creating settings with minimal required fields."""
        settings = MongoStatePersisterSettings(
            mongo_provider_name="test_mongo",
            database_name="test_db"
        )
        assert settings.mongo_provider_name == "test_mongo"
        assert settings.database_name == "test_db"
        assert settings.collection_name == "agent_states"
    
    def test_mongo_settings_creation_full(self):
        """Test creating settings with all fields."""
        settings = MongoStatePersisterSettings(
            mongo_provider_name="custom_mongo",
            database_name="custom_db",
            collection_name="custom_collection"
        )
        assert settings.mongo_provider_name == "custom_mongo"
        assert settings.database_name == "custom_db"
        assert settings.collection_name == "custom_collection"
    
    def test_mongo_settings_validation_required_fields(self):
        """Test that required fields are validated."""
        with pytest.raises(ValidationError) as exc_info:
            MongoStatePersisterSettings()
        
        errors = str(exc_info.value)
        assert "mongo_provider_name" in errors
        assert "database_name" in errors


class TestMongoStatePersister:
    """Test MongoStatePersister class."""
    
    def test_mongo_persister_creation(self):
        """Test creating MongoStatePersister instance."""
        settings = MongoStatePersisterSettings(
            mongo_provider_name="test_mongo",
            database_name="test_db"
        )
        persister = MongoStatePersister(name="test_persister", settings=settings)
        
        assert persister.name == "test_persister"
        assert persister.settings == settings
        assert persister._mongo_provider is None
        assert isinstance(persister, BaseStatePersister)
    
    @pytest.mark.asyncio
    async def test_mongo_persister_initialize_success(self):
        """Test successful initialization."""
        settings = MongoStatePersisterSettings(
            mongo_provider_name="test_mongo",
            database_name="test_db"
        )
        persister = MongoStatePersister(name="test_persister", settings=settings)
        
        mock_db_provider = Mock(spec=DBProvider)
        mock_db_provider.initialized = True
        
        with patch('flowlib.agent.components.persistence.adapters.provider_registry') as mock_registry:
            mock_registry.get_by_config = AsyncMock(return_value=mock_db_provider)
            
            await persister._initialize()
            
            assert persister._mongo_provider == mock_db_provider
    
    @pytest.mark.asyncio
    async def test_mongo_persister_save_state_success(self):
        """Test successful state saving."""
        settings = MongoStatePersisterSettings(
            mongo_provider_name="test_mongo",
            database_name="test_db",
            collection_name="test_collection"
        )
        persister = MongoStatePersister(name="test_persister", settings=settings)
        
        # Mock provider
        mock_db_provider = Mock(spec=DBProvider)
        mock_db_provider.upsert_document = AsyncMock(return_value="upserted_id")
        persister._mongo_provider = mock_db_provider
        
        test_state = Mock(spec=AgentState)
        test_state.task_id = "test_task_123"
        test_state.model_dump.return_value = {"task_id": "test_task_123", "data": "test"}
        
        persister._initialized = True
        result = await persister._save_state_impl(test_state, metadata={"type": "test"})
        
        assert result is True
        mock_db_provider.upsert_document.assert_called_once()
        call_args = mock_db_provider.upsert_document.call_args
        assert call_args.kwargs["database_name"] == "test_db"
        assert call_args.kwargs["collection_name"] == "test_collection"
        assert call_args.kwargs["document"]["_id"] == "test_task_123"
        assert call_args.kwargs["document"]["metadata"] == {"type": "test"}
    
    @pytest.mark.asyncio
    async def test_mongo_persister_load_state_success(self):
        """Test successful state loading."""
        settings = MongoStatePersisterSettings(
            mongo_provider_name="test_mongo",
            database_name="test_db"
        )
        persister = MongoStatePersister(name="test_persister", settings=settings)
        
        # Mock provider
        mock_db_provider = Mock(spec=DBProvider)
        mock_db_provider.find_document_by_id = AsyncMock(
            return_value={"_id": "test_task_123", "task_id": "test_task_123"}
        )
        persister._mongo_provider = mock_db_provider
        
        persister._initialized = True
        with patch('flowlib.agent.components.persistence.adapters.AgentState') as mock_agent_state:
            mock_state = Mock()
            mock_agent_state.model_validate.return_value = mock_state
            
            result = await persister._load_state_impl("test_task_123")
            
            assert result == mock_state
            mock_agent_state.model_validate.assert_called_once_with(
                {"_id": "test_task_123", "task_id": "test_task_123"}
            )
    
    @pytest.mark.asyncio
    async def test_mongo_persister_delete_state_success(self):
        """Test successful state deletion."""
        settings = MongoStatePersisterSettings(
            mongo_provider_name="test_mongo",
            database_name="test_db"
        )
        persister = MongoStatePersister(name="test_persister", settings=settings)
        
        # Mock provider
        mock_db_provider = Mock(spec=DBProvider)
        mock_db_provider.delete_document_by_id = AsyncMock(return_value=1)
        persister._mongo_provider = mock_db_provider
        
        persister._initialized = True
        result = await persister._delete_state_impl("test_task_123")
        
        assert result is True


class TestPostgresStatePersisterSettings:
    """Test PostgresStatePersisterSettings model."""
    
    def test_postgres_settings_creation_minimal(self):
        """Test creating settings with minimal required fields."""
        settings = PostgresStatePersisterSettings(
            postgres_provider_name="test_postgres"
        )
        assert settings.postgres_provider_name == "test_postgres"
        assert settings.table_name == "agent_states"
        assert settings.id_column == "task_id"
        assert settings.data_column == "state_data"
    
    def test_postgres_settings_creation_full(self):
        """Test creating settings with all fields."""
        settings = PostgresStatePersisterSettings(
            postgres_provider_name="custom_postgres",
            table_name="custom_table",
            id_column="custom_id",
            data_column="custom_data"
        )
        assert settings.postgres_provider_name == "custom_postgres"
        assert settings.table_name == "custom_table"
        assert settings.id_column == "custom_id"
        assert settings.data_column == "custom_data"


class TestPostgresStatePersister:
    """Test PostgresStatePersister class."""
    
    def test_postgres_persister_creation(self):
        """Test creating PostgresStatePersister instance."""
        settings = PostgresStatePersisterSettings(
            postgres_provider_name="test_postgres"
        )
        persister = PostgresStatePersister(name="test_persister", settings=settings)
        
        assert persister.name == "test_persister"
        assert persister.settings == settings
        assert persister._postgres_provider is None
        assert isinstance(persister, BaseStatePersister)
    
    @pytest.mark.asyncio
    async def test_postgres_persister_save_state_success(self):
        """Test successful state saving."""
        settings = PostgresStatePersisterSettings(
            postgres_provider_name="test_postgres",
            table_name="test_table",
            id_column="test_id",
            data_column="test_data"
        )
        persister = PostgresStatePersister(name="test_persister", settings=settings)
        
        # Mock provider
        mock_db_provider = Mock(spec=DBProvider)
        mock_db_provider.upsert_record = AsyncMock(return_value="success")
        persister._postgres_provider = mock_db_provider
        
        test_state = Mock(spec=AgentState)
        test_state.task_id = "test_task_123"
        test_state.model_dump_json.return_value = '{"task_id": "test_task_123"}'
        
        persister._initialized = True
        result = await persister._save_state_impl(test_state)
        
        assert result is True
        mock_db_provider.upsert_record.assert_called_once()
        call_args = mock_db_provider.upsert_record.call_args
        assert call_args.kwargs["table_name"] == "test_table"
        assert call_args.kwargs["record_data"]["test_id"] == "test_task_123"
        assert call_args.kwargs["record_data"]["test_data"] == '{"task_id": "test_task_123"}'
    
    @pytest.mark.asyncio
    async def test_postgres_persister_load_state_json_string(self):
        """Test loading state from JSON string."""
        settings = PostgresStatePersisterSettings(
            postgres_provider_name="test_postgres"
        )
        persister = PostgresStatePersister(name="test_persister", settings=settings)
        
        # Mock provider
        mock_db_provider = Mock(spec=DBProvider)
        mock_db_provider.find_record_by_field = AsyncMock(
            return_value={"task_id": "test_task_123", "state_data": '{"task_id": "test_task_123"}'}
        )
        persister._postgres_provider = mock_db_provider
        
        persister._initialized = True
        with patch('flowlib.agent.components.persistence.adapters.AgentState') as mock_agent_state:
            mock_state = Mock()
            mock_agent_state.model_validate_json.return_value = mock_state
            
            result = await persister._load_state_impl("test_task_123")
            
            assert result == mock_state
            mock_agent_state.model_validate_json.assert_called_once_with('{"task_id": "test_task_123"}')
    
    @pytest.mark.asyncio
    async def test_postgres_persister_load_state_dict_data(self):
        """Test loading state from dict data."""
        settings = PostgresStatePersisterSettings(
            postgres_provider_name="test_postgres"
        )
        persister = PostgresStatePersister(name="test_persister", settings=settings)
        
        # Mock provider
        mock_db_provider = Mock(spec=DBProvider)
        mock_db_provider.find_record_by_field = AsyncMock(
            return_value={"task_id": "test_task_123", "state_data": {"task_id": "test_task_123"}}
        )
        persister._postgres_provider = mock_db_provider
        
        persister._initialized = True
        with patch('flowlib.agent.components.persistence.adapters.AgentState') as mock_agent_state:
            mock_state = Mock()
            mock_agent_state.model_validate.return_value = mock_state
            
            result = await persister._load_state_impl("test_task_123")
            
            assert result == mock_state
            mock_agent_state.model_validate.assert_called_once_with({"task_id": "test_task_123"})
    
    @pytest.mark.asyncio
    async def test_postgres_persister_load_state_invalid_data_type(self):
        """Test loading state with invalid data type."""
        settings = PostgresStatePersisterSettings(
            postgres_provider_name="test_postgres"
        )
        persister = PostgresStatePersister(name="test_persister", settings=settings)
        
        # Mock provider
        mock_db_provider = Mock(spec=DBProvider)
        mock_db_provider.find_record_by_field = AsyncMock(
            return_value={"task_id": "test_task_123", "state_data": 12345}  # Invalid type
        )
        persister._postgres_provider = mock_db_provider
        
        persister._initialized = True
        with pytest.raises(TypeError) as exc_info:
            await persister._load_state_impl("test_task_123")
        
        assert "Unexpected type" in str(exc_info.value)


class TestStatePersisterIntegration:
    """Test integration aspects of state persisters."""
    
    @pytest.mark.asyncio
    async def test_persister_decorator_registration(self):
        """Test that persisters are properly decorated for registration."""
        # Check that the classes have the expected provider decorator attributes
        assert hasattr(RedisStatePersister, '__provider_name__')
        assert hasattr(MongoStatePersister, '__provider_name__')
        assert hasattr(PostgresStatePersister, '__provider_name__')
        
        assert RedisStatePersister.__provider_name__ == "redis"
        assert MongoStatePersister.__provider_name__ == "mongodb"
        assert PostgresStatePersister.__provider_name__ == "postgres"
    
    @pytest.mark.asyncio
    async def test_complete_workflow_redis(self):
        """Test complete workflow with Redis persister."""
        settings = RedisStatePersisterSettings(
            redis_provider_name="test_redis",
            key_prefix="workflow:"
        )
        persister = RedisStatePersister(name="workflow_persister", settings=settings)
        
        # Mock provider and initialization
        mock_cache_provider = Mock(spec=CacheProvider)
        mock_cache_provider.initialized = True
        mock_cache_provider.set = AsyncMock()
        mock_cache_provider.get = AsyncMock(return_value='{"task_id": "workflow_test", "data": "test"}')
        mock_cache_provider.delete = AsyncMock(return_value=1)
        
        with patch('flowlib.agent.components.persistence.adapters.provider_registry') as mock_registry:
            mock_registry.get_by_config = AsyncMock(return_value=mock_cache_provider)
            
            # Initialize
            await persister._initialize()
            assert persister._redis_provider == mock_cache_provider
            persister._initialized = True  # Mock the initialized state
            
            # Create and save state
            test_state = Mock(spec=AgentState)
            test_state.task_id = "workflow_test"
            test_state.model_dump_json.return_value = '{"task_id": "workflow_test", "data": "test"}'
            test_state.config = None
            
            save_result = await persister._save_state_impl(test_state)
            assert save_result is True
            
            # Load state
            with patch('flowlib.agent.components.persistence.adapters.AgentState') as mock_agent_state:
                mock_loaded_state = Mock()
                mock_agent_state.model_validate_json.return_value = mock_loaded_state
                
                load_result = await persister._load_state_impl("workflow_test")
                assert load_result == mock_loaded_state
            
            # Delete state
            delete_result = await persister._delete_state_impl("workflow_test")
            assert delete_result is True
            
            # Shutdown
            await persister._shutdown()
            assert persister._redis_provider is None
    
    def test_settings_serialization(self):
        """Test that all settings can be serialized/deserialized."""
        # Test Redis settings
        redis_settings = RedisStatePersisterSettings(
            redis_provider_name="test_redis",
            key_prefix="test:"
        )
        redis_dict = redis_settings.model_dump()
        redis_restored = RedisStatePersisterSettings(**redis_dict)
        assert redis_restored.redis_provider_name == redis_settings.redis_provider_name
        assert redis_restored.key_prefix == redis_settings.key_prefix
        
        # Test Mongo settings
        mongo_settings = MongoStatePersisterSettings(
            mongo_provider_name="test_mongo",
            database_name="test_db",
            collection_name="test_collection"
        )
        mongo_dict = mongo_settings.model_dump()
        mongo_restored = MongoStatePersisterSettings(**mongo_dict)
        assert mongo_restored.mongo_provider_name == mongo_settings.mongo_provider_name
        assert mongo_restored.database_name == mongo_settings.database_name
        
        # Test Postgres settings
        postgres_settings = PostgresStatePersisterSettings(
            postgres_provider_name="test_postgres",
            table_name="test_table"
        )
        postgres_dict = postgres_settings.model_dump()
        postgres_restored = PostgresStatePersisterSettings(**postgres_dict)
        assert postgres_restored.postgres_provider_name == postgres_settings.postgres_provider_name
        assert postgres_restored.table_name == postgres_settings.table_name
    
    @pytest.mark.asyncio
    async def test_error_handling_consistency(self):
        """Test that all persisters handle errors consistently."""
        # Test initialization errors
        for persister_class, settings_class in [
            (RedisStatePersister, RedisStatePersisterSettings),
            (MongoStatePersister, MongoStatePersisterSettings),
            (PostgresStatePersister, PostgresStatePersisterSettings)
        ]:
            if persister_class == RedisStatePersister:
                settings = settings_class(redis_provider_name="test")
            elif persister_class == MongoStatePersister:
                settings = settings_class(mongo_provider_name="test", database_name="test")
            else:  # PostgresStatePersister
                settings = settings_class(postgres_provider_name="test")
            
            persister = persister_class(name="error_test", settings=settings)
            
            # Test not initialized error
            test_state = Mock(spec=AgentState)
            test_state.task_id = "test"
            
            with pytest.raises(RuntimeError):
                await persister._save_state_impl(test_state)
            
            with pytest.raises(RuntimeError):
                await persister._load_state_impl("test")
            
            with pytest.raises(RuntimeError):
                await persister._delete_state_impl("test")