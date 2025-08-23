"""Tests for database provider base classes."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Optional, Any, Dict, List
from pydantic import BaseModel

from flowlib.providers.db.base import DBProviderSettings, DBProvider
from flowlib.providers.core.base import Provider
from flowlib.core.errors.errors import ProviderError


class MockRecord(BaseModel):
    """Mock model for structured database operations."""
    id: int
    name: str
    email: str
    active: bool = True


class TestDBProviderSettings:
    """Test DBProviderSettings configuration class."""
    
    def test_default_settings_values(self):
        """Test default settings values."""
        settings = DBProviderSettings()
        
        assert settings.host == "localhost"
        assert settings.port is None
        assert settings.database == ""
        assert settings.username is None
        assert settings.password is None
        assert settings.pool_size == 5
        assert settings.min_size == 1
        assert settings.max_overflow == 10
        assert settings.timeout == 30.0
        assert settings.connect_timeout == 10.0
        assert settings.use_ssl is False
        assert settings.ssl_ca_cert is None
        assert settings.ssl_cert is None
        assert settings.ssl_key is None
        assert settings.query_timeout == 30.0
        assert settings.max_query_size == 1000
        assert settings.application_name == "flowlib"
        assert settings.auto_reconnect is True
        assert settings.retry_count == 3
        assert settings.retry_delay == 1.0
    
    def test_custom_values(self):
        """Test settings with custom values."""
        settings = DBProviderSettings(
            host="db.example.com",
            port=5432,
            database="testdb",
            username="testuser",
            password="testpass",
            pool_size=10,
            min_size=2,
            max_overflow=20,
            timeout=60.0,
            connect_timeout=15.0,
            use_ssl=True,
            ssl_ca_cert="/path/to/ca.crt",
            ssl_cert="/path/to/client.crt",
            ssl_key="/path/to/client.key",
            query_timeout=45.0,
            max_query_size=2000,
            application_name="test_app",
            auto_reconnect=False,
            retry_count=5,
            retry_delay=2.0
        )
        
        assert settings.host == "db.example.com"
        assert settings.port == 5432
        assert settings.database == "testdb"
        assert settings.username == "testuser"
        assert settings.password == "testpass"
        assert settings.pool_size == 10
        assert settings.min_size == 2
        assert settings.max_overflow == 20
        assert settings.timeout == 60.0
        assert settings.connect_timeout == 15.0
        assert settings.use_ssl is True
        assert settings.ssl_ca_cert == "/path/to/ca.crt"
        assert settings.ssl_cert == "/path/to/client.crt"
        assert settings.ssl_key == "/path/to/client.key"
        assert settings.query_timeout == 45.0
        assert settings.max_query_size == 2000
        assert settings.application_name == "test_app"
        assert settings.auto_reconnect is False
        assert settings.retry_count == 5
        assert settings.retry_delay == 2.0


class ConcreteDBProviderSettings(DBProviderSettings):
    """Settings for concrete test provider."""
    pass


class ConcreteDBProvider(Provider[ConcreteDBProviderSettings]):
    """Concrete implementation for testing."""
    
    def __init__(self, name: str = "test_db", settings: Optional[ConcreteDBProviderSettings] = None):
        # Always provide settings to bypass default discovery
        super().__init__(name=name, settings=settings or ConcreteDBProviderSettings(), provider_type="db")
        self._mock_data = []
        self._transaction = None
        self._initialized = False
        self._pool = None
        self._settings = settings or ConcreteDBProviderSettings()
    
    @property
    def initialized(self) -> bool:
        """Return whether provider has been initialized."""
        return self._initialized
    
    async def initialize(self):
        """Initialize the provider."""
        self._initialized = True
    
    async def shutdown(self):
        """Shutdown the provider."""
        self._initialized = False
        self._pool = None
    
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        # Simple mock implementation
        if query.upper().startswith("SELECT"):
            return self._mock_data
        elif query.upper().startswith("INSERT"):
            return {"id": len(self._mock_data) + 1}
        elif query.upper().startswith("UPDATE"):
            return {"affected_rows": 1}
        elif query.upper().startswith("DELETE"):
            return {"affected_rows": 1}
        else:
            return {}
    
    async def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> List[Any]:
        results = []
        for params in params_list:
            result = await self.execute(query, params)
            results.append(result)
        return results
    
    async def execute_structured(self, query: str, output_type: type, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Execute a query and parse results into structured types."""
        from typing import get_type_hints
        
        try:
            # Execute the query
            results = await self.execute(query, params)
            
            # If results is empty, return empty list
            if not results:
                return []
                
            # Parse results into output type
            parsed_results = []
            for row in results:
                # If result is already a dict, use as is
                if isinstance(row, dict):
                    data = row
                # If result is a tuple/list, convert to dict
                elif isinstance(row, (tuple, list)):
                    # This is implementation-specific and should be handled by subclasses
                    raise NotImplementedError("Tuple/list conversion must be implemented by subclasses")
                else:
                    # Try to convert to dict if it has a method for it
                    if hasattr(row, "_asdict"):
                        data = row._asdict()
                    elif hasattr(row, "__dict__"):
                        data = row.__dict__
                    else:
                        raise TypeError(f"Cannot convert result of type {type(row)} to dict")
                
                # Parse into output type
                parsed_results.append(output_type.model_validate(data))
                
            return parsed_results
            
        except Exception as e:
            # Import here to avoid circular imports
            from flowlib.core.errors.errors import ProviderError, ErrorContext
            from flowlib.core.errors.models import ProviderErrorContext
            
            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to execute structured query: {str(e)}",
                context=ErrorContext.create(
                    flow_name="db_provider",
                    error_type="StructuredQueryError",
                    error_location="execute_structured",
                    component=self.name,
                    operation=f"query_execution"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="execute_structured",
                    retry_count=0
                ),
                cause=e
            )
    
    async def begin_transaction(self):
        self._transaction = Mock()
        return self._transaction
    
    async def commit_transaction(self, transaction: Any) -> bool:
        return True
    
    async def rollback_transaction(self, transaction: Any) -> bool:
        return True
    
    async def check_connection(self) -> bool:
        return self._initialized
    
    async def get_health(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized else "unhealthy",
            "connections": 5,
            "pool_size": self._settings.pool_size
        }
    
    def set_mock_data(self, data: List[Dict[str, Any]]):
        """Helper method to set mock data for testing."""
        self._mock_data = data


class TestDBProvider:
    """Test DBProvider base class."""
    
    def test_initialization_default(self):
        """Test provider initialization with defaults."""
        provider = ConcreteDBProvider()
        
        assert provider.name == "test_db"
        assert provider.provider_type == "db"
        assert provider.initialized is False
        assert provider._pool is None
        assert isinstance(provider._settings, ConcreteDBProviderSettings)
    
    def test_initialization_custom(self):
        """Test provider initialization with custom settings."""
        settings = ConcreteDBProviderSettings(
            host="custom.db.com",
            port=3306,
            database="custom_db",
            pool_size=15
        )
        provider = ConcreteDBProvider("custom_db", settings)
        
        assert provider.name == "custom_db"
        assert provider._settings.host == "custom.db.com"
        assert provider._settings.port == 3306
        assert provider._settings.database == "custom_db"
        assert provider._settings.pool_size == 15
    
    @pytest.mark.asyncio
    async def test_initialization_lifecycle(self):
        """Test provider initialization lifecycle."""
        provider = ConcreteDBProvider()
        
        # Initially not initialized
        assert provider.initialized is False
        
        # Initialize
        await provider.initialize()
        assert provider.initialized is True
        
        # Shutdown
        await provider.shutdown()
        assert provider.initialized is False
        assert provider._pool is None
    
    @pytest.mark.asyncio
    async def test_execute_select_query(self):
        """Test executing SELECT query."""
        provider = ConcreteDBProvider()
        await provider.initialize()
        
        # Set mock data
        mock_data = [
            {"id": 1, "name": "John", "email": "john@example.com"},
            {"id": 2, "name": "Jane", "email": "jane@example.com"}
        ]
        provider.set_mock_data(mock_data)
        
        # Execute SELECT query
        result = await provider.execute("SELECT * FROM users")
        
        assert result == mock_data
    
    @pytest.mark.asyncio
    async def test_execute_insert_query(self):
        """Test executing INSERT query."""
        provider = ConcreteDBProvider()
        await provider.initialize()
        
        # Execute INSERT query
        result = await provider.execute(
            "INSERT INTO users (name, email) VALUES (%(name)s, %(email)s)",
            {"name": "John", "email": "john@example.com"}
        )
        
        assert result == {"id": 1}
    
    @pytest.mark.asyncio
    async def test_execute_update_query(self):
        """Test executing UPDATE query."""
        provider = ConcreteDBProvider()
        await provider.initialize()
        
        # Execute UPDATE query
        result = await provider.execute(
            "UPDATE users SET name = %(name)s WHERE id = %(id)s",
            {"name": "John Updated", "id": 1}
        )
        
        assert result == {"affected_rows": 1}
    
    @pytest.mark.asyncio
    async def test_execute_delete_query(self):
        """Test executing DELETE query."""
        provider = ConcreteDBProvider()
        await provider.initialize()
        
        # Execute DELETE query
        result = await provider.execute(
            "DELETE FROM users WHERE id = %(id)s",
            {"id": 1}
        )
        
        assert result == {"affected_rows": 1}
    
    @pytest.mark.asyncio
    async def test_execute_many(self):
        """Test executing multiple queries."""
        provider = ConcreteDBProvider()
        await provider.initialize()
        
        # Execute multiple INSERT queries
        params_list = [
            {"name": "John", "email": "john@example.com"},
            {"name": "Jane", "email": "jane@example.com"},
            {"name": "Bob", "email": "bob@example.com"}
        ]
        
        results = await provider.execute_many(
            "INSERT INTO users (name, email) VALUES (%(name)s, %(email)s)",
            params_list
        )
        
        assert len(results) == 3
        assert all("id" in result for result in results)
    
    @pytest.mark.asyncio
    async def test_execute_structured_success(self):
        """Test successful structured query execution."""
        provider = ConcreteDBProvider()
        await provider.initialize()
        
        # Set mock data as list of dictionaries
        mock_data = [
            {"id": 1, "name": "John", "email": "john@example.com", "active": True},
            {"id": 2, "name": "Jane", "email": "jane@example.com", "active": False}
        ]
        provider.set_mock_data(mock_data)
        
        # Execute structured query
        results = await provider.execute_structured(
            "SELECT * FROM users",
            MockRecord
        )
        
        assert len(results) == 2
        assert all(isinstance(record, MockRecord) for record in results)
        assert results[0].id == 1
        assert results[0].name == "John"
        assert results[0].email == "john@example.com"
        assert results[0].active is True
        assert results[1].id == 2
        assert results[1].name == "Jane"
        assert results[1].email == "jane@example.com"
        assert results[1].active is False
    
    @pytest.mark.asyncio
    async def test_execute_structured_empty_results(self):
        """Test structured query execution with empty results."""
        provider = ConcreteDBProvider()
        await provider.initialize()
        
        # Set empty mock data
        provider.set_mock_data([])
        
        # Execute structured query
        results = await provider.execute_structured(
            "SELECT * FROM users WHERE id = 999",
            MockRecord
        )
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_execute_structured_parse_error(self):
        """Test structured query execution with parse error."""
        provider = ConcreteDBProvider()
        await provider.initialize()
        
        # Set invalid mock data (missing required fields)
        mock_data = [
            {"id": 1, "name": "John"}  # Missing email field
        ]
        provider.set_mock_data(mock_data)
        
        # Execute structured query - should raise ProviderError
        with pytest.raises(ProviderError) as exc_info:
            await provider.execute_structured("SELECT * FROM users", MockRecord)
        
        assert "Failed to execute structured query" in str(exc_info.value)
        assert exc_info.value.context.data.component == "test_db"
    
    @pytest.mark.asyncio
    async def test_execute_structured_with_object_having_asdict(self):
        """Test structured query with object having _asdict method."""
        provider = ConcreteDBProvider()
        await provider.initialize()
        
        # Create mock object with _asdict method
        class MockRow:
            def _asdict(self):
                return {"id": 1, "name": "John", "email": "john@example.com", "active": True}
        
        mock_data = [MockRow()]
        provider.set_mock_data(mock_data)
        
        # Execute structured query
        results = await provider.execute_structured("SELECT * FROM users", MockRecord)
        
        assert len(results) == 1
        assert isinstance(results[0], MockRecord)
        assert results[0].id == 1
        assert results[0].name == "John"
    
    @pytest.mark.asyncio
    async def test_execute_structured_with_object_having_dict(self):
        """Test structured query with object having __dict__ attribute."""
        provider = ConcreteDBProvider()
        await provider.initialize()
        
        # Create mock object with __dict__
        class MockRow:
            def __init__(self):
                self.id = 1
                self.name = "John"
                self.email = "john@example.com"
                self.active = True
        
        mock_data = [MockRow()]
        provider.set_mock_data(mock_data)
        
        # Execute structured query
        results = await provider.execute_structured("SELECT * FROM users", MockRecord)
        
        assert len(results) == 1
        assert isinstance(results[0], MockRecord)
        assert results[0].id == 1
        assert results[0].name == "John"
    
    @pytest.mark.asyncio
    async def test_execute_structured_unsupported_type(self):
        """Test structured query with unsupported result type."""
        provider = ConcreteDBProvider()
        await provider.initialize()
        
        # Set mock data with unsupported type
        mock_data = ["unsupported_string_result"]
        provider.set_mock_data(mock_data)
        
        # Execute structured query - should raise ProviderError
        with pytest.raises(ProviderError) as exc_info:
            await provider.execute_structured("SELECT * FROM users", MockRecord)
        
        assert "Failed to execute structured query" in str(exc_info.value)
        assert "Cannot convert result of type" in str(exc_info.value.cause)
    
    @pytest.mark.asyncio
    async def test_transaction_operations(self):
        """Test transaction operations."""
        provider = ConcreteDBProvider()
        await provider.initialize()
        
        # Begin transaction
        transaction = await provider.begin_transaction()
        assert transaction is not None
        
        # Commit transaction
        result = await provider.commit_transaction(transaction)
        assert result is True
        
        # Begin another transaction
        transaction2 = await provider.begin_transaction()
        
        # Rollback transaction
        result = await provider.rollback_transaction(transaction2)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_connection(self):
        """Test connection checking."""
        provider = ConcreteDBProvider()
        
        # Initially not connected
        assert await provider.check_connection() is False
        
        # After initialization, should be connected
        await provider.initialize()
        assert await provider.check_connection() is True
        
        # After shutdown, should be disconnected
        await provider.shutdown()
        assert await provider.check_connection() is False
    
    @pytest.mark.asyncio
    async def test_get_health(self):
        """Test health information retrieval."""
        provider = ConcreteDBProvider()
        
        # Health check when not initialized
        health = await provider.get_health()
        assert health["status"] == "unhealthy"
        
        # Health check when initialized
        await provider.initialize()
        health = await provider.get_health()
        assert health["status"] == "healthy"
        assert "connections" in health
        assert "pool_size" in health


class AbstractDBProvider(DBProvider):
    """Abstract provider implementation to test NotImplementedError methods."""
    
    def __init__(self, name: str = "abstract_db", settings: Optional[DBProviderSettings] = None):
        if settings is None:
            settings = DBProviderSettings()
        super().__init__(name=name, settings=settings)


class TestAbstractMethods:
    """Test that abstract methods raise NotImplementedError."""
    
    @pytest.mark.asyncio
    async def test_abstract_execute(self):
        """Test that execute method raises NotImplementedError."""
        provider = AbstractDBProvider()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement execute"):
            await provider.execute("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_abstract_execute_many(self):
        """Test that execute_many method raises NotImplementedError."""
        provider = AbstractDBProvider()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement execute_many"):
            await provider.execute_many("INSERT INTO table VALUES (?)", [{"value": 1}])
    
    @pytest.mark.asyncio
    async def test_abstract_begin_transaction(self):
        """Test that begin_transaction method raises NotImplementedError."""
        provider = AbstractDBProvider()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement begin_transaction"):
            await provider.begin_transaction()
    
    @pytest.mark.asyncio
    async def test_abstract_commit_transaction(self):
        """Test that commit_transaction method raises NotImplementedError."""
        provider = AbstractDBProvider()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement commit_transaction"):
            await provider.commit_transaction(Mock())
    
    @pytest.mark.asyncio
    async def test_abstract_rollback_transaction(self):
        """Test that rollback_transaction method raises NotImplementedError."""
        provider = AbstractDBProvider()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement rollback_transaction"):
            await provider.rollback_transaction(Mock())
    
    @pytest.mark.asyncio
    async def test_abstract_check_connection(self):
        """Test that check_connection method raises NotImplementedError."""
        provider = AbstractDBProvider()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement check_connection"):
            await provider.check_connection()
    
    @pytest.mark.asyncio
    async def test_abstract_get_health(self):
        """Test that get_health method raises NotImplementedError."""
        provider = AbstractDBProvider()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement get_health"):
            await provider.get_health()