"""Tests for PostgreSQL database provider."""
import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Any, Dict, List

# Test both with and without asyncpg installed
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

from flowlib.providers.db.postgres.provider import (
    PostgreSQLProvider,
    PostgreSQLProviderSettings,
)
from flowlib.core.errors.errors import ProviderError


class TestPostgreSQLProviderSettings:
    """Test PostgreSQL provider settings."""
    
    def test_default_settings(self):
        """Test default PostgreSQL provider settings."""
        settings = PostgreSQLProviderSettings(
            database="test_db",
            username="test_user", 
            password="test_pass"
        )
        
        # Test PostgreSQL-specific defaults
        assert settings.host == "localhost"
        assert settings.port == 5432
        assert settings.database == "test_db"
        assert settings.username == "test_user"
        assert settings.password == "test_pass"
        assert settings.db_schema == "public"
        assert settings.ssl_mode == "prefer"
        assert settings.statement_timeout is None
        assert settings.min_connections == 1
        assert settings.max_connections == 10
        assert settings.connect_args == {}
        
        # Test inherited provider settings
        assert settings.timeout_seconds == 60.0
        assert settings.max_retries == 3
        assert settings.retry_delay_seconds == 1.0
        assert settings.api_key is None
        assert settings.api_base is None
    
    def test_custom_settings(self):
        """Test custom PostgreSQL provider settings."""
        settings = PostgreSQLProviderSettings(
            host="custom-postgres",
            port=5433,
            database="test_db",
            username="test_user",
            password="test_pass",
            db_schema="custom_schema",
            statement_timeout=30000,
            ssl_mode="require",
            # Removed invalid fields: server_version, pool_size, min_size, connect_timeout, query_timeout, retry_count, retry_delay, auto_reconnect, use_ssl, ssl_ca_cert, ssl_cert, ssl_key, application_name
            min_connections=2,  # Use actual field
            max_connections=20,  # Use actual field
            timeout_seconds=60.0,  # Use provider base field
            max_retries=5,  # Use provider base field
            retry_delay_seconds=2.0,  # Use provider base field
            connect_args={"server_settings": {"timezone": "UTC"}}
        )
        
        assert settings.host == "custom-postgres"
        assert settings.port == 5433
        assert settings.database == "test_db"
        assert settings.username == "test_user"
        assert settings.password == "test_pass"
        assert settings.db_schema == "custom_schema"
        assert settings.statement_timeout == 30000
        assert settings.ssl_mode == "require"
        # Test actual fields
        assert settings.min_connections == 2
        assert settings.max_connections == 20
        assert settings.timeout_seconds == 60.0
        assert settings.max_retries == 5
        assert settings.retry_delay_seconds == 2.0
        assert settings.connect_args == {"server_settings": {"timezone": "UTC"}}

    def test_settings_inheritance(self):
        """Test that PostgreSQLProviderSettings inherits from ProviderSettings."""
        from flowlib.providers.core.base import ProviderSettings
        settings = PostgreSQLProviderSettings(database="test", username="user", password="pass")
        assert isinstance(settings, ProviderSettings)


class TestPostgreSQLProvider:
    """Test PostgreSQL provider."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return PostgreSQLProviderSettings(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass",
            db_schema="test_schema",
            max_connections=5
        )
    
    @pytest.fixture
    def provider(self, settings):
        """Create test provider."""
        return PostgreSQLProvider(name="test_postgres", settings=settings)
    
    @pytest.fixture
    def mock_pool(self):
        """Create mock connection pool."""
        mock = AsyncMock()
        mock.close = AsyncMock()
        mock.release = AsyncMock()
        mock._holders = []
        mock._queue = Mock()
        mock._queue._queue = []
        
        # Create proper async context manager for acquire()
        class MockAcquireContextManager:
            def __init__(self, connection):
                self.connection = connection
                
            async def __aenter__(self):
                return self.connection
                
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False
        
        # Create a mock that can be both awaited and used as context manager
        class MockAcquireResult:
            def __init__(self, connection):
                self.connection = connection
                
            async def __aenter__(self):
                return self.connection
                
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False
                
            def __await__(self):
                # Allow await mock.acquire() to return the connection directly
                async def _awaitable():
                    return self.connection
                return _awaitable().__await__()
        
        # Mock the acquire method to return the hybrid object
        def acquire():
            return MockAcquireResult(mock.connection)
            
        mock.acquire = acquire
        mock.connection = AsyncMock()
        
        # Create a mock transaction object
        mock_transaction_obj = AsyncMock()
        mock_transaction_obj.start = AsyncMock()
        mock_transaction_obj.commit = AsyncMock()
        mock_transaction_obj.rollback = AsyncMock()
        
        # Mock the transaction method to return the mock transaction
        mock_transaction_method = Mock(return_value=mock_transaction_obj)
        mock.connection.transaction = mock_transaction_method
        return mock
    
    @pytest.fixture
    def mock_connection(self):
        """Create mock database connection."""
        mock = AsyncMock()
        mock.fetch.return_value = []
        mock.execute.return_value = "INSERT 0 1"
        mock.fetchval.return_value = 1
        mock.transaction = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_transaction(self):
        """Create mock database transaction."""
        mock = AsyncMock()
        mock.start = AsyncMock()
        mock.commit = AsyncMock()
        mock.rollback = AsyncMock()
        return mock
    
    def test_init_default_settings(self):
        """Test provider initialization with default settings."""
        provider = PostgreSQLProvider()
        
        assert provider.name == "postgres"
        assert isinstance(provider._settings, PostgreSQLProviderSettings)
        assert provider._pool is None
        assert provider._json_encoders is not None
    
    def test_init_custom_settings(self, settings):
        """Test provider initialization with custom settings."""
        provider = PostgreSQLProvider(name="custom_postgres", settings=settings)
        
        assert provider.name == "custom_postgres"
        assert provider._settings == settings
        assert provider._pool is None
    
    @patch('flowlib.providers.db.postgres.provider.asyncpg')
    async def test_initialize_success(self, mock_asyncpg, provider, mock_pool):
        """Test successful provider initialization."""
        # Setup mocks
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)
        
        # Initialize provider
        await provider.initialize()
        
        # Verify pool creation
        mock_asyncpg.create_pool.assert_called_once()
        assert provider._pool == mock_pool
        assert provider._initialized is True
    
    async def test_initialize_asyncpg_not_installed(self, provider):
        """Test provider initialization without asyncpg."""
        with patch('flowlib.providers.db.postgres.provider.asyncpg', None):
            with pytest.raises(ProviderError) as exc_info:
                await provider.initialize()
            
            assert "asyncpg package not installed" in str(exc_info.value)
            assert exc_info.value.context.data.component == "test_postgres"
    
    @patch('flowlib.providers.db.postgres.provider.asyncpg')
    async def test_initialize_connection_error(self, mock_asyncpg, provider):
        """Test provider initialization with connection error."""
        # Setup mocks
        mock_asyncpg.create_pool.side_effect = Exception("Connection failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.initialize()
        
        assert "Failed to initialize PostgreSQL provider" in str(exc_info.value)
        assert exc_info.value.context.data.component == "test_postgres"
    
    async def test_shutdown_success(self, provider, mock_pool):
        """Test successful provider shutdown."""
        # Set up provider state
        provider._pool = mock_pool
        provider._initialized = True
        
        # Shutdown provider
        await provider.shutdown()
        
        # Verify cleanup
        mock_pool.close.assert_called_once()
        assert provider._pool is None
        assert provider._initialized is False
    
    async def test_shutdown_not_initialized(self, provider):
        """Test shutdown when not initialized."""
        # Should not raise error
        await provider.shutdown()
        assert provider._pool is None
        assert provider._initialized is False
    
    async def test_shutdown_error(self, provider, mock_pool):
        """Test shutdown with error."""
        # Set up provider state
        provider._pool = mock_pool
        provider._initialized = True
        mock_pool.close.side_effect = Exception("Shutdown failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.shutdown()
        
        assert "Failed to shut down PostgreSQL provider" in str(exc_info.value)
        assert exc_info.value.context.data.component == "test_postgres"
    
    async def test_execute_select_success(self, provider, mock_pool, mock_connection):
        """Test successful SELECT query execution."""
        # Set up provider state
        provider._pool = mock_pool
        provider._initialized = True
        
        # Setup mocks - use the connection from the pool fixture
        mock_pool.connection.fetch.return_value = [
            {"id": 1, "name": "test1"},
            {"id": 2, "name": "test2"}
        ]
        
        # Execute query
        result = await provider.execute("SELECT * FROM users", {"limit": 10})
        
        # Verify results
        assert result == [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]
        mock_pool.connection.fetch.assert_called_once_with("SELECT * FROM users", 10)
    
    async def test_execute_insert_success(self, provider, mock_pool, mock_connection):
        """Test successful INSERT query execution."""
        # Set up provider state
        provider._pool = mock_pool
        provider._initialized = True
        
        # Setup mocks - use the connection from the pool fixture
        mock_pool.connection.execute.return_value = "INSERT 0 1"
        
        # Execute query
        result = await provider.execute("INSERT INTO users (name) VALUES ($1)", {"name": "test"})
        
        # Verify results
        assert result == [{"affected_rows": 1}]
        mock_pool.connection.execute.assert_called_once_with("INSERT INTO users (name) VALUES ($1)", "test")
    
    async def test_execute_not_initialized(self, provider):
        """Test execute when not initialized."""
        with pytest.raises(ProviderError) as exc_info:
            await provider.execute("SELECT 1")
        
        assert "Provider not initialized" in str(exc_info.value)
        assert exc_info.value.context.data.component == "test_postgres"
    
    async def test_execute_with_params(self, provider, mock_pool, mock_connection):
        """Test execute with named parameters."""
        # Set up provider state
        provider._pool = mock_pool
        provider._initialized = True
        
        # Setup mocks
        mock_pool.connection.fetch.return_value = [{"count": 5}]
        
        # Execute query with named parameters
        result = await provider.execute(
            "SELECT COUNT(*) as count FROM users WHERE age > :min_age AND city = :city",
            {"min_age": 18, "city": "New York"}
        )
        
        # Verify parameter conversion
        expected_query = "SELECT COUNT(*) as count FROM users WHERE age > $1 AND city = $2"
        mock_pool.connection.fetch.assert_called_once_with(expected_query, 18, "New York")
        assert result == [{"count": 5}]
    
    async def test_execute_many_success(self, provider, mock_pool, mock_connection, mock_transaction):
        """Test successful batch query execution."""
        # Set up provider state
        provider._pool = mock_pool
        provider._initialized = True
        
        # Setup mocks - the transaction is handled by the fixture
        mock_pool.connection.execute.side_effect = ["INSERT 0 1", "INSERT 0 1", "INSERT 0 1"]
        
        # Execute batch
        params_list = [
            {"name": "user1", "age": 25},
            {"name": "user2", "age": 30},
            {"name": "user3", "age": 35}
        ]
        result = await provider.execute_many(
            "INSERT INTO users (name, age) VALUES (:name, :age)",
            params_list
        )
        
        # Verify results
        assert result == ["INSERT 0 1", "INSERT 0 1", "INSERT 0 1"]
        assert mock_pool.connection.execute.call_count == 3
    
    async def test_execute_many_empty_params(self, provider, mock_pool):
        """Test execute_many with empty parameters list."""
        # Set up provider state
        provider._pool = mock_pool
        provider._initialized = True
        
        result = await provider.execute_many("INSERT INTO users (name) VALUES (:name)", [])
        assert result == []
    
    async def test_begin_transaction_success(self, provider, mock_pool, mock_connection, mock_transaction):
        """Test successful transaction begin."""
        # Set up provider state
        provider._pool = mock_pool
        provider._initialized = True
        
        # Begin transaction
        result = await provider.begin_transaction()
        
        # Verify results - should be a wrapper object
        assert hasattr(result, 'connection')
        assert hasattr(result, 'transaction')
        assert result.connection == mock_pool.connection
        mock_pool.connection.transaction.assert_called_once()
        mock_pool.connection.transaction.return_value.start.assert_called_once()
    
    async def test_begin_transaction_not_initialized(self, provider):
        """Test begin_transaction when not initialized."""
        with pytest.raises(ProviderError) as exc_info:
            await provider.begin_transaction()
        
        assert "Provider not initialized" in str(exc_info.value)
        assert exc_info.value.context.data.component == "test_postgres"
    
    async def test_commit_transaction_success(self, provider, mock_pool, mock_connection, mock_transaction):
        """Test successful transaction commit."""
        # Set up provider state
        provider._pool = mock_pool
        provider._initialized = True
        
        # Create a mock transaction wrapper
        class MockTransactionWrapper:
            def __init__(self, connection, transaction):
                self.connection = connection
                self.transaction = transaction
        
        mock_wrapper = MockTransactionWrapper(mock_pool.connection, mock_transaction)
        
        # Commit transaction
        result = await provider.commit_transaction(mock_wrapper)
        
        # Verify results
        assert result is True
        mock_transaction.commit.assert_called_once()
        mock_pool.release.assert_called_once_with(mock_pool.connection)
    
    async def test_commit_transaction_invalid(self, provider):
        """Test commit_transaction with invalid transaction."""
        with pytest.raises(ProviderError) as exc_info:
            await provider.commit_transaction(None)
        
        assert "Invalid transaction object" in str(exc_info.value)
        assert exc_info.value.context.data.component == "test_postgres"
    
    async def test_rollback_transaction_success(self, provider, mock_pool, mock_connection, mock_transaction):
        """Test successful transaction rollback."""
        # Set up provider state
        provider._pool = mock_pool
        provider._initialized = True
        
        # Create a mock transaction wrapper
        class MockTransactionWrapper:
            def __init__(self, connection, transaction):
                self.connection = connection
                self.transaction = transaction
        
        mock_wrapper = MockTransactionWrapper(mock_pool.connection, mock_transaction)
        
        # Rollback transaction
        result = await provider.rollback_transaction(mock_wrapper)
        
        # Verify results
        assert result is True
        mock_transaction.rollback.assert_called_once()
        mock_pool.release.assert_called_once_with(mock_pool.connection)
    
    async def test_rollback_transaction_invalid(self, provider):
        """Test rollback_transaction with invalid transaction."""
        with pytest.raises(ProviderError) as exc_info:
            await provider.rollback_transaction(None)
        
        assert "Invalid transaction object" in str(exc_info.value)
        assert exc_info.value.context.data.component == "test_postgres"
    
    async def test_check_connection_success(self, provider, mock_pool, mock_connection):
        """Test successful connection check."""
        # Set up provider state
        provider._pool = mock_pool
        provider._initialized = True
        
        # Setup mocks
        mock_pool.connection.fetchval.return_value = 1
        
        result = await provider.check_connection()
        
        assert result is True
        mock_pool.connection.fetchval.assert_called_once_with("SELECT 1")
    
    async def test_check_connection_failure(self, provider, mock_pool, mock_connection):
        """Test failed connection check."""
        # Set up provider state
        provider._pool = mock_pool
        provider._initialized = True
        
        # Setup mocks
        mock_pool.connection.fetchval.side_effect = Exception("Connection failed")
        
        result = await provider.check_connection()
        assert result is False
    
    async def test_check_connection_not_initialized(self, provider):
        """Test connection check when not initialized."""
        result = await provider.check_connection()
        assert result is False
    
    async def test_get_health_success(self, provider, mock_pool, mock_connection):
        """Test successful health check."""
        # Set up provider state
        provider._pool = mock_pool
        provider._initialized = True
        
        # Setup mocks
        mock_pool.connection.fetchval.side_effect = [1, "PostgreSQL 13.0"]
        
        result = await provider.get_health()
        
        assert result["status"] == "healthy"
        assert result["connection_active"] is True
        assert result["version"] == "PostgreSQL 13.0"
        assert "pool" in result
        assert result["host"] == "localhost"
        assert result["database"] == "test_db"
    
    async def test_get_health_not_initialized(self, provider):
        """Test health check when not initialized."""
        result = await provider.get_health()
        
        assert result["status"] == "not_initialized"
        assert result["pool"] is None
        assert result["connection_active"] is False
    
    async def test_get_health_error(self, provider, mock_pool, mock_connection):
        """Test health check with error."""
        # Set up provider state
        provider._pool = mock_pool
        provider._initialized = True
        
        # Setup mocks - mock the acquire to raise an exception
        mock_pool.acquire.side_effect = Exception("Pool error")
        
        result = await provider.get_health()
        
        # The check_connection method catches the exception and returns False,
        # so the status will be "unhealthy" not "error"
        assert result["status"] == "unhealthy"
        assert result["connection_active"] is False
    
    def test_create_connection_string(self, provider):
        """Test DSN creation."""
        dsn = provider._create_connection_string()
        
        # Should be a proper PostgreSQL URL
        assert dsn.startswith("postgresql://")
        assert "test_user:test_pass@localhost:5432/test_db" in dsn
    
    def test_create_connection_string_minimal(self):
        """Test DSN creation with minimal settings."""
        settings = PostgreSQLProviderSettings(database="testdb")
        provider = PostgreSQLProvider(settings=settings)
        
        dsn = provider._create_connection_string()
        # Should be URL format with defaults (including SSL mode)
        assert dsn == "postgresql://localhost:5432/testdb?sslmode=prefer"
    
    def test_create_connection_string_with_ssl(self):
        """Test DSN creation with SSL mode."""
        settings = PostgreSQLProviderSettings(
            host="localhost",
            database="testdb",
            ssl_mode="require"
        )
        provider = PostgreSQLProvider(settings=settings)
        
        dsn = provider._create_connection_string()
        assert dsn == "postgresql://localhost:5432/testdb?sslmode=require"
    
    def test_convert_params_named(self, provider):
        """Test parameter conversion from named to positional."""
        query = "SELECT * FROM users WHERE age > :min_age AND city = :city"
        params = {"min_age": 18, "city": "New York"}
        
        new_query, positional_params = provider._convert_params(query, params)
        
        assert "$1" in new_query
        assert "$2" in new_query
        assert ":min_age" not in new_query
        assert ":city" not in new_query
        assert positional_params == [18, "New York"]
    
    def test_convert_params_empty(self, provider):
        """Test parameter conversion with no parameters."""
        query = "SELECT * FROM users"
        
        new_query, positional_params = provider._convert_params(query, None)
        
        assert new_query == query
        assert positional_params == []
    
    def test_convert_params_at_style(self, provider):
        """Test parameter conversion with @ style parameters."""
        query = "SELECT * FROM users WHERE age > @min_age"
        params = {"min_age": 18}
        
        new_query, positional_params = provider._convert_params(query, params)
        
        assert "$1" in new_query
        assert "@min_age" not in new_query
        assert positional_params == [18]
    
    def test_should_retry_connection_errors(self, provider):
        """Test retry logic for connection errors."""
        # Mock asyncpg not being available for testing
        with patch('flowlib.providers.db.postgres.provider.asyncpg', None):
            # Test with generic connection error
            error = Exception("connection closed")
            assert provider._should_retry(error) is True
            
            # Test with non-connection error
            error = Exception("syntax error")
            assert provider._should_retry(error) is False
    
    @patch('flowlib.providers.db.postgres.provider.asyncpg')
    def test_should_retry_asyncpg_errors(self, mock_asyncpg, provider):
        """Test retry logic for asyncpg-specific errors."""
        # Create mock exception classes
        class ConnectionDoesNotExistError(Exception):
            pass
        
        class ConnectionFailureError(Exception):
            pass
        
        class InterfaceError(Exception):
            pass
        
        class DataError(Exception):
            pass
        
        # Set up mock asyncpg module
        mock_asyncpg.ConnectionDoesNotExistError = ConnectionDoesNotExistError
        mock_asyncpg.ConnectionFailureError = ConnectionFailureError
        mock_asyncpg.InterfaceError = InterfaceError
        mock_asyncpg.DataError = DataError
        
        # Test retriable errors
        assert provider._should_retry(ConnectionDoesNotExistError()) is True
        assert provider._should_retry(ConnectionFailureError()) is True
        assert provider._should_retry(InterfaceError()) is True
        
        # Test non-retriable errors
        assert provider._should_retry(DataError()) is False
    
    def test_provider_registration(self):
        """Test that provider is properly registered."""
        # This would be tested in integration tests
        # Here we just verify the decorator is applied
        assert hasattr(PostgreSQLProvider, '__provider_type__')
        assert hasattr(PostgreSQLProvider, '__provider_name__')
        assert PostgreSQLProvider.__provider_type__ == 'db'
        assert PostgreSQLProvider.__provider_name__ == 'postgresql'


@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg package not available")
@pytest.mark.integration
class TestPostgreSQLProviderIntegration:
    """Integration tests for PostgreSQL provider.
    
    These tests require a running PostgreSQL instance.
    """
    
    @pytest.fixture
    def settings(self, postgres_settings):
        """Create integration test settings from global config."""
        return postgres_settings
    
    @pytest.fixture
    async def provider(self, settings):
        """Create and initialize provider for integration tests."""
        provider = PostgreSQLProvider(name="integration_postgres", settings=settings)
        
        try:
            await provider.initialize()
            yield provider
        finally:
            # Cleanup
            try:
                await provider.shutdown()
            except:
                pass
    
    async def test_full_database_cycle(self, provider):
        """Test complete database operations cycle."""
        # Clean up any existing test data
        await provider.execute("DROP TABLE IF EXISTS test_users")
        
        # Create test table
        await provider.execute("""
            CREATE TABLE IF NOT EXISTS test_users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                age INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert test data
        result = await provider.execute(
            "INSERT INTO test_users (name, age) VALUES (:name, :age)",
            {"name": "John Doe", "age": 30}
        )
        assert result[0]["affected_rows"] == 1
        
        # Query data
        users = await provider.execute(
            "SELECT * FROM test_users WHERE name = :name",
            {"name": "John Doe"}
        )
        assert len(users) == 1
        assert users[0]["name"] == "John Doe"
        assert users[0]["age"] == 30
        
        # Update data
        result = await provider.execute(
            "UPDATE test_users SET age = :age WHERE name = :name",
            {"age": 31, "name": "John Doe"}
        )
        assert result[0]["affected_rows"] == 1
        
        # Delete data
        result = await provider.execute(
            "DELETE FROM test_users WHERE name = :name",
            {"name": "John Doe"}
        )
        assert result[0]["affected_rows"] == 1
        
        # Clean up
        await provider.execute("DROP TABLE IF EXISTS test_users")
    
    async def test_batch_operations(self, provider):
        """Test batch database operations."""
        # Clean up any existing test data
        await provider.execute("DROP TABLE IF EXISTS batch_test")
        
        # Create test table
        await provider.execute("""
            CREATE TABLE IF NOT EXISTS batch_test (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                value INTEGER
            )
        """)
        
        # Batch insert
        params_list = [
            {"name": "item1", "value": 10},
            {"name": "item2", "value": 20},
            {"name": "item3", "value": 30}
        ]
        
        results = await provider.execute_many(
            "INSERT INTO batch_test (name, value) VALUES (:name, :value)",
            params_list
        )
        assert len(results) == 3
        
        # Verify data
        items = await provider.execute("SELECT * FROM batch_test ORDER BY id")
        assert len(items) == 3
        assert items[0]["name"] == "item1"
        assert items[1]["value"] == 20
        
        # Clean up
        await provider.execute("DROP TABLE IF EXISTS batch_test")
    
    async def test_transactions(self, provider):
        """Test transaction handling."""
        # Clean up any existing test data
        await provider.execute("DROP TABLE IF EXISTS transaction_test")
        
        # Create test table
        await provider.execute("""
            CREATE TABLE IF NOT EXISTS transaction_test (
                id SERIAL PRIMARY KEY,
                amount INTEGER NOT NULL
            )
        """)
        
        # Test successful transaction
        transaction = await provider.begin_transaction()
        try:
            # Execute operations within transaction using the connection directly
            await transaction.execute("INSERT INTO transaction_test (amount) VALUES ($1)", 100)
            await transaction.execute("INSERT INTO transaction_test (amount) VALUES ($1)", 200)
            
            # Commit transaction
            await provider.commit_transaction(transaction)
        except Exception:
            await provider.rollback_transaction(transaction)
            raise
        
        # Verify data was committed
        result = await provider.execute("SELECT COUNT(*) as count FROM transaction_test")
        assert result[0]["count"] == 2
        
        # Test rollback
        transaction = await provider.begin_transaction()
        try:
            await transaction.execute("INSERT INTO transaction_test (amount) VALUES ($1)", 300)
            # Simulate error
            raise Exception("Simulated error")
        except Exception:
            await provider.rollback_transaction(transaction)
        
        # Verify rollback worked
        result = await provider.execute("SELECT COUNT(*) as count FROM transaction_test")
        assert result[0]["count"] == 2  # Should still be 2
        
        # Clean up
        await provider.execute("DROP TABLE IF EXISTS transaction_test")
    
    async def test_connection_health(self, provider):
        """Test connection and health monitoring."""
        # Check connection
        assert await provider.check_connection() is True
        
        # Get health information
        health = await provider.get_health()
        assert health["status"] == "healthy"
        assert health["connection_active"] is True
        assert "version" in health
        assert "pool" in health