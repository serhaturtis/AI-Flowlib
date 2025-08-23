"""Tests for SQLite database provider."""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Any, Dict, List

# Test both with and without aiosqlite installed
try:
    import aiosqlite
    import sqlite3
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False

from flowlib.providers.db.sqlite.provider import (
    SQLiteDBProvider,
    SQLiteProviderSettings,
)
from flowlib.core.errors.errors import ProviderError


class TestSQLiteDBProviderSettings:
    """Test SQLite provider settings."""
    
    def test_default_settings(self):
        """Test default SQLite provider settings."""
        settings = SQLiteProviderSettings(database_path="/tmp/test.db")
        
        # Test SQLite-specific defaults
        assert settings.database_path == "/tmp/test.db"
        assert settings.journal_mode == "WAL"
        assert settings.isolation_level is None
        assert settings.timeout == 5.0
        assert settings.detect_types == 0
        assert settings.create_if_missing is True
        assert settings.connect_args == {}
        
        # Test inherited provider settings
        assert settings.api_key is None
        assert settings.api_base is None
        assert settings.timeout_seconds == 60.0
        assert settings.max_retries == 3
        assert settings.retry_delay_seconds == 1.0
    
    def test_custom_settings(self):
        """Test custom SQLite provider settings."""
        settings = SQLiteProviderSettings(
            database_path="/custom/path/test.db",
            journal_mode="DELETE",
            isolation_level="IMMEDIATE",
            timeout=10.0,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES if AIOSQLITE_AVAILABLE else 3,
            create_if_missing=False,
            # Use base ProviderSettings fields
            timeout_seconds=120.0,
            max_retries=5,
            retry_delay_seconds=2.0,
            connect_args={"check_same_thread": False}
        )
        
        assert settings.database_path == "/custom/path/test.db"
        assert settings.journal_mode == "DELETE"
        assert settings.isolation_level == "IMMEDIATE"
        assert settings.timeout == 10.0
        assert settings.detect_types == (sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES if AIOSQLITE_AVAILABLE else 3)
        assert settings.create_if_missing is False
        # Test base provider settings
        assert settings.timeout_seconds == 120.0
        assert settings.max_retries == 5
        assert settings.retry_delay_seconds == 2.0
        assert settings.connect_args == {"check_same_thread": False}
    
    def test_validation_missing_database_path(self):
        """Test validation when database_path is missing."""
        with pytest.raises(Exception):  # Pydantic validation error
            SQLiteProviderSettings()


class TestSQLiteDBProvider:
    """Test SQLite provider."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database file path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        yield path
        # Cleanup
        try:
            os.unlink(path)
        except OSError:
            pass
    
    @pytest.fixture
    def settings(self, temp_db_path):
        """Create test settings."""
        return SQLiteProviderSettings(
            database_path=temp_db_path,
            journal_mode="WAL",
            timeout=10.0,
            # Use valid base provider settings instead of pool_size
            max_retries=3
        )
    
    @pytest.fixture
    def provider(self, settings):
        """Create test provider."""
        return SQLiteDBProvider(name="test_sqlite", settings=settings)
    
    @pytest.fixture
    def mock_connection(self):
        """Create mock database connection."""
        mock = AsyncMock()
        mock.execute.return_value = MagicMock()
        mock.executemany.return_value = MagicMock()
        mock.fetchall.return_value = []
        mock.fetchone.return_value = None
        mock.commit = AsyncMock()
        mock.rollback = AsyncMock()
        mock.close = AsyncMock()
        mock.cursor.return_value = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_cursor(self):
        """Create mock database cursor."""
        mock = AsyncMock()
        mock.execute.return_value = None
        mock.executemany.return_value = None
        mock.fetchall.return_value = []
        mock.fetchone.return_value = None
        mock.close = AsyncMock()
        mock.rowcount = 0
        mock.description = []
        return mock
    
    def test_init_default_settings(self, temp_db_path):
        """Test provider initialization with default settings."""
        settings = SQLiteProviderSettings(database_path=temp_db_path)
        provider = SQLiteDBProvider(settings=settings)
        
        assert provider.name == "sqlite"
        assert isinstance(provider._settings, SQLiteProviderSettings)
        assert provider._connection is None
    
    def test_init_custom_settings(self, settings):
        """Test provider initialization with custom settings."""
        provider = SQLiteDBProvider(name="custom_sqlite", settings=settings)
        
        assert provider.name == "custom_sqlite"
        assert provider._settings == settings
        assert provider._connection is None
    
    @patch('flowlib.providers.db.sqlite.provider.aiosqlite')
    async def test_initialize_success(self, mock_aiosqlite, provider, mock_connection):
        """Test successful provider initialization."""
        # Setup mocks
        mock_aiosqlite.connect = AsyncMock(return_value=mock_connection)
        
        # Initialize provider
        await provider.initialize()
        
        # Verify initialization
        assert provider._initialized is True
        assert provider._connection == mock_connection
        
        # Verify connections were configured
        mock_connection.execute.assert_called()  # WAL mode and foreign keys setup
    
    async def test_initialize_aiosqlite_not_installed(self, provider):
        """Test provider initialization without aiosqlite."""
        with patch('flowlib.providers.db.sqlite.provider.aiosqlite', None):
            with pytest.raises(ProviderError) as exc_info:
                await provider.initialize()
            
            assert "aiosqlite package not installed" in str(exc_info.value)
            assert exc_info.value.provider_context.provider_name == "test_sqlite"
    
    @patch('flowlib.providers.db.sqlite.provider.aiosqlite')
    async def test_initialize_database_creation_error(self, mock_aiosqlite, provider):
        """Test provider initialization with database creation error."""
        # Setup mocks to simulate file creation error
        mock_aiosqlite.connect.side_effect = Exception("Permission denied")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.initialize()
        
        assert "Failed to connect to SQLite database" in str(exc_info.value)
        assert exc_info.value.provider_context.provider_name == "test_sqlite"
    
    @patch('flowlib.providers.db.sqlite.provider.aiosqlite')
    async def test_initialize_with_directory_creation(self, mock_aiosqlite, mock_connection):
        """Test provider initialization with directory creation."""
        # Create settings with non-existent directory
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "subdir", "test.db")
            settings = SQLiteProviderSettings(database_path=db_path)
            provider = SQLiteDBProvider(settings=settings)
            
            mock_aiosqlite.connect = AsyncMock(return_value=mock_connection)
            
            await provider.initialize()
            
            # Directory should be created
            assert os.path.exists(os.path.dirname(db_path))
            assert provider._initialized is True
    
    async def test_shutdown_success(self, provider):
        """Test successful provider shutdown."""
        # Set up provider state
        mock_connection = AsyncMock()
        provider._connection = mock_connection
        provider._initialized = True
        
        # Shutdown provider
        await provider.shutdown()
        
        # Verify cleanup
        mock_connection.close.assert_called_once()
        
        assert provider._connection is None
        assert provider._initialized is False
    
    async def test_shutdown_not_initialized(self, provider):
        """Test shutdown when not initialized."""
        # Should not raise error
        await provider.shutdown()
        assert provider._connection is None
        assert provider._initialized is False
    
    async def test_shutdown_error(self, provider):
        """Test shutdown with error."""
        # Set up provider state
        mock_connection = AsyncMock()
        mock_connection.close.side_effect = Exception("Close failed")
        provider._connection = mock_connection
        provider._initialized = True
        
        # Shutdown should handle errors gracefully and not raise
        await provider.shutdown()
        
        # Provider should remain initialized if shutdown fails (base provider behavior)
        # The base provider only sets _initialized = False if shutdown succeeds
        assert provider._initialized is True
    
    async def test_connection_state_after_init(self, provider):
        """Test connection state after initialization."""
        # Set up provider state
        mock_connection = AsyncMock()
        provider._connection = mock_connection
        provider._initialized = True
        
        # Verify connection is available
        assert provider._connection == mock_connection
        assert provider._initialized is True
    
    async def test_connection_lazy_initialization(self, provider, mock_connection):
        """Test that connection is created on demand during query execution."""
        # Set up provider state - not initialized, no connection
        provider._connection = None
        provider._initialized = False
        
        with patch('flowlib.providers.db.sqlite.provider.aiosqlite') as mock_aiosqlite:
            mock_aiosqlite.connect = AsyncMock(return_value=mock_connection)
            mock_connection.execute.return_value = AsyncMock()
            mock_connection.execute.return_value.description = []
            mock_connection.execute.return_value.fetchall.return_value = []
            
            # Execute query (should trigger initialization)
            await provider.execute_query("SELECT 1")
            
            # Verify connection was created
            mock_aiosqlite.connect.assert_called_once()
            assert provider._connection == mock_connection
    
    async def test_execute_without_initialization(self, provider):
        """Test executing query without explicit initialization."""
        # Provider should handle this gracefully by auto-initializing
        with patch('flowlib.providers.db.sqlite.provider.aiosqlite') as mock_aiosqlite:
            mock_connection = AsyncMock()
            mock_aiosqlite.connect = AsyncMock(return_value=mock_connection)
            mock_connection.execute.return_value = AsyncMock()
            mock_connection.execute.return_value.description = []
            mock_connection.execute.return_value.fetchall.return_value = []
            
            # This should not raise an error - should auto-initialize
            await provider.execute_query("SELECT 1")
            
            # Verify initialization occurred
            mock_aiosqlite.connect.assert_called_once()
    
    async def test_connection_reuse(self, provider):
        """Test that SQLite provider reuses the same connection."""
        # Set up provider state
        mock_connection = AsyncMock()
        provider._connection = mock_connection
        provider._initialized = True
        
        mock_connection.execute.return_value = AsyncMock()
        mock_connection.execute.return_value.description = []
        mock_connection.execute.return_value.fetchall.return_value = []
        
        # Execute multiple queries
        await provider.execute_query("SELECT 1")
        await provider.execute_query("SELECT 2")
        
        # Verify same connection was reused
        assert provider._connection == mock_connection
        assert mock_connection.execute.call_count == 2
    
    async def test_shutdown_closes_connection(self, provider):
        """Test that shutdown properly closes the connection."""
        # Set up provider state with existing connection
        existing_connection = AsyncMock()
        provider._connection = existing_connection
        provider._initialized = True
        
        # Shutdown provider
        await provider.shutdown()
        
        # Connection should be closed and cleared
        existing_connection.close.assert_called_once()
        assert provider._connection is None
        assert provider._initialized is False
    
    async def test_execute_select_success(self, provider, mock_connection):
        """Test successful SELECT query execution."""
        # Set up provider state
        provider._connection = mock_connection
        provider._initialized = True
        
        # Setup mocks
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = [
            (1, "test1"),
            (2, "test2")
        ]
        mock_cursor.description = [("id",), ("name",)]
        mock_connection.execute.return_value = mock_cursor
        
        # Execute query
        result = await provider.execute_query("SELECT * FROM users")
        
        # Verify results
        assert result == [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]
        mock_connection.execute.assert_called_once_with("SELECT * FROM users", ())
    
    async def test_execute_insert_success(self, provider, mock_connection):
        """Test successful INSERT query execution."""
        # Set up provider state
        provider._connection = mock_connection
        provider._initialized = True
        
        # Setup mocks
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 1
        mock_connection.execute.return_value = mock_cursor
        
        # Execute query
        result = await provider.execute_update("INSERT INTO users (name) VALUES (?)", ("test",))
        
        # Verify results
        assert result == 1
        mock_connection.execute.assert_called_once_with("INSERT INTO users (name) VALUES (?)", ("test",))
        mock_connection.commit.assert_called_once()
    
    async def test_execute_with_params(self, provider, mock_connection):
        """Test execute with parameters."""
        # Set up provider state
        provider._connection = mock_connection
        provider._initialized = True
        
        # Setup mocks
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = [(5,)]
        mock_cursor.description = [("count",)]
        mock_connection.execute.return_value = mock_cursor
        
        # Execute query with parameters
        result = await provider.execute_query(
            "SELECT COUNT(*) as count FROM users WHERE age > ? AND city = ?",
            (18, "New York")
        )
        
        # Verify results
        mock_connection.execute.assert_called_once_with(
            "SELECT COUNT(*) as count FROM users WHERE age > ? AND city = ?", 
            (18, "New York")
        )
        assert result == [{"count": 5}]
    
    async def test_execute_transaction_success(self, provider, mock_connection):
        """Test successful transaction execution."""
        # Set up provider state
        provider._connection = mock_connection
        provider._initialized = True
        
        # Setup mocks
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 1
        mock_connection.execute.return_value = mock_cursor
        mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_connection.__aexit__ = AsyncMock(return_value=False)
        
        # Execute transaction
        queries = [
            ("INSERT INTO users (name, age) VALUES (?, ?)", ("user1", 25)),
            ("INSERT INTO users (name, age) VALUES (?, ?)", ("user2", 30)),
            ("INSERT INTO users (name, age) VALUES (?, ?)", ("user3", 35))
        ]
        result = await provider.execute_transaction(queries)
        
        # Verify transaction was executed
        assert len(result) == 3
        # Should be 4 calls: BEGIN TRANSACTION + 3 user queries
        assert mock_connection.execute.call_count == 4
        mock_connection.commit.assert_called_once()
    
    async def test_execute_transaction_empty(self, provider, mock_connection):
        """Test execute_transaction with empty queries list."""
        # Set up provider state
        provider._connection = mock_connection
        provider._initialized = True
        
        result = await provider.execute_transaction([])
        assert result == []
    
    async def test_table_exists_success(self, provider, mock_connection):
        """Test successful table existence check."""
        # Set up provider state
        provider._connection = mock_connection
        provider._initialized = True
        
        # Mock execute_query to return a result (table exists)
        with patch('flowlib.providers.db.sqlite.provider.SQLiteDBProvider.execute_query', new_callable=AsyncMock) as mock_execute_query:
            mock_execute_query.return_value = [{"name": "users"}]
            
            # Check table existence
            result = await provider.table_exists("users")
            
            # Verify results
            assert result is True
            mock_execute_query.assert_called_once_with(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
                ("users",)
            )
    
    async def test_table_exists_not_found(self, provider, mock_connection):
        """Test table existence check when table doesn't exist."""
        # Set up provider state
        provider._connection = mock_connection
        provider._initialized = True
        
        # Mock execute_query to return empty list (table doesn't exist)
        with patch('flowlib.providers.db.sqlite.provider.SQLiteDBProvider.execute_query', new_callable=AsyncMock) as mock_execute_query:
            mock_execute_query.return_value = []
            
            # Check table existence
            result = await provider.table_exists("nonexistent")
            
            # Verify results
            assert result is False
            mock_execute_query.assert_called_once_with(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
                ("nonexistent",)
            )
    
    async def test_get_table_schema_success(self, provider, mock_connection):
        """Test successful table schema retrieval."""
        # Set up provider state
        provider._connection = mock_connection
        provider._initialized = True
        
        # Mock execute_query to return schema information
        schema_data = [
            {"cid": 0, "name": "id", "type": "INTEGER", "notnull": 0, "dflt_value": None, "pk": 1},
            {"cid": 1, "name": "name", "type": "TEXT", "notnull": 1, "dflt_value": None, "pk": 0}
        ]
        
        with patch('flowlib.providers.db.sqlite.provider.SQLiteDBProvider.execute_query', new_callable=AsyncMock) as mock_execute_query:
            mock_execute_query.return_value = schema_data
            
            # Get table schema
            result = await provider.get_table_schema("users")
            
            # Verify results
            assert len(result) == 2
            assert result[0]["name"] == "id"
            assert result[1]["name"] == "name"
            mock_execute_query.assert_called_once_with("PRAGMA table_info(users)")
    
    async def test_execute_script_success(self, provider, mock_connection):
        """Test successful script execution."""
        # Set up provider state
        provider._connection = mock_connection
        provider._initialized = True
        
        # Setup mocks
        mock_connection.executescript = AsyncMock()
        
        # Execute script
        script = "CREATE TABLE test (id INTEGER); INSERT INTO test VALUES (1);"
        await provider.execute_script(script)
        
        # Verify script was executed
        mock_connection.executescript.assert_called_once_with(script)
        mock_connection.commit.assert_called_once()
    
    async def test_execute_query_auto_initialize(self, provider, mock_connection):
        """Test that execute_query auto-initializes when needed."""
        # Set up provider state - not initialized
        provider._connection = None
        provider._initialized = False
        
        with patch('flowlib.providers.db.sqlite.provider.aiosqlite') as mock_aiosqlite:
            mock_aiosqlite.connect = AsyncMock(return_value=mock_connection)
            mock_cursor = AsyncMock()
            mock_cursor.fetchall.return_value = [(1,)]
            mock_cursor.description = [("count",)]
            mock_connection.execute.return_value = mock_cursor
            
            # Execute query (should auto-initialize)
            result = await provider.execute_query("SELECT 1 as count")
            
            # Verify initialization and execution
            mock_aiosqlite.connect.assert_called_once()
            assert result == [{"count": 1}]
    
    async def test_execute_update_auto_initialize(self, provider, mock_connection):
        """Test that execute_update auto-initializes when needed."""
        # Set up provider state - not initialized
        provider._connection = None
        provider._initialized = False
        
        with patch('flowlib.providers.db.sqlite.provider.aiosqlite') as mock_aiosqlite:
            mock_aiosqlite.connect = AsyncMock(return_value=mock_connection)
            mock_cursor = AsyncMock()
            mock_cursor.rowcount = 1
            mock_connection.execute.return_value = mock_cursor
            
            # Execute update (should auto-initialize)
            result = await provider.execute_update("UPDATE users SET name = ?", ("test",))
            
            # Verify initialization and execution
            mock_aiosqlite.connect.assert_called_once()
            assert result == 1
    
    async def test_error_handling_query(self, provider, mock_connection):
        """Test error handling in query execution."""
        # Set up provider state
        provider._connection = mock_connection
        provider._initialized = True
        
        # Setup mocks to raise an error
        mock_connection.execute.side_effect = Exception("Database error")
        
        # Execute query and expect ProviderError
        with pytest.raises(ProviderError) as exc_info:
            await provider.execute_query("SELECT * FROM users")
        
        assert "Failed to execute SQLite query" in str(exc_info.value)
    
    async def test_error_handling_update(self, provider, mock_connection):
        """Test error handling in update execution."""
        # Set up provider state
        provider._connection = mock_connection
        provider._initialized = True
        
        # Setup mocks to raise an error
        mock_connection.execute.side_effect = Exception("Database error")
        
        # Execute update and expect ProviderError
        with pytest.raises(ProviderError) as exc_info:
            await provider.execute_update("UPDATE users SET name = ?", ("test",))
        
        assert "Failed to execute SQLite update" in str(exc_info.value)
    
    async def test_multiple_queries_same_connection(self, provider, mock_connection):
        """Test that multiple queries use the same connection."""
        # Set up provider state
        provider._connection = mock_connection
        provider._initialized = True
        
        # Setup mocks
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = [(1,)]
        mock_cursor.description = [("result",)]
        mock_connection.execute.return_value = mock_cursor
        
        # Execute multiple queries
        await provider.execute_query("SELECT 1 as result")
        await provider.execute_query("SELECT 2 as result")
        
        # Verify same connection was used
        assert mock_connection.execute.call_count == 2
        assert provider._connection == mock_connection
    
    async def test_script_execution_with_commit(self, provider, mock_connection):
        """Test script execution includes commit."""
        # Set up provider state
        provider._connection = mock_connection
        provider._initialized = True
        
        # Setup mocks
        mock_connection.executescript = AsyncMock()
        
        # Execute script
        script = "CREATE TABLE test (id INTEGER);"
        await provider.execute_script(script)
        
        # Verify script execution and commit
        mock_connection.executescript.assert_called_once_with(script)
        mock_connection.commit.assert_called_once()
    
    async def test_transaction_rollback_on_error(self, provider, mock_connection):
        """Test transaction rollback when error occurs."""
        # Set up provider state
        provider._connection = mock_connection
        provider._initialized = True
        
        # Setup mocks - second query fails
        mock_cursor = AsyncMock()
        mock_connection.execute.side_effect = [mock_cursor, Exception("Query failed")]
        mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_connection.__aexit__ = AsyncMock(return_value=False)
        
        # Execute transaction that should fail
        queries = [
            ("INSERT INTO users VALUES (1, 'test')", None),
            ("INVALID SQL", None)
        ]
        
        with pytest.raises(ProviderError):
            await provider.execute_transaction(queries)
        
        # Verify rollback was called
        mock_connection.rollback.assert_called_once()
    
    async def test_table_schema_format(self, provider, mock_connection):
        """Test table schema returns proper format."""
        # Set up provider state
        provider._connection = mock_connection
        provider._initialized = True
        
        # Setup mocks
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = [
            (0, "id", "INTEGER", 1, None, 1),
            (1, "name", "TEXT", 0, None, 0),
            (2, "age", "INTEGER", 0, None, 0)
        ]
        mock_connection.execute.return_value = mock_cursor
        
        # Get table schema
        result = await provider.get_table_schema("users")
        
        # Verify schema format
        assert len(result) == 3
        # Schema should return list of column info dictionaries
        for column in result:
            assert isinstance(column, dict)
    
    async def test_database_file_creation(self, provider):
        """Test that database file is created when missing."""
        # Create temporary directory for test
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "new_test.db")
            settings = SQLiteProviderSettings(database_path=db_path)
            test_provider = SQLiteDBProvider(settings=settings)
            
            # Ensure file doesn't exist initially
            assert not os.path.exists(db_path)
            
            with patch('flowlib.providers.db.sqlite.provider.aiosqlite') as mock_aiosqlite:
                mock_connection = AsyncMock()
                mock_aiosqlite.connect = AsyncMock(return_value=mock_connection)
                
                # Initialize provider
                await test_provider.initialize()
                
                # Verify connect was called with correct path
                mock_aiosqlite.connect.assert_called_once()
    
    
    
    
    
    
    


@pytest.mark.skipif(not AIOSQLITE_AVAILABLE, reason="aiosqlite package not available")
@pytest.mark.integration
class TestSQLiteDBProviderIntegration:
    """Integration tests for SQLite provider.
    
    These tests require aiosqlite to be installed.
    """
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database file path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        yield path
        # Cleanup
        try:
            os.unlink(path)
        except OSError:
            pass
    
    @pytest.fixture
    def settings(self, temp_db_path):
        """Create integration test settings."""
        return SQLiteProviderSettings(
            database_path=temp_db_path,
            journal_mode="WAL",
            timeout=10.0,
            pool_size=2
        )
    
    @pytest.fixture
    async def provider(self, settings):
        """Create and initialize provider for integration tests."""
        provider = SQLiteDBProvider(name="integration_sqlite", settings=settings)
        
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
        # Create test table
        await provider.execute("""
            CREATE TABLE IF NOT EXISTS test_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
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
        # Create test table
        await provider.execute("""
            CREATE TABLE IF NOT EXISTS batch_test (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
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
        assert len(results) == 1  # SQLite executemany returns single result
        assert results[0]["affected_rows"] == 3
        
        # Verify data
        items = await provider.execute("SELECT * FROM batch_test ORDER BY id")
        assert len(items) == 3
        assert items[0]["name"] == "item1"
        assert items[1]["value"] == 20
        
        # Clean up
        await provider.execute("DROP TABLE IF EXISTS batch_test")
    
    async def test_transactions(self, provider):
        """Test transaction handling."""
        # Create test table
        await provider.execute("""
            CREATE TABLE IF NOT EXISTS transaction_test (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                amount INTEGER NOT NULL
            )
        """)
        
        # Test successful transaction
        transaction = await provider.begin_transaction()
        try:
            # Execute operations within transaction
            cursor = await transaction.cursor()
            await cursor.execute("INSERT INTO transaction_test (amount) VALUES (?)", (100,))
            await cursor.execute("INSERT INTO transaction_test (amount) VALUES (?)", (200,))
            await cursor.close()
            
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
            cursor = await transaction.cursor()
            await cursor.execute("INSERT INTO transaction_test (amount) VALUES (?)", (300,))
            await cursor.close()
            # Simulate error
            raise Exception("Simulated error")
        except Exception:
            await provider.rollback_transaction(transaction)
        
        # Verify rollback worked
        result = await provider.execute("SELECT COUNT(*) as count FROM transaction_test")
        assert result[0]["count"] == 2  # Should still be 2
        
        # Clean up
        await provider.execute("DROP TABLE IF EXISTS transaction_test")
    
    async def test_concurrent_operations(self, provider):
        """Test concurrent operations with single connection."""
        # Test that multiple operations can be executed concurrently
        tasks = []
        for i in range(5):
            task = provider.execute("SELECT ? as value", {"value": i})
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Verify all operations completed
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result[0]["value"] == i
    
    async def test_connection_health(self, provider):
        """Test connection and health monitoring."""
        # Check connection
        assert await provider.check_connection() is True
        
        # Get health information
        health = await provider.get_health()
        assert health.status == "healthy"
        assert health.connection_active is True
        assert health.version is not None
        assert health.database is not None
        assert health.pool is not None
        
        # Verify database file exists
        assert os.path.exists(health.database.path)
    
    async def test_concurrent_access(self, provider):
        """Test concurrent database access."""
        # Create test table
        await provider.execute("""
            CREATE TABLE IF NOT EXISTS concurrent_test (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER,
                value TEXT
            )
        """)
        
        # Define concurrent operation
        async def insert_data(thread_id: int):
            for i in range(10):
                await provider.execute(
                    "INSERT INTO concurrent_test (thread_id, value) VALUES (:thread_id, :value)",
                    {"thread_id": thread_id, "value": f"value_{i}"}
                )
        
        # Run concurrent operations
        tasks = [insert_data(i) for i in range(3)]
        await asyncio.gather(*tasks)
        
        # Verify all data was inserted
        result = await provider.execute("SELECT COUNT(*) as count FROM concurrent_test")
        assert result[0]["count"] == 30  # 3 threads * 10 inserts each
        
        # Clean up
        await provider.execute("DROP TABLE IF EXISTS concurrent_test")