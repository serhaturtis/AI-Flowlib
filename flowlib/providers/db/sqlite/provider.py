"""SQLite database provider implementation.

This module provides a concrete implementation of the DBProvider
for SQLite database using aiosqlite.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import json

from pydantic import Field

from flowlib.core.errors.errors import ProviderError, ErrorContext
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.db.base import DBProvider, DatabaseHealthInfo, DatabaseInfo, PoolInfo
from flowlib.providers.core.base import ProviderSettings
from flowlib.providers.core.base import Provider
from flowlib.providers.core.decorators import provider
# Removed ProviderType import - using config-driven provider access

logger = logging.getLogger(__name__)

try:
    import aiosqlite
    import sqlite3
except ImportError:
    aiosqlite = None
    sqlite3 = None
    logger.warning("aiosqlite package not found. Install with 'pip install aiosqlite'")


class SQLiteProviderSettings(ProviderSettings):
    """SQLite provider settings - direct inheritance, only SQLite-specific fields.
    
    SQLite requires:
    1. Database file path
    2. SQLite-specific configuration (journal mode, isolation)
    3. File creation and timeout settings
    
    Note: SQLite is file-based, no host/port/authentication needed.
    """
    
    # SQLite file settings
    database_path: str = Field(default="./database.db", description="Path to SQLite database file (e.g., './app.db', '/data/database.sqlite')")
    create_if_missing: bool = Field(default=True, description="Create database file if it doesn't exist")
    
    # SQLite performance settings
    journal_mode: str = Field(default="WAL", description="SQLite journal mode (WAL for better concurrency)")
    isolation_level: Optional[str] = Field(default=None, description="SQLite isolation level (None = autocommit mode)")
    timeout: float = Field(default=5.0, description="Connection timeout in seconds")
    detect_types: int = Field(default=0, description="SQLite type detection (can use sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)")
    
    # Additional connection arguments
    connect_args: Dict[str, Any] = Field(default_factory=dict, description="Additional connection arguments")


@provider(provider_type="db", name="sqlite", settings_class=SQLiteProviderSettings)
class SQLiteDBProvider(DBProvider):
    """SQLite implementation of the DBProvider.
    
    This provider implements database operations using aiosqlite,
    an efficient asynchronous SQLite driver.
    """
    
    def __init__(self, name: str = "sqlite", settings: Optional[SQLiteProviderSettings] = None):
        """Initialize SQLite provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        super().__init__(name=name, settings=settings)
        self._settings = settings or SQLiteProviderSettings(database_path=":memory:")
        self._connection = None
        
    async def initialize(self) -> None:
        """Initialize the SQLite provider."""
        await self._initialize()
        await super().initialize()
        
    async def shutdown(self) -> None:
        """Shutdown the SQLite provider."""
        shutdown_success = True
        try:
            await self._shutdown()
        except Exception:
            shutdown_success = False
            # Don't raise, let shutdown complete gracefully
        
        # Only call super().shutdown() if our shutdown was successful
        # This preserves the initialized state if shutdown fails
        if shutdown_success:
            await super().shutdown()
        
    async def _initialize(self) -> None:
        """Initialize SQLite connection.
        
        Raises:
            ProviderError: If initialization fails
        """
        try:
            # Check if aiosqlite is installed
            if aiosqlite is None:
                raise ProviderError(
                    message="aiosqlite package not installed. Install with 'pip install aiosqlite'",
                    context=ErrorContext.create(
                        flow_name="sqlite_provider",
                        error_type="DependencyError",
                        error_location="_initialize",
                        component=self.name,
                        operation="check_aiosqlite_dependency"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="db",
                        operation="initialize",
                        retry_count=0
                    )
                )
            
            # Check if database file exists
            is_memory_db = self._settings.database_path == ":memory:"
            if not is_memory_db and not os.path.exists(self._settings.database_path):
                if self._settings.create_if_missing:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(self._settings.database_path), exist_ok=True)
                    logger.info(f"Created directory for SQLite database: {os.path.dirname(self._settings.database_path)}")
                else:
                    raise ProviderError(
                        message=f"SQLite database file does not exist: {self._settings.database_path}",
                        context=ErrorContext.create(
                            flow_name="sqlite_provider",
                            error_type="FileNotFoundError",
                            error_location="_initialize",
                            component=self.name,
                            operation="check_database_file"
                        ),
                        provider_context=ProviderErrorContext(
                            provider_name=self.name,
                            provider_type="db",
                            operation="initialize",
                            retry_count=0
                        )
                    )
            
            # Connect to database
            self._connection = await aiosqlite.connect(
                database=self._settings.database_path,
                timeout=self._settings.timeout,
                isolation_level=self._settings.isolation_level,
                detect_types=self._settings.detect_types,
                **self._settings.connect_args
            )
            
            # Enable foreign keys
            await self._connection.execute("PRAGMA foreign_keys = ON")
            
            # Set journal mode
            await self._connection.execute(f"PRAGMA journal_mode = {self._settings.journal_mode}")
            
            # Configure JSON serialization/deserialization
            sqlite3.register_adapter(dict, json.dumps)
            sqlite3.register_adapter(list, json.dumps)
            sqlite3.register_converter("JSON", json.loads)
            
            logger.info(f"Connected to SQLite database: {self._settings.database_path}")
            
        except Exception as e:
            self._connection = None
            raise ProviderError(
                message=f"Failed to connect to SQLite database: {str(e)}",
                context=ErrorContext.create(
                    flow_name="sqlite_provider",
                    error_type="ConnectionError",
                    error_location="_initialize",
                    component=self.name,
                    operation="connect_to_database"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="initialize",
                    retry_count=0
                ),
                cause=e
            )
    
    async def _shutdown(self) -> None:
        """Close SQLite connection."""
        if self._connection:
            try:
                await self._connection.close()
                logger.info(f"Closed SQLite connection: {self._settings.database_path}")
            except Exception as e:
                logger.warning(f"Error closing SQLite connection: {e}")
                # Re-raise so shutdown method can handle it
                raise
            finally:
                self._connection = None
    
    async def execute_query(self, query: str, params: Optional[Union[tuple, dict]] = None) -> List[Dict[str, Any]]:
        """Execute a SQL query.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            List of rows as dictionaries
            
        Raises:
            ProviderError: If query execution fails
        """
        if not self._connection:
            await self.initialize()
            
        try:
            # Execute query
            cursor = await self._connection.execute(query, params or ())
            
            # Get column names
            columns = [column[0] for column in cursor.description] if cursor.description else []
            
            # Fetch all rows
            rows = await cursor.fetchall()
            
            # Convert rows to dictionaries
            results = []
            for row in rows:
                result = {}
                for i, column in enumerate(columns):
                    value = row[i]
                    # Handle SQLite-specific types
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except UnicodeDecodeError:
                            pass  # Keep as bytes if not valid UTF-8
                    result[column] = value
                results.append(result)
                
            return results
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute SQLite query: {str(e)}",
                context=ErrorContext.create(
                    flow_name="sqlite_provider",
                    error_type="QueryExecutionError",
                    error_location="execute_query",
                    component=self.name,
                    operation="execute_sql_query"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="execute_query",
                    retry_count=0
                ),
                cause=e
            )
    
    async def execute_update(self, query: str, params: Optional[Union[tuple, dict]] = None) -> int:
        """Execute a SQL update.
        
        Args:
            query: SQL query (INSERT, UPDATE, DELETE)
            params: Query parameters
            
        Returns:
            Number of rows affected
            
        Raises:
            ProviderError: If update execution fails
        """
        if not self._connection:
            await self.initialize()
            
        try:
            # Execute query
            cursor = await self._connection.execute(query, params or ())
            
            # Commit changes
            await self._connection.commit()
            
            # Return number of rows affected
            return cursor.rowcount
            
        except Exception as e:
            # Rollback transaction
            await self._connection.rollback()
            
            raise ProviderError(
                message=f"Failed to execute SQLite update: {str(e)}",
                context=ErrorContext.create(
                    flow_name="sqlite_provider",
                    error_type="UpdateExecutionError",
                    error_location="execute_update",
                    component=self.name,
                    operation="execute_sql_update"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="execute_update",
                    retry_count=0
                ),
                cause=e
            )
    
    async def execute_script(self, script: str) -> None:
        """Execute a SQL script.
        
        Args:
            script: SQL script
            
        Raises:
            ProviderError: If script execution fails
        """
        if not self._connection:
            await self.initialize()
            
        try:
            # Execute script
            await self._connection.executescript(script)
            
            # Commit changes
            await self._connection.commit()
            
        except Exception as e:
            # Rollback transaction
            await self._connection.rollback()
            
            raise ProviderError(
                message=f"Failed to execute SQLite script: {str(e)}",
                context=ErrorContext.create(
                    flow_name="sqlite_provider",
                    error_type="ScriptExecutionError",
                    error_location="execute_script",
                    component=self.name,
                    operation="execute_sql_script"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="execute_script",
                    retry_count=0
                ),
                cause=e
            )
    
    async def execute_transaction(self, queries: List[Tuple[str, Optional[Union[tuple, dict]]]]) -> List[Any]:
        """Execute queries in a transaction.
        
        Args:
            queries: List of (query, params) tuples
            
        Returns:
            List of results
            
        Raises:
            ProviderError: If transaction fails
        """
        if not self._connection:
            await self.initialize()
            
        try:
            # Start transaction
            await self._connection.execute("BEGIN TRANSACTION")
            
            results = []
            for query, params in queries:
                # Execute query
                cursor = await self._connection.execute(query, params or ())
                
                # Check if query returns rows
                if cursor.description:
                    # Get column names
                    columns = [column[0] for column in cursor.description]
                    
                    # Fetch all rows
                    rows = await cursor.fetchall()
                    
                    # Convert rows to dictionaries
                    query_results = []
                    for row in rows:
                        result = {}
                        for i, column in enumerate(columns):
                            result[column] = row[i]
                        query_results.append(result)
                    
                    results.append(query_results)
                else:
                    # For INSERT, UPDATE, DELETE queries
                    results.append(cursor.rowcount)
                
            # Commit transaction
            await self._connection.commit()
            
            return results
            
        except Exception as e:
            # Rollback transaction
            await self._connection.rollback()
            
            raise ProviderError(
                message=f"Failed to execute SQLite transaction: {str(e)}",
                context=ErrorContext.create(
                    flow_name="sqlite_provider",
                    error_type="TransactionExecutionError",
                    error_location="execute_transaction",
                    component=self.name,
                    operation="execute_sql_transaction"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="execute_transaction",
                    retry_count=0
                ),
                cause=e
            )
    
    async def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema for a table.
        
        Args:
            table_name: Table name
            
        Returns:
            List of column definitions
            
        Raises:
            ProviderError: If schema retrieval fails
        """
        if not self._connection:
            await self.initialize()
            
        try:
            # Get table schema
            query = f"PRAGMA table_info({table_name})"
            return await self.execute_query(query)
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to get SQLite table schema: {str(e)}",
                context=ErrorContext.create(
                    flow_name="sqlite_provider",
                    error_type="SchemaRetrievalError",
                    error_location="get_table_schema",
                    component=self.name,
                    operation="get_table_schema"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="get_table_schema",
                    retry_count=0
                ),
                cause=e
            )
    
    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists.
        
        Args:
            table_name: Table name
            
        Returns:
            True if table exists
            
        Raises:
            ProviderError: If check fails
        """
        if not self._connection:
            await self.initialize()
            
        try:
            # Check if table exists
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
            result = await self.execute_query(query, (table_name,))
            
            return len(result) > 0
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to check if SQLite table exists: {str(e)}",
                context=ErrorContext.create(
                    flow_name="sqlite_provider",
                    error_type="TableExistenceCheckError",
                    error_location="table_exists",
                    component=self.name,
                    operation="check_table_existence"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="table_exists",
                    retry_count=0
                ),
                cause=e
            )
    
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a database query.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Query results (list for SELECT, row count for UPDATE/INSERT/DELETE)
            
        Raises:
            ProviderError: If query execution fails
        """
        if not self._connection:
            raise ProviderError(
                message="SQLite connection not initialized",
                context=ErrorContext.create(
                    flow_name="sqlite_provider",
                    error_type="ConnectionError",
                    error_location="execute",
                    component=self.name,
                    operation="execute"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="execute",
                    retry_count=0
                )
            )
        
        try:
            # Handle parameter conversion based on query style
            param_values = None
            if params:
                if isinstance(params, dict):
                    # Check if query uses named parameters (:name) or positional (?)
                    if ':' in query:
                        # Query uses named parameters, keep as dict
                        param_values = params
                    else:
                        # Query uses positional parameters, convert dict to tuple
                        # This assumes the dict keys are in the order expected by the query
                        param_values = tuple(params.values())
                else:
                    param_values = params
            
            # Determine if this is a SELECT query or not
            query_lower = query.strip().lower()
            if query_lower.startswith('select') or query_lower.startswith('with'):
                # For SELECT queries, return all rows as dicts
                return await self.execute_query(query, param_values)
            else:
                # For INSERT/UPDATE/DELETE, return formatted result
                affected_rows = await self.execute_update(query, param_values)
                return [{"affected_rows": affected_rows}]
                
        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute SQLite query: {str(e)}",
                context=ErrorContext.create(
                    flow_name="sqlite_provider",
                    error_type="QueryExecutionError",
                    error_location="execute",
                    component=self.name,
                    operation="execute"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="execute",
                    retry_count=0
                ),
                cause=e
            )
    
    async def check_connection(self) -> bool:
        """Check if the database connection is healthy.
        
        Returns:
            True if connection is healthy
            
        Raises:
            ProviderError: If connection check fails
        """
        if not self._connection:
            return False
            
        try:
            # Execute a simple query to test the connection
            async with self._connection.cursor() as cursor:
                await cursor.execute("SELECT 1")
                result = await cursor.fetchone()
                return result is not None
                
        except Exception as e:
            logger.warning(f"SQLite connection check failed: {e}")
            return False
    
    async def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> List[Any]:
        """Execute a batch of database queries.
        
        Args:
            query: SQL query to execute
            params_list: List of query parameters
            
        Returns:
            List with single result containing total affected rows
            
        Raises:
            ProviderError: If query execution fails
        """
        if not self._connection:
            raise ProviderError(
                message="SQLite connection not initialized",
                context=ErrorContext.create(
                    flow_name="sqlite_provider",
                    error_type="ConnectionError",
                    error_location="execute_many",
                    component=self.name,
                    operation="execute_many"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="execute_many",
                    retry_count=0
                )
            )
        
        try:
            # Use executemany for batch operations with named parameters
            cursor = await self._connection.executemany(query, params_list)
            
            await self._connection.commit()
            
            # Return total affected rows as single result
            total_affected = cursor.rowcount
            return [{"affected_rows": total_affected}]
            
        except Exception as e:
            # Rollback on error
            await self._connection.rollback()
            
            raise ProviderError(
                message=f"Failed to execute SQLite batch queries: {str(e)}",
                context=ErrorContext.create(
                    flow_name="sqlite_provider",
                    error_type="BatchQueryExecutionError",
                    error_location="execute_many",
                    component=self.name,
                    operation="execute_many"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="execute_many",
                    retry_count=0
                ),
                cause=e
            )
    
    async def begin_transaction(self):
        """Begin a database transaction.
        
        Returns:
            Transaction object (connection for SQLite)
            
        Raises:
            ProviderError: If transaction start fails
        """
        if not self._connection:
            raise ProviderError(
                message="SQLite connection not initialized",
                context=ErrorContext.create(
                    flow_name="sqlite_provider",
                    error_type="ConnectionError",
                    error_location="begin_transaction",
                    component=self.name,
                    operation="begin_transaction"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="begin_transaction",
                    retry_count=0
                )
            )
        
        try:
            # SQLite uses connection-level transactions
            await self._connection.execute("BEGIN")
            return self._connection
        except Exception as e:
            raise ProviderError(
                message=f"Failed to begin SQLite transaction: {str(e)}",
                context=ErrorContext.create(
                    flow_name="sqlite_provider",
                    error_type="TransactionError",
                    error_location="begin_transaction",
                    component=self.name,
                    operation="begin_transaction"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="begin_transaction",
                    retry_count=0
                ),
                cause=e
            )
    
    async def commit_transaction(self, transaction: Any) -> bool:
        """Commit a database transaction.
        
        Args:
            transaction: Transaction object (SQLite connection)
            
        Returns:
            True if transaction was committed successfully
            
        Raises:
            ProviderError: If transaction commit fails
        """
        try:
            await transaction.commit()
            return True
        except Exception as e:
            raise ProviderError(
                message=f"Failed to commit SQLite transaction: {str(e)}",
                context=ErrorContext.create(
                    flow_name="sqlite_provider",
                    error_type="TransactionCommitError",
                    error_location="commit_transaction",
                    component=self.name,
                    operation="commit_transaction"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="commit_transaction",
                    retry_count=0
                ),
                cause=e
            )
    
    async def rollback_transaction(self, transaction: Any) -> bool:
        """Rollback a database transaction.
        
        Args:
            transaction: Transaction object (SQLite connection)
            
        Returns:
            True if transaction was rolled back successfully
            
        Raises:
            ProviderError: If transaction rollback fails
        """
        try:
            await transaction.rollback()
            return True
        except Exception as e:
            raise ProviderError(
                message=f"Failed to rollback SQLite transaction: {str(e)}",
                context=ErrorContext.create(
                    flow_name="sqlite_provider",
                    error_type="TransactionRollbackError",
                    error_location="rollback_transaction",
                    component=self.name,
                    operation="rollback_transaction"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="rollback_transaction",
                    retry_count=0
                ),
                cause=e
            )
    
    async def get_health(self) -> DatabaseHealthInfo:
        """Get database health information.
        
        Returns:
            Structured health information
            
        Raises:
            ProviderError: If health check fails
        """
        try:
            is_connected = await self.check_connection()
            
            # Get database version if connected
            version = None
            additional_info = {}
            
            if is_connected:
                try:
                    async with self._connection.cursor() as cursor:
                        # Get SQLite version
                        await cursor.execute("SELECT sqlite_version()")
                        version_result = await cursor.fetchone()
                        version = version_result[0] if version_result else "unknown"
                        
                        # Get database file size if it's a file
                        if self._settings.database_path != ":memory:":
                            db_path = Path(self._settings.database_path)
                            if db_path.exists():
                                additional_info["database_size_bytes"] = db_path.stat().st_size
                                
                        additional_info["journal_mode"] = self._settings.journal_mode
                        additional_info["connection_initialized"] = self._connection is not None
                        
                except Exception as e:
                    additional_info["version_check_error"] = str(e)
            
            return DatabaseHealthInfo(
                status="healthy" if is_connected else "unhealthy",
                connected=is_connected,
                connection_active=is_connected,
                database=DatabaseInfo(
                    path=str(self._settings.database_path),
                    name="SQLite"
                ),
                pool=PoolInfo(
                    active_connections=1 if self._connection else 0,
                    pool_size=1  # SQLite uses single connection
                ),
                version=version,
                additional_info=additional_info
            )
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to get SQLite health info: {str(e)}",
                context=ErrorContext.create(
                    flow_name="sqlite_provider",
                    error_type="HealthCheckError",
                    error_location="get_health",
                    component=self.name,
                    operation="get_health"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="get_health",
                    retry_count=0
                ),
                cause=e
            ) 