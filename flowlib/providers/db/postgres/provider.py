"""PostgreSQL database provider implementation.

This module provides a concrete implementation of the DBProvider
for PostgreSQL database using asyncpg.
"""

import asyncio
import logging
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from pydantic import Field

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.decorators import provider
from flowlib.providers.db.base import (
    DatabaseHealthInfo,
    DatabaseInfo,
    DBProvider,
    DBProviderSettings,
    PoolInfo,
)

# Removed ProviderType import - using config-driven provider access

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import asyncpg  # type: ignore[import-untyped]

    ASYNCPG_AVAILABLE = True
else:
    try:
        import asyncpg  # type: ignore[import-untyped]

        ASYNCPG_AVAILABLE = True
    except ImportError:
        ASYNCPG_AVAILABLE = False
        logger.warning("asyncpg package not found. Install with 'pip install asyncpg'")

        class _AsyncPGModule:
            pass

        asyncpg = _AsyncPGModule()  # type: ignore

try:
    from ..base import DBProvider as BaseDBProvider  # noqa: F401 - Alias to avoid confusion
except ImportError:
    logger.warning("Provider not found. Install with 'from ..base import Provider'")


class PostgreSQLProviderSettings(DBProviderSettings):
    """Settings for PostgreSQL provider - direct inheritance, only PostgreSQL-specific fields.

    PostgreSQL is a traditional database requiring:
    1. Host/port connection details
    2. Database name and credentials
    3. PostgreSQL-specific configuration

    This follows Interface Segregation - only fields PostgreSQL actually needs.
    """

    # PostgreSQL connection settings
    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, description="PostgreSQL port")
    database: str = Field(
        default="postgres", description="Database name (e.g., 'myapp', 'flowlib_db')"
    )
    username: str | None = Field(default=None, description="Database username")
    password: str | None = Field(default=None, description="Database password")

    # PostgreSQL-specific settings
    db_schema: str = Field(default="public", description="Database schema to use")
    ssl_mode: str = Field(
        default="prefer",
        description="SSL mode: disable, allow, prefer, require, verify-ca, verify-full",
    )
    statement_timeout: int | None = Field(default=None, description="Statement timeout in ms")

    # Connection pool settings
    min_connections: int = Field(default=1, description="Minimum connections in pool")
    max_connections: int = Field(default=10, description="Maximum connections in pool")

    # Additional connection arguments
    connect_args: dict[str, Any] = Field(
        default_factory=dict, description="Additional asyncpg connection arguments"
    )


@provider(provider_type="db", name="postgresql", settings_class=PostgreSQLProviderSettings)
class PostgreSQLProvider(DBProvider[PostgreSQLProviderSettings]):
    """PostgreSQL implementation of the DBProvider.

    This provider implements database operations using asyncpg,
    an efficient asynchronous PostgreSQL driver.
    """

    def __init__(
        self, name: str = "postgres", settings: PostgreSQLProviderSettings | None = None
    ):
        """Initialize PostgreSQL provider.

        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        # Create settings first to avoid issues with _default_settings() method
        settings = settings or PostgreSQLProviderSettings(
            database="test", username="test", password="test"
        )

        # Pass explicit settings to parent class
        super().__init__(name=name, settings=settings)

        # Store settings for local use
        self._settings = settings
        self._pool = None
        self._json_encoders = {
            # Custom JSON encoders for PostgreSQL
            datetime: lambda dt: dt.isoformat(),
            date: lambda d: d.isoformat(),
        }

    async def initialize(self) -> None:
        """Initialize the PostgreSQL connection pool."""
        if self._initialized:
            return

        try:
            # Check if asyncpg is installed
            if not ASYNCPG_AVAILABLE:
                raise ProviderError(
                    message="asyncpg package not installed. Install with 'pip install asyncpg'",
                    context=ErrorContext.create(
                        flow_name="postgres_provider",
                        error_type="DependencyError",
                        error_location="initialize",
                        component=self.name,
                        operation="check_asyncpg_dependency",
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="db",
                        operation="initialize",
                        retry_count=0,
                    ),
                )

            # Prepare DSN (Data Source Name) for connection
            dsn = self._create_connection_string()

            # Prepare SSL context if needed
            ssl = None
            if self._settings.ssl_mode not in ["disable", "allow"]:
                import ssl as ssl_module

                ssl_context = ssl_module.create_default_context()
                ssl = ssl_context

            # Create connection pool
            self._pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=self._settings.min_connections,
                max_size=self._settings.max_connections,
                max_inactive_connection_lifetime=300.0,  # 5 minutes
                timeout=30.0,  # Default connection timeout
                command_timeout=60.0,  # Default command timeout
                statement_cache_size=100,
                max_cached_statement_lifetime=300.0,  # 5 minutes
                ssl=ssl,
                server_settings={
                    "search_path": self._settings.db_schema,
                    **(
                        {"statement_timeout": str(self._settings.statement_timeout)}
                        if self._settings.statement_timeout
                        else {}
                    ),
                },
                **self._settings.connect_args,
            )

            # Register custom type encoders and decoders
            self._register_type_codecs()

            self._initialized = True
            logger.debug(f"{self.name} provider initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize {self.name} provider: {str(e)}")
            raise ProviderError(
                message=f"Failed to initialize PostgreSQL provider: {str(e)}",
                context=ErrorContext.create(
                    flow_name="postgres_provider",
                    error_type="InitializationError",
                    error_location="initialize",
                    component=self.name,
                    operation="create_connection_pool",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="initialize",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def shutdown(self) -> None:
        """Close PostgreSQL connection pool and release resources."""
        try:
            # Close the connection pool if it exists
            if self._pool is not None:
                await self._pool.close()  # type: ignore[unreachable]
            self._pool = None
            self._initialized = False
            logger.debug(f"{self.name} provider shut down successfully")

        except Exception as e:
            logger.error(f"Error during {self.name} provider shutdown: {str(e)}")
            raise ProviderError(
                message=f"Failed to shut down PostgreSQL provider: {str(e)}",
                context=ErrorContext.create(
                    flow_name="postgres_provider",
                    error_type="ShutdownError",
                    error_location="shutdown",
                    component=self.name,
                    operation="close_connection_pool",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name, provider_type="db", operation="shutdown", retry_count=0
                ),
                cause=e,
) from e

    async def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a database query.

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            List of dictionaries representing rows

        Raises:
            ProviderError: If query execution fails
        """
        if not self._initialized or self._pool is None:
            error_msg = (
                "Provider not initialized"
                if not self._initialized
                else "Connection pool not available"
            )
            operation = "check_initialization" if not self._initialized else "check_pool"
            raise ProviderError(
                message=error_msg,
                context=ErrorContext.create(
                    flow_name="postgres_provider",
                    error_type="StateError",
                    error_location="execute",
                    component=self.name,
                    operation=operation,
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name, provider_type="db", operation="execute", retry_count=0
                ),
            )

        # Convert named parameters to positional if present
        positional_query, positional_params = self._convert_params(query, params)  # type: ignore[unreachable]

        try:
            # Execute the query in the pool
            async with self._pool.acquire() as conn:
                # Check if it's a SELECT query or other type (INSERT, UPDATE, etc.)
                is_select = positional_query.strip().lower().startswith("select")

                if is_select:
                    # Execute SELECT and fetch results
                    records = await conn.fetch(positional_query, *positional_params)
                    # Convert to dict for easier consumption
                    return [dict(record) for record in records]
                else:
                    # Execute non-SELECT query
                    result = await conn.execute(positional_query, *positional_params)

                    # Parse result string
                    command, *rest = result.split()
                    if command in ("INSERT", "UPDATE", "DELETE"):
                        # Return affected rows count
                        if command == "INSERT":
                            # INSERT format: "INSERT 0 1" -> count is rest[1]
                            count = int(rest[1]) if len(rest) > 1 else 0
                        else:
                            # UPDATE/DELETE format: "UPDATE 1" or "DELETE 1" -> count is rest[0]
                            count = int(rest[0]) if len(rest) > 0 else 0
                        return [{"affected_rows": count}]
                    else:
                        # Return raw result for other commands
                        return [{"result": result}]

        except Exception as e:
            # Retry on connection errors if enabled (use provider base retry settings)
            if self._should_retry(e) and self._settings.max_retries > 0:
                return await self._retry_execute(query, params)

            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to execute query: {str(e)}",
                context=ErrorContext.create(
                    flow_name="postgres_provider",
                    error_type="QueryExecutionError",
                    error_location="execute",
                    component=self.name,
                    operation="execute_sql_query",
                    query=query[:100] + "..." if len(query) > 100 else query,
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name, provider_type="db", operation="execute", retry_count=0
                ),
                cause=e,
) from e

    async def execute_many(self, query: str, params_list: list[dict[str, Any]]) -> list[Any]:
        """Execute a batch of database queries.

        Args:
            query: SQL query to execute
            params_list: List of query parameters

        Returns:
            List of query results

        Raises:
            ProviderError: If query execution fails
        """
        if not self._initialized or not self._pool:
            raise ProviderError(
                message="Provider not initialized",
                context=ErrorContext.create(
                    flow_name="postgres_provider",
                    error_type="StateError",
                    error_location="execute_many",
                    component=self.name,
                    operation="check_initialization",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="execute_many",
                    retry_count=0,
                ),
            )

        if not params_list:  # type: ignore[unreachable]
            return []

        try:
            # Prepare positional parameters for each set of params
            all_positional_params = []
            first_param_set = params_list[0]

            # Extract parameter names from the first param set
            param_names = list(first_param_set.keys())

            # Convert query to use positional parameters
            positional_query = query
            for i, name in enumerate(param_names):
                positional_query = positional_query.replace(f":{name}", f"${i + 1}")
                positional_query = positional_query.replace(f"@{name}", f"${i + 1}")

            # Prepare positional parameters for each set
            for params in params_list:
                positional_params = []
                for name in param_names:
                    if name not in params:
                        raise ValueError(f"Parameter '{name}' missing from parameter set: {params}")
                    positional_params.append(params[name])
                all_positional_params.append(positional_params)

            # Execute batch with connection from pool
            async with self._pool.acquire() as conn:
                # Start a transaction
                async with conn.transaction():
                    # Execute each query in the batch
                    results = []
                    for params in all_positional_params:
                        # Execute the query
                        result = await conn.execute(positional_query, *params)
                        results.append(result)

                    return results

        except Exception as e:
            # Retry on connection errors if enabled (use provider base retry settings)
            if self._should_retry(e) and self._settings.max_retries > 0:
                return await self._retry_execute_many(query, params_list)

            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to execute batch query: {str(e)}",
                context=ErrorContext.create(
                    flow_name="postgres_provider",
                    error_type="BatchQueryExecutionError",
                    error_location="execute_many",
                    component=self.name,
                    operation="execute_batch_sql_query",
                    query=query[:100] + "..." if len(query) > 100 else query,
                    batch_size=len(params_list),
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="execute_many",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def begin_transaction(self) -> Any:
        """Begin a database transaction.

        Returns:
            Transaction wrapper object

        Raises:
            ProviderError: If transaction start fails
        """
        if not self._initialized or not self._pool:
            raise ProviderError(
                message="Provider not initialized",
                context=ErrorContext.create(
                    flow_name="postgres_provider",
                    error_type="StateError",
                    error_location="begin_transaction",
                    component=self.name,
                    operation="check_initialization",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="begin_transaction",
                    retry_count=0,
                ),
            )

        try:  # type: ignore[unreachable]
            # Acquire connection from pool
            conn = await self._pool.acquire()

            # Start transaction
            transaction = conn.transaction()
            await transaction.start()

            # Return a wrapper object that contains both the connection and transaction
            class TransactionWrapper:
                def __init__(self, connection, transaction):
                    self.connection = connection
                    self.transaction = transaction

                async def execute(self, query: str, *args):
                    """Execute a query within the transaction."""
                    return await self.connection.execute(query, *args)

                async def fetch(self, query: str, *args):
                    """Fetch results from a query within the transaction."""
                    return await self.connection.fetch(query, *args)

            return TransactionWrapper(conn, transaction)

        except Exception as e:
            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to begin transaction: {str(e)}",
                context=ErrorContext.create(
                    flow_name="postgres_provider",
                    error_type="TransactionError",
                    error_location="begin_transaction",
                    component=self.name,
                    operation="start_database_transaction",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="begin_transaction",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def commit_transaction(self, transaction: Any) -> bool:
        """Commit a database transaction.

        Args:
            transaction: Transaction wrapper from begin_transaction()

        Returns:
            True if transaction was committed successfully

        Raises:
            ProviderError: If transaction commit fails
        """
        if (
            not transaction
            or not hasattr(transaction, "transaction")
            or not hasattr(transaction, "connection")
        ):
            raise ProviderError(
                message="Invalid transaction object",
                context=ErrorContext.create(
                    flow_name="postgres_provider",
                    error_type="ValidationError",
                    error_location="commit_transaction",
                    component=self.name,
                    operation="validate_transaction_object",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="commit_transaction",
                    retry_count=0,
                ),
            )

        try:
            # Commit the transaction
            await transaction.transaction.commit()

            # Release the connection back to the pool
            if self._pool is not None:
                await self._pool.release(transaction.connection)  # type: ignore[unreachable]  # type: ignore[unreachable]

            return True

        except Exception as e:
            # Try to release the connection
            try:
                if self._pool is not None:
                    await self._pool.release(transaction.connection)  # type: ignore[unreachable]
            except Exception:
                pass

            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to commit transaction: {str(e)}",
                context=ErrorContext.create(
                    flow_name="postgres_provider",
                    error_type="TransactionError",
                    error_location="commit_transaction",
                    component=self.name,
                    operation="commit_database_transaction",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="commit_transaction",
                    retry_count=0,
                ),
                cause=e,
) from None

    async def rollback_transaction(self, transaction: Any) -> bool:
        """Rollback a database transaction.

        Args:
            transaction: Transaction wrapper from begin_transaction()

        Returns:
            True if transaction was rolled back successfully

        Raises:
            ProviderError: If transaction rollback fails
        """
        if (
            not transaction
            or not hasattr(transaction, "transaction")
            or not hasattr(transaction, "connection")
        ):
            raise ProviderError(
                message="Invalid transaction object",
                context=ErrorContext.create(
                    flow_name="postgres_provider",
                    error_type="ValidationError",
                    error_location="rollback_transaction",
                    component=self.name,
                    operation="validate_transaction_object",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="rollback_transaction",
                    retry_count=0,
                ),
            )

        try:
            # Rollback the transaction
            await transaction.transaction.rollback()

            # Release the connection back to the pool
            if self._pool is not None:
                await self._pool.release(transaction.connection)  # type: ignore[unreachable]

            return True

        except Exception as e:
            # Try to release the connection
            try:
                if self._pool is not None:
                    await self._pool.release(transaction.connection)  # type: ignore[unreachable]
            except Exception:
                pass

            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to rollback transaction: {str(e)}",
                context=ErrorContext.create(
                    flow_name="postgres_provider",
                    error_type="TransactionError",
                    error_location="rollback_transaction",
                    component=self.name,
                    operation="rollback_database_transaction",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="rollback_transaction",
                    retry_count=0,
                ),
                cause=e,
) from None

    async def check_connection(self) -> bool:
        """Check if database connection is active.

        Returns:
            True if connection is active, False otherwise
        """
        if not self._initialized or not self._pool:
            return False

        try:  # type: ignore[unreachable]
            # Try to acquire a connection and execute a simple query
            async with self._pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception:
            return False

    async def get_health(self) -> DatabaseHealthInfo:
        """Get database health information.

        Returns:
            Dict containing health metrics

        Raises:
            ProviderError: If health check fails
        """
        if not self._initialized or not self._pool:
            return DatabaseHealthInfo(
                status="not_initialized",
                connected=False,
                connection_active=False,
                database=DatabaseInfo(
                    path=f"{self._settings.host}:{self._settings.port}",
                    name=self._settings.database,
                ),
                pool=PoolInfo(active_connections=0, pool_size=0),
                version=None,
            )

        try:  # type: ignore[unreachable]
            # Check connection
            connection_active = await self.check_connection()

            # Get pool statistics
            pool_stats = {
                "min_size": self._settings.min_connections,
                "max_size": self._settings.max_connections,
                "current_size": len(self._pool._holders) if hasattr(self._pool, "_holders") else 0,
                "free_connections": len(self._pool._queue._queue)
                if hasattr(self._pool, "_queue")
                else 0,
                "used_connections": (len(self._pool._holders) - len(self._pool._queue._queue))
                if hasattr(self._pool, "_holders") and hasattr(self._pool, "_queue")
                else 0,
            }

            # Get PostgreSQL version
            version = None
            if connection_active:
                async with self._pool.acquire() as conn:
                    version = await conn.fetchval("SHOW server_version")

            return {
                "status": "healthy" if connection_active else "unhealthy",
                "pool": pool_stats,
                "connection_active": connection_active,
                "version": version,
                "host": self._settings.host,
                "database": self._settings.database,
            }

        except Exception as e:
            # Return error status
            return {"status": "error", "error": str(e), "connection_active": False}

    def _create_connection_string(self) -> str:
        """Create PostgreSQL connection string from settings.

        Returns:
            Connection string (DSN)
        """
        # Build DSN URL format for asyncpg
        auth_part = ""
        if self._settings.username:
            auth_part = self._settings.username
            if self._settings.password:
                auth_part += f":{self._settings.password}"
            auth_part += "@"

        host_part = self._settings.host
        port_part = f":{self._settings.port}" if self._settings.port else ""
        database_part = f"/{self._settings.database}" if self._settings.database else ""

        # Build query parameters
        params = []
        if self._settings.ssl_mode:
            params.append(f"sslmode={self._settings.ssl_mode}")

        query_part = f"?{'&'.join(params)}" if params else ""

        return f"postgresql://{auth_part}{host_part}{port_part}{database_part}{query_part}"

    def _convert_params(
        self, query: str, params: dict[str, Any] | None
    ) -> tuple[str, list[Any]]:
        """Convert named parameters to positional parameters.

        Args:
            query: SQL query with named parameters
            params: Query parameters

        Returns:
            Tuple of (modified query, positional parameters)
        """
        if not params:
            return query, []

        # Map of parameter names to their positions
        param_map = {}
        positional_params = []

        # Find all parameter references in the query
        modified_query = query
        for i, (name, value) in enumerate(params.items()):
            position = i + 1
            param_map[name] = position
            positional_params.append(value)

            # Replace named parameters with positional ones
            # Handle both :param and @param styles
            modified_query = modified_query.replace(f":{name}", f"${position}")
            modified_query = modified_query.replace(f"@{name}", f"${position}")

        return modified_query, positional_params

    def _register_type_codecs(self) -> None:
        """Register custom type encoders and decoders for PostgreSQL."""
        # This would be implemented to handle JSON, arrays, etc.
        # For example, registering JSON encoding/decoding
        pass

    def _should_retry(self, exception: Exception) -> bool:
        """Check if an exception should trigger a retry.

        Args:
            exception: Exception that occurred

        Returns:
            True if should retry, False otherwise
        """
        # Check if asyncpg is available and it's a specific asyncpg error
        if ASYNCPG_AVAILABLE:
            if isinstance(
                exception,
                (
                    asyncpg.ConnectionDoesNotExistError,
                    asyncpg.ConnectionFailureError,
                    asyncpg.InterfaceError,
                ),
            ):
                return True

        # Check for network-related errors
        if "connection" in str(exception).lower() and "closed" in str(exception).lower():
            return True

        return False

    async def _retry_execute(
        self, query: str, params: dict[str, Any] | None = None, attempt: int = 1
    ) -> list[dict[str, Any]]:
        """Retry executing a query with exponential backoff.

        Args:
            query: SQL query to execute
            params: Query parameters
            attempt: Current attempt number

        Returns:
            Query results

        Raises:
            ProviderError: If all retries fail
        """
        if attempt > self._settings.max_retries:
            raise ProviderError(
                message=f"Failed to execute query after {attempt - 1} retries",
                context=ErrorContext.create(
                    flow_name="postgres_provider",
                    error_type="RetryExhaustionError",
                    error_location="_retry_execute",
                    component=self.name,
                    operation="retry_query_execution",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="retry_execute",
                    retry_count=attempt - 1,
                ),
            )

        # Calculate backoff delay
        delay = self._settings.retry_delay_seconds * (2 ** (attempt - 1))

        # Wait before retrying
        await asyncio.sleep(delay)

        try:
            # Try to execute the query again
            return await self.execute(query, params)
        except Exception as e:
            if self._should_retry(e):
                # Retry again
                return await self._retry_execute(query, params, attempt + 1)
            else:
                # Re-raise if not retriable
                raise

    async def _retry_execute_many(
        self, query: str, params_list: list[dict[str, Any]], attempt: int = 1
    ) -> list[Any]:
        """Retry executing a batch query with exponential backoff.

        Args:
            query: SQL query to execute
            params_list: List of query parameters
            attempt: Current attempt number

        Returns:
            List of query results

        Raises:
            ProviderError: If all retries fail
        """
        if attempt > self._settings.max_retries:
            raise ProviderError(
                message=f"Failed to execute batch query after {attempt - 1} retries",
                context=ErrorContext.create(
                    flow_name="postgres_provider",
                    error_type="RetryExhaustionError",
                    error_location="_retry_execute_many",
                    component=self.name,
                    operation="retry_batch_query_execution",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="retry_execute_many",
                    retry_count=attempt - 1,
                ),
            )

        # Calculate backoff delay
        delay = self._settings.retry_delay_seconds * (2 ** (attempt - 1))

        # Wait before retrying
        await asyncio.sleep(delay)

        try:
            # Try to execute the batch query again
            return await self.execute_many(query, params_list)
        except Exception as e:
            if self._should_retry(e):
                # Retry again
                return await self._retry_execute_many(query, params_list, attempt + 1)
            else:
                # Re-raise if not retriable
                raise
