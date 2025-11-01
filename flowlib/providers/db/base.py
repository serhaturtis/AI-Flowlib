"""Database provider base class and related functionality.

This module provides the base class for implementing database providers
that share common functionality for querying, updating, and managing
database operations.
"""

import logging
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.base import Provider, ProviderSettings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
SettingsT = TypeVar("SettingsT", bound="DBProviderSettings")


class DatabaseInfo(BaseModel):
    """Database connection information."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(..., description="Database path or connection string")
    name: str = Field(..., description="Database name or type")


class PoolInfo(BaseModel):
    """Connection pool information."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    active_connections: int = Field(..., description="Number of active connections")
    pool_size: int = Field(..., description="Maximum pool size")


class DatabaseHealthInfo(BaseModel):
    """Database health information."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="Overall health status")
    connected: bool = Field(..., description="Whether database is connected")
    connection_active: bool = Field(..., description="Whether connection is active")
    database: DatabaseInfo = Field(..., description="Database information")
    pool: PoolInfo = Field(..., description="Connection pool information")
    version: str | None = Field(None, description="Database version")
    additional_info: dict[str, Any] = Field(
        default_factory=dict, description="Additional health metrics"
    )


class DBProviderSettings(ProviderSettings):
    """Base settings for database providers.

    Attributes:
        host: Database server host address
        port: Database server port
        database: Database name
        username: Authentication username
        password: Authentication password
        pool_size: Connection pool size
        timeout: Connection/query timeout in seconds
    """

    # Connection settings
    host: str = "localhost"
    port: int | None = None
    database: str = ""
    username: str | None = None
    password: str | None = None

    # Pool settings
    pool_size: int = 5
    min_size: int = 1
    max_overflow: int = 10

    # Timeout settings
    timeout: float = 30.0
    connect_timeout: float = 10.0

    # SSL settings
    use_ssl: bool = False
    ssl_ca_cert: str | None = None
    ssl_cert: str | None = None
    ssl_key: str | None = None

    # Query settings
    query_timeout: float = 30.0
    max_query_size: int = 1000
    application_name: str = "flowlib"
    auto_reconnect: bool = True
    retry_count: int = 3
    retry_delay: float = 1.0


class DBProvider(Provider[SettingsT], Generic[SettingsT]):
    """Base class for database providers.

    This class provides:
    1. Common database operations (query, execute, transaction)
    2. Type-safe operations with Pydantic models
    3. Connection pooling and lifecycle management
    4. Error handling and retries
    """

    def __init__(self, name: str = "db", settings: SettingsT | None = None):
        """Initialize database provider.

        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        # Pass provider_type="db" to the parent class
        super().__init__(name=name, settings=settings, provider_type="db")
        self._initialized = False
        self._pool = None

    @property
    def initialized(self) -> bool:
        """Return whether provider has been initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the database provider.

        This method should be implemented by subclasses to establish
        connections to the database and set up connection pools.
        """
        self._initialized = True

    async def shutdown(self) -> None:
        """Close all connections and release resources.

        This method should be implemented by subclasses to properly
        close connections and clean up resources.
        """
        self._initialized = False
        self._pool = None

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a database query.

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            Query results

        Raises:
            ProviderError: If query execution fails
        """
        raise NotImplementedError("Subclasses must implement execute()")

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
        raise NotImplementedError("Subclasses must implement execute_many()")

    async def execute_structured(
        self, query: str, output_type: type[T], params: dict[str, Any] | None = None
    ) -> list[T]:
        """Execute a query and parse results into structured types.

        Args:
            query: SQL query to execute
            output_type: Pydantic model to parse results into
            params: Query parameters

        Returns:
            List of parsed model instances

        Raises:
            ProviderError: If query execution or parsing fails
        """
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
                    # Get column names from the cursor description
                    # This is implementation-specific and should be handled by subclasses
                    raise NotImplementedError(
                        "Tuple/list conversion must be implemented by subclasses"
                    )
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
            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to execute structured query: {str(e)}",
                context=ErrorContext.create(
                    flow_name="db_provider",
                    error_type="StructuredQueryError",
                    error_location="execute_structured",
                    component=self.name,
                    operation="query_execution",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="execute_structured",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def begin_transaction(self) -> Any:
        """Begin a database transaction.

        Returns:
            Transaction object

        Raises:
            ProviderError: If transaction start fails
        """
        raise NotImplementedError("Subclasses must implement begin_transaction()")

    async def commit_transaction(self, transaction: Any) -> bool:
        """Commit a database transaction.

        Args:
            transaction: Transaction object from begin_transaction()

        Returns:
            True if transaction was committed successfully

        Raises:
            ProviderError: If transaction commit fails
        """
        raise NotImplementedError("Subclasses must implement commit_transaction()")

    async def rollback_transaction(self, transaction: Any) -> bool:
        """Rollback a database transaction.

        Args:
            transaction: Transaction object from begin_transaction()

        Returns:
            True if transaction was rolled back successfully

        Raises:
            ProviderError: If transaction rollback fails
        """
        raise NotImplementedError("Subclasses must implement rollback_transaction()")

    async def check_connection(self) -> bool:
        """Check if database connection is active.

        Returns:
            True if connection is active, False otherwise
        """
        raise NotImplementedError("Subclasses must implement check_connection()")

    async def get_health(self) -> DatabaseHealthInfo:
        """Get database health information.

        Returns:
            Structured health information

        Raises:
            ProviderError: If health check fails
        """
        raise NotImplementedError("Subclasses must implement get_health()")
