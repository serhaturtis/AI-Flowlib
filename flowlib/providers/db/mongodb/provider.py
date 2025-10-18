"""MongoDB database provider implementation.

This module provides a concrete implementation of the DBProvider
for MongoDB database using motor (async driver).
"""

import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional, cast
from typing import List as ListType

from pydantic import BaseModel, Field

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.base import Provider, ProviderSettings
from flowlib.providers.core.decorators import provider
from flowlib.providers.db.base import DatabaseHealthInfo, DatabaseInfo, PoolInfo

logger = logging.getLogger(__name__)

try:
    from bson import ObjectId
except ImportError:
    ObjectId = Any  # type: ignore
    logger.warning("bson module not found. Install with 'pip install pymongo'")
# Removed ProviderType import - using config-driven provider access

try:
    import motor.motor_asyncio
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
    MOTOR_AVAILABLE = True
except ImportError:
    AsyncIOMotorClient = None  # type: ignore
    AsyncIOMotorDatabase = None  # type: ignore
    MOTOR_AVAILABLE = False
    logger.warning("motor package not found. Install with 'pip install motor'")


class MongoDBInsertOneResult(BaseModel):
    """Result model for MongoDB insert_one operation."""
    inserted_id: str = Field(..., description="ID of the inserted document")


class MongoDBInsertManyResult(BaseModel):
    """Result model for MongoDB insert_many operation."""
    inserted_ids: ListType[str] = Field(..., description="IDs of the inserted documents")


class MongoDBUpdateResult(BaseModel):
    """Result model for MongoDB update operations."""
    modified_count: int = Field(..., description="Number of documents modified")
    matched_count: int = Field(default=0, description="Number of documents matched")


class MongoDBDeleteResult(BaseModel):
    """Result model for MongoDB delete operations."""
    deleted_count: int = Field(..., description="Number of documents deleted")


class MongoDBCountResult(BaseModel):
    """Result model for MongoDB count operations."""
    count: int = Field(..., description="Number of documents counted")


class MongoDBProviderSettings(ProviderSettings):
    """MongoDB provider settings - direct inheritance, only MongoDB-specific fields.
    
    MongoDB requires:
    1. Database connection (host, port, database name)
    2. Authentication (username, password, auth_source)
    3. Connection pooling and timeout configuration
    4. MongoDB-specific features (replica sets, read/write concerns)
    """

    # Connection settings
    host: str = Field(default="localhost", description="MongoDB server host")
    port: int = Field(default=27017, description="MongoDB server port")
    username: Optional[str] = Field(default=None, description="MongoDB username")
    password: Optional[str] = Field(default=None, description="MongoDB password")
    database: str = Field(default="test", description="MongoDB database name")

    # Connection string override (takes precedence over host/port)
    connection_string: Optional[str] = Field(default=None, description="MongoDB connection string (overrides host/port if provided)")

    # Authentication settings
    auth_source: str = Field(default="admin", description="Authentication database name")
    auth_mechanism: Optional[str] = Field(default=None, description="Authentication mechanism")

    # Timeout settings
    connect_timeout_ms: int = Field(default=20000, description="Connection timeout in milliseconds")
    server_selection_timeout_ms: int = Field(default=20000, description="Server selection timeout in milliseconds")

    # Connection pool settings
    max_pool_size: Optional[int] = Field(default=None, description="Maximum number of connections in the pool")
    min_pool_size: Optional[int] = Field(default=None, description="Minimum number of connections in the pool")
    max_idle_time_ms: Optional[int] = Field(default=None, description="Maximum idle time for connections in milliseconds")

    # MongoDB operation settings
    read_preference: Optional[str] = Field(default=None, description="Read preference setting")
    write_concern: Optional[Dict[str, Any]] = Field(default=None, description="Write concern settings")
    read_concern: Optional[Dict[str, Any]] = Field(default=None, description="Read concern settings")
    replica_set: Optional[str] = Field(default=None, description="Replica set name")

    # SSL settings
    ssl_enabled: Optional[bool] = Field(default=None, description="Whether SSL is enabled")
    ssl_cert_reqs: Optional[str] = Field(default=None, description="SSL certificate requirements")
    ssl_ca_certs: Optional[str] = Field(default=None, description="SSL CA certificates path")
    ssl_certfile: Optional[str] = Field(default=None, description="SSL certificate file path")
    ssl_keyfile: Optional[str] = Field(default=None, description="SSL key file path")


    # Additional connection arguments
    connect_args: Dict[str, Any] = Field(default_factory=dict)



@provider(provider_type="db", name="mongodb", settings_class=MongoDBProviderSettings)
class MongoDBProvider(Provider[MongoDBProviderSettings]):
    """MongoDB implementation of the DBProvider.
    
    This provider implements database operations using motor,
    an asynchronous driver for MongoDB.
    """

    def __init__(self, name: str = "mongodb", settings: Optional[MongoDBProviderSettings] = None):
        """Initialize MongoDB provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        super().__init__(name=name, provider_type="database", settings=settings or MongoDBProviderSettings(database="test"))
        self._settings = settings or MongoDBProviderSettings(database="test")
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None

    async def _initialize(self) -> None:
        """Initialize MongoDB connection.
        
        Raises:
            ProviderError: If initialization fails
        """
        try:
            # Create client
            if self._settings.connection_string:
                # Prepare client arguments with connection string
                client_args = {
                    "serverSelectionTimeoutMS": self._settings.server_selection_timeout_ms,
                    "connectTimeoutMS": self._settings.connect_timeout_ms,
                    **self._settings.connect_args
                }

                # Add MongoDB-specific settings if provided
                if self._settings.max_pool_size is not None:
                    client_args["maxPoolSize"] = self._settings.max_pool_size
                if self._settings.min_pool_size is not None:
                    client_args["minPoolSize"] = self._settings.min_pool_size
                if self._settings.max_idle_time_ms is not None:
                    client_args["maxIdleTimeMS"] = self._settings.max_idle_time_ms
                if self._settings.read_preference is not None:
                    client_args["readPreference"] = self._settings.read_preference
                if self._settings.write_concern is not None:
                    if "w" not in self._settings.write_concern:
                        raise ValueError("Write concern missing required 'w' field")
                    client_args["w"] = self._settings.write_concern["w"]
                    if "j" in self._settings.write_concern:
                        client_args["j"] = self._settings.write_concern["j"]
                if self._settings.read_concern is not None:
                    if "level" not in self._settings.read_concern:
                        raise ValueError("Read concern missing required 'level' field")
                    client_args["readConcernLevel"] = self._settings.read_concern["level"]
                if self._settings.replica_set is not None:
                    client_args["replicaSet"] = self._settings.replica_set
                if self._settings.ssl_enabled is not None:
                    client_args["ssl"] = self._settings.ssl_enabled
                if self._settings.ssl_cert_reqs is not None:
                    client_args["ssl_cert_reqs"] = self._settings.ssl_cert_reqs
                if self._settings.ssl_ca_certs is not None:
                    client_args["ssl_ca_certs"] = self._settings.ssl_ca_certs
                if self._settings.ssl_certfile is not None:
                    client_args["ssl_certfile"] = self._settings.ssl_certfile
                if self._settings.ssl_keyfile is not None:
                    client_args["ssl_keyfile"] = self._settings.ssl_keyfile

                # Use connection string if provided
                self._client = motor.motor_asyncio.AsyncIOMotorClient(
                    self._settings.connection_string,
                    **client_args
                )
            else:
                # Prepare client arguments - only include non-None values
                client_args = {
                    "host": self._settings.host,
                    "port": self._settings.port,
                    "serverSelectionTimeoutMS": self._settings.server_selection_timeout_ms,
                    "connectTimeoutMS": self._settings.connect_timeout_ms,
                    **self._settings.connect_args
                }

                # Only add authentication fields if they're specified
                if self._settings.username is not None:
                    client_args["username"] = self._settings.username
                if self._settings.password is not None:
                    client_args["password"] = self._settings.password
                if self._settings.auth_source is not None:
                    client_args["authSource"] = self._settings.auth_source
                if self._settings.auth_mechanism is not None:
                    client_args["authMechanism"] = self._settings.auth_mechanism

                # Add MongoDB-specific settings if provided
                if self._settings.max_pool_size is not None:
                    client_args["maxPoolSize"] = self._settings.max_pool_size
                if self._settings.min_pool_size is not None:
                    client_args["minPoolSize"] = self._settings.min_pool_size
                if self._settings.max_idle_time_ms is not None:
                    client_args["maxIdleTimeMS"] = self._settings.max_idle_time_ms
                if self._settings.read_preference is not None:
                    client_args["readPreference"] = self._settings.read_preference
                if self._settings.write_concern is not None:
                    if "w" not in self._settings.write_concern:
                        raise ValueError("Write concern missing required 'w' field")
                    client_args["w"] = self._settings.write_concern["w"]
                    if "j" in self._settings.write_concern:
                        client_args["j"] = self._settings.write_concern["j"]
                if self._settings.read_concern is not None:
                    if "level" not in self._settings.read_concern:
                        raise ValueError("Read concern missing required 'level' field")
                    client_args["readConcernLevel"] = self._settings.read_concern["level"]
                if self._settings.replica_set is not None:
                    client_args["replicaSet"] = self._settings.replica_set
                if self._settings.ssl_enabled is not None:
                    client_args["ssl"] = self._settings.ssl_enabled
                if self._settings.ssl_cert_reqs is not None:
                    client_args["ssl_cert_reqs"] = self._settings.ssl_cert_reqs
                if self._settings.ssl_ca_certs is not None:
                    client_args["ssl_ca_certs"] = self._settings.ssl_ca_certs
                if self._settings.ssl_certfile is not None:
                    client_args["ssl_certfile"] = self._settings.ssl_certfile
                if self._settings.ssl_keyfile is not None:
                    client_args["ssl_keyfile"] = self._settings.ssl_keyfile

                # Use host and port
                self._client = motor.motor_asyncio.AsyncIOMotorClient(**client_args)

            # Get database
            self._db = self._client[self._settings.database]

            # Ping database to verify connection
            await self._client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {self._settings.host}:{self._settings.port}/{self._settings.database}")

        except Exception as e:
            self._client = None
            self._db = None
            raise ProviderError(
                message=f"Failed to connect to MongoDB: {str(e)}",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="ConnectionError",
                    error_location="_initialize",
                    component=self.name,
                    operation="database_connection"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="database_connection",
                    retry_count=0
                ),
                cause=e
            )

    async def _shutdown(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Error during MongoDB shutdown: {str(e)}")
            finally:
                self._client = None
                self._db = None
                logger.info(f"Closed MongoDB connection: {self._settings.host}:{self._settings.port}")

    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a MongoDB operation.
        
        Args:
            query: Operation type (find, insert_one, update_one, delete_one, etc.)
            params: Operation parameters
            
        Returns:
            List of results
            
        Raises:
            ProviderError: If operation fails
        """
        if not params:
            raise ProviderError(
                message="MongoDB operations require parameters",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="ValidationError",
                    error_location="execute",
                    component=self.name,
                    operation="validate_parameters"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="execute",
                    retry_count=0
                )
            )

        if "collection" not in params:
            raise ProviderError(
                message="MongoDB query operation requires 'collection' parameter",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="ValidationError",
                    error_location="execute",
                    component=self.name,
                    operation="validate_collection_parameter"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="execute",
                    retry_count=0
                )
            )
        collection = params["collection"]
        if not collection:
            raise ProviderError(
                message="Collection name required for MongoDB operations",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="ValidationError",
                    error_location="execute",
                    component=self.name,
                    operation="validate_collection"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="execute",
                    retry_count=0
                )
            )

        try:
            if query == "find":
                if "filter" not in params:
                    raise KeyError("Required 'filter' parameter missing for find query")
                filter_query = params["filter"]
                projection = params["projection"] if "projection" in params else None
                sort = params["sort"] if "sort" in params else None
                limit = params["limit"] if "limit" in params else None
                skip = params["skip"] if "skip" in params else None
                return await self.execute_query(collection, filter_query, projection, sort, limit, skip)

            elif query == "insert_one":
                if "document" not in params:
                    raise KeyError("Required 'document' parameter missing for insert_one query")
                document = params["document"]
                document_id = await self.insert_document(collection, document)
                return [{"operation": "insert_one", "inserted_id": str(document_id), "acknowledged": True}]

            elif query == "insert_many":
                if "documents" not in params:
                    raise KeyError("Required 'documents' parameter missing for insert_many query")
                documents = params["documents"]
                inserted_ids = []
                for document in documents:
                    document_id = await self.insert_document(collection, document)
                    inserted_ids.append(document_id)
                return [{"operation": "insert_many", "inserted_ids": [str(id) for id in inserted_ids], "acknowledged": True}]

            elif query == "update_one":
                if "filter" not in params:
                    raise ValueError("update_one operation requires 'filter' parameter")
                if "update" not in params:
                    raise ValueError("update_one operation requires 'update' parameter")
                filter_query = params["filter"]
                update = params["update"]
                upsert = params["upsert"] if "upsert" in params else False
                modified_count = await self.update_document(collection, filter_query, update, upsert)
                return [{"operation": "update_one", "modified_count": modified_count, "acknowledged": True}]

            elif query == "delete_one":
                if "filter" not in params:
                    raise ValueError("delete_one operation requires 'filter' parameter")
                filter_query = params["filter"]
                deleted_count = await self.delete_document(collection, filter_query)
                return [{"operation": "delete_one", "deleted_count": deleted_count, "acknowledged": True}]

            elif query == "delete_many":
                if "filter" not in params:
                    raise ValueError("delete_many operation requires 'filter' parameter")
                filter_query = params["filter"]
                deleted_count = await self.delete_document(collection, filter_query)
                return [{"operation": "delete_many", "deleted_count": deleted_count, "acknowledged": True}]

            elif query == "count":
                filter_query = params["filter"] if "filter" in params else {}
                count = await self.count_documents(collection, filter_query)
                return [{"operation": "count", "count": count}]

            else:
                raise ProviderError(
                    message=f"Unsupported MongoDB operation: {query}",
                    context=ErrorContext.create(
                        flow_name="mongodb_provider",
                        error_type="UnsupportedOperationError",
                        error_location="execute",
                        component=self.name,
                        operation="validate_query_type"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="db",
                        operation="execute",
                        retry_count=0
                    )
                )

        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                message=f"Failed to execute MongoDB operation '{query}': {str(e)}",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="OperationExecutionError",
                    error_location="execute",
                    component=self.name,
                    operation=f"execute_{query}"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="execute",
                    retry_count=0
                ),
                cause=e
            )

    async def execute_query(self,
                           collection: str,
                           query: Dict[str, Any],
                           projection: Optional[Dict[str, Any]] = None,
                           sort: Optional[List[tuple]] = None,
                           limit: Optional[int] = None,
                           skip: Optional[int] = None) -> List[Dict[str, Any]]:
        """Execute a MongoDB query.
        
        Args:
            collection: Collection name
            query: MongoDB query dict
            projection: Optional fields to return
            sort: Optional sort specification
            limit: Optional limit
            skip: Optional skip
            
        Returns:
            List of documents as dictionaries
            
        Raises:
            ProviderError: If query execution fails
        """
        if self._db is None:
            await self.initialize()

        if self._db is None:
            raise ProviderError(
                message="MongoDB database not available after initialization",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="InitializationError",
                    error_location=f"{self.__class__.__name__}",
                    component=self.name,
                    operation="database_access"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="database",
                    operation="database_access",
                    retry_count=0
                )
            )

        try:
            # Get collection
            coll = self._db[collection]

            # Build cursor
            cursor = coll.find(query, projection)

            # Apply sort, limit, skip if provided
            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)

            # Get results
            results = await cursor.to_list(length=None)

            # Convert ObjectId to string
            for doc in results:
                if '_id' in doc and isinstance(doc['_id'], ObjectId):
                    doc['_id'] = str(doc['_id'])

            return cast(List[Dict[str, Any]], results)

        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute MongoDB query: {str(e)}",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="QueryExecutionError",
                    error_location="execute_query",
                    component=self.name,
                    operation="document_query"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="document_query",
                    retry_count=0
                ),
                cause=e
            )

    async def insert_document(self, collection: str, document: Dict[str, Any]) -> str:
        """Insert a document into a collection.
        
        Args:
            collection: Collection name
            document: Document to insert
            
        Returns:
            ID of inserted document
            
        Raises:
            ProviderError: If insert fails
        """
        if self._db is None:
            await self.initialize()

        if self._db is None:
            raise ProviderError(
                message="MongoDB database not available after initialization",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="InitializationError",
                    error_location=f"{self.__class__.__name__}",
                    component=self.name,
                    operation="database_access"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="database",
                    operation="database_access",
                    retry_count=0
                )
            )

        try:
            # Get collection
            coll = self._db[collection]

            # Insert document
            result = await coll.insert_one(document)

            # Return inserted ID as string
            return str(result.inserted_id)

        except Exception as e:
            raise ProviderError(
                message=f"Failed to insert document: {str(e)}",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="DocumentInsertError",
                    error_location="insert_document",
                    component=self.name,
                    operation="document_insert"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="document_insert",
                    retry_count=0
                ),
                cause=e
            )

    async def update_document(self,
                             collection: str,
                             query: Dict[str, Any],
                             update: Dict[str, Any],
                             upsert: bool = False) -> int:
        """Update documents in a collection.
        
        Args:
            collection: Collection name
            query: Query to match documents
            update: Update operations
            upsert: Whether to insert if no documents match
            
        Returns:
            Number of documents modified
            
        Raises:
            ProviderError: If update fails
        """
        if self._db is None:
            await self.initialize()

        if self._db is None:
            raise ProviderError(
                message="MongoDB database not available after initialization",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="InitializationError",
                    error_location=f"{self.__class__.__name__}",
                    component=self.name,
                    operation="database_access"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="database",
                    operation="database_access",
                    retry_count=0
                )
            )

        try:
            # Get collection
            coll = self._db[collection]

            # Update documents
            result = await coll.update_many(query, update, upsert=upsert)

            # Return modified count
            return result.modified_count

        except Exception as e:
            raise ProviderError(
                message=f"Failed to update documents: {str(e)}",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="DocumentUpdateError",
                    error_location="update_document",
                    component=self.name,
                    operation="document_update"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="document_update",
                    retry_count=0
                ),
                cause=e
            )

    async def delete_document(self, collection: str, query: Dict[str, Any]) -> int:
        """Delete documents from a collection.
        
        Args:
            collection: Collection name
            query: Query to match documents
            
        Returns:
            Number of documents deleted
            
        Raises:
            ProviderError: If delete fails
        """
        if self._db is None:
            await self.initialize()

        if self._db is None:
            raise ProviderError(
                message="MongoDB database not available after initialization",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="InitializationError",
                    error_location=f"{self.__class__.__name__}",
                    component=self.name,
                    operation="database_access"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="database",
                    operation="database_access",
                    retry_count=0
                )
            )

        try:
            # Get collection
            coll = self._db[collection]

            # Delete documents
            result = await coll.delete_many(query)

            # Return deleted count
            return result.deleted_count

        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete documents: {str(e)}",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="DocumentDeleteError",
                    error_location="delete_document",
                    component=self.name,
                    operation="document_delete"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="document_delete",
                    retry_count=0
                ),
                cause=e
            )

    async def create_index(self,
                          collection: str,
                          keys: List[tuple],
                          unique: bool = False,
                          sparse: bool = False) -> str:
        """Create an index on a collection.
        
        Args:
            collection: Collection name
            keys: List of (field, direction) tuples
            unique: Whether index should enforce uniqueness
            sparse: Whether index should be sparse
            
        Returns:
            Name of created index
            
        Raises:
            ProviderError: If index creation fails
        """
        if self._db is None:
            await self.initialize()

        if self._db is None:
            raise ProviderError(
                message="MongoDB database not available after initialization",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="InitializationError",
                    error_location=f"{self.__class__.__name__}",
                    component=self.name,
                    operation="database_access"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="database",
                    operation="database_access",
                    retry_count=0
                )
            )

        try:
            # Get collection
            coll = self._db[collection]

            # Create index
            result = await coll.create_index(keys, unique=unique, sparse=sparse)

            # Return index name
            return result

        except Exception as e:
            raise ProviderError(
                message=f"Failed to create index: {str(e)}",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="IndexCreationError",
                    error_location="create_index",
                    component=self.name,
                    operation="index_creation"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="index_creation",
                    retry_count=0
                ),
                cause=e
            )

    async def begin_transaction(self) -> Any:
        """Begin a MongoDB transaction.
        
        Returns:
            Transaction session
            
        Raises:
            ProviderError: If transaction start fails
        """
        if self._client is None:
            await self.initialize()

        if self._client is None:
            raise ProviderError(
                message="MongoDB client not available after initialization",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="InitializationError",
                    error_location=f"{self.__class__.__name__}",
                    component=self.name,
                    operation="client_access"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="database",
                    operation="client_access",
                    retry_count=0
                )
            )

        try:
            # Start a session for transaction
            session = await self._client.start_session()
            # Start a transaction on the session
            session.start_transaction()
            return session

        except Exception as e:
            raise ProviderError(
                message=f"Failed to begin transaction: {str(e)}",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="TransactionError",
                    error_location="begin_transaction",
                    component=self.name,
                    operation="start_transaction"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="begin_transaction",
                    retry_count=0
                ),
                cause=e
            )

    async def commit_transaction(self, session: Any) -> bool:
        """Commit a MongoDB transaction.
        
        Args:
            session: Transaction session from begin_transaction()
            
        Returns:
            True if committed successfully
            
        Raises:
            ProviderError: If commit fails
        """
        try:
            await session.commit_transaction()
            await session.end_session()
            return True

        except Exception as e:
            try:
                await session.end_session()
            except Exception as session_error:
                # Log session cleanup failure but don't mask the original transaction error
                logger.warning(f"Failed to cleanup MongoDB session during transaction error: {session_error}")
            raise ProviderError(
                message=f"Failed to commit transaction: {str(e)}",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="TransactionError",
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

    async def rollback_transaction(self, session: Any) -> bool:
        """Rollback a MongoDB transaction.
        
        Args:
            session: Transaction session from begin_transaction()
            
        Returns:
            True if rolled back successfully
            
        Raises:
            ProviderError: If rollback fails
        """
        try:
            await session.abort_transaction()
            await session.end_session()
            return True

        except Exception as e:
            try:
                await session.end_session()
            except Exception as session_error:
                # Log session cleanup failure but don't mask the original rollback error
                logger.warning(f"Failed to cleanup MongoDB session during rollback error: {session_error}")
            raise ProviderError(
                message=f"Failed to rollback transaction: {str(e)}",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="TransactionError",
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

    async def check_connection(self) -> bool:
        """Check if MongoDB connection is active.
        
        Returns:
            True if connection is active, False otherwise
        """
        if self._client is None:
            return False

        try:
            # Try to ping the database
            await self._client.admin.command('ping')
            return True
        except Exception:
            return False

    async def get_health(self) -> DatabaseHealthInfo:
        """Get MongoDB health information.
        
        Returns:
            Structured health information
        """
        if self._client is None:
            return DatabaseHealthInfo(
                status="not_initialized",
                connected=False,
                connection_active=False,
                database=DatabaseInfo(
                    path=f"{self._settings.host}:{self._settings.port}/{self._settings.database}",
                    name=self._settings.database
                ),
                pool=PoolInfo(
                    active_connections=0,
                    pool_size=self._settings.max_pool_size if self._settings.max_pool_size is not None else 10
                ),
                version=None
            )

        try:
            connection_active = await self.check_connection()

            # Get server info if connected
            server_info = None
            version = None
            if connection_active:
                try:
                    server_info = await self._client.server_info()
                    version = server_info["version"] if "version" in server_info else "unknown"
                except Exception as server_info_error:
                    # Log server info retrieval failure but continue with health check
                    logger.debug(f"Could not retrieve MongoDB server info: {server_info_error}")
                    server_info = None
                    version = "unavailable"

            # Build additional info
            additional_info = {}
            if server_info:
                additional_info["server_info"] = server_info

            return DatabaseHealthInfo(
                status="healthy" if connection_active else "unhealthy",
                connected=connection_active,
                connection_active=connection_active,
                database=DatabaseInfo(
                    path=f"{self._settings.host}:{self._settings.port}/{self._settings.database}",
                    name=self._settings.database
                ),
                pool=PoolInfo(
                    active_connections=1 if connection_active else 0,  # MongoDB client manages connections internally
                    pool_size=self._settings.max_pool_size if self._settings.max_pool_size is not None else 0
                ),
                version=version,
                additional_info=additional_info
            )

        except Exception as e:
            return DatabaseHealthInfo(
                status="error",
                connected=False,
                connection_active=False,
                database=DatabaseInfo(
                    path=f"{self._settings.host}:{self._settings.port}/{self._settings.database}",
                    name=self._settings.database
                ),
                pool=PoolInfo(
                    active_connections=0,
                    pool_size=self._settings.max_pool_size if self._settings.max_pool_size is not None else 10
                ),
                version=None,
                additional_info={"error": str(e)}
            )

    async def execute_transaction(self, operations: Callable[..., Coroutine[Any, Any, Dict[str, Any]]]) -> Dict[str, Any]:
        """Execute operations in a transaction.
        
        Args:
            operations: Async callable that takes the database as argument
            
        Returns:
            Transaction result
            
        Raises:
            ProviderError: If transaction fails
        """
        if self._client is None:
            await self.initialize()

        if self._client is None:
            raise ProviderError(
                message="MongoDB client not available after initialization",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="InitializationError",
                    error_location=f"{self.__class__.__name__}",
                    component=self.name,
                    operation="client_access"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="database",
                    operation="client_access",
                    retry_count=0
                )
            )

        try:
            # Start a session
            async with await self._client.start_session() as session:
                # Start a transaction
                result: Dict[str, Any] = await session.with_transaction(operations)
                return result

        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute transaction: {str(e)}",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="TransactionError",
                    error_location="execute_transaction",
                    component=self.name,
                    operation="transaction_execution"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="transaction_execution",
                    retry_count=0
                ),
                cause=e
            )

    async def count_documents(self, collection: str, query: Dict[str, Any]) -> int:
        """Count documents in a collection.
        
        Args:
            collection: Collection name
            query: Query to match documents
            
        Returns:
            Number of documents matched
            
        Raises:
            ProviderError: If count fails
        """
        if self._db is None:
            await self.initialize()

        if self._db is None:
            raise ProviderError(
                message="MongoDB database not available after initialization",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="InitializationError",
                    error_location=f"{self.__class__.__name__}",
                    component=self.name,
                    operation="database_access"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="database",
                    operation="database_access",
                    retry_count=0
                )
            )

        try:
            # Get collection
            coll = self._db[collection]

            # Count documents
            return await coll.count_documents(query)

        except Exception as e:
            raise ProviderError(
                message=f"Failed to count documents: {str(e)}",
                context=ErrorContext.create(
                    flow_name="mongodb_provider",
                    error_type="DocumentCountError",
                    error_location="count_documents",
                    component=self.name,
                    operation="document_count"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="db",
                    operation="document_count",
                    retry_count=0
                ),
                cause=e
            )
