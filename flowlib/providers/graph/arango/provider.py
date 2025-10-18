"""ArangoDB graph database provider implementation.

This module provides a concrete implementation of the GraphDBProvider 
for ArangoDB, a multi-model database with strong graph capabilities.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Union, cast

from arango import ArangoClient
from arango.collection import StandardCollection
from arango.database import StandardDatabase
from arango.exceptions import ArangoError
from pydantic import Field

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.decorators import provider
from flowlib.providers.graph.base import GraphDBProvider, GraphDBProviderSettings
from flowlib.providers.graph.models import (
    Entity,
    EntityAttribute,
    EntityRelationship,
    EntitySearchResult,
    GraphDeleteResult,
    GraphQueryResult,
    GraphStoreResult,
    RelationshipSearchResult,
)

logger = logging.getLogger(__name__)


class ArangoProviderSettings(GraphDBProviderSettings):
    """ArangoDB provider settings - direct inheritance, only ArangoDB-specific fields.
    
    ArangoDB requires:
    1. HTTP connection (URL-based)
    2. Authentication (username, password)
    3. Database and graph naming
    4. Collection configuration
    """

    # ArangoDB connection settings
    url: str = Field(default="http://localhost:8529", description="ArangoDB server URL (e.g., 'http://localhost:8529')")
    username: str = Field(default="root", description="Username for authentication")
    password: str = Field(default="", description="Password for authentication (should be overridden in production)")
    database: str = Field(default="flowlib", description="Database name")

    # ArangoDB graph settings
    graph_name: str = Field(default="memory_graph", description="Graph name")
    entity_collection: str = Field(default="entities", description="Collection name for entities")
    relation_collection: str = Field(default="relationships", description="Collection name for relationships")

    # SSL verification
    verify: bool = Field(default=True, description="Whether to verify SSL certificates")


@provider(provider_type="graph_db", name="arango", settings_class=ArangoProviderSettings)
class ArangoProvider(GraphDBProvider):
    """ArangoDB graph database provider implementation.
    
    This provider interfaces with ArangoDB using the python-arango client,
    mapping entities and relationships to ArangoDB's graph model.
    """

    def __init__(self, name: str = "arango", provider_type: str = "graph_db", settings: Optional[ArangoProviderSettings] = None):
        """Initialize ArangoDB graph database provider.
        Args:
            name: Provider name (default: "arango")
            provider_type: Provider type (default: "graph_db")
            settings: Provider settings (optional ArangoProviderSettings)
        """
        if settings is not None and not isinstance(settings, ArangoProviderSettings):
            raise TypeError("settings must be an instance of ArangoProviderSettings")
        super().__init__(name=name, provider_type=provider_type, settings=settings)
        self._client: Optional[ArangoClient] = None
        self._db: Optional[StandardDatabase] = None
        self._entity_collection: Optional[StandardCollection] = None
        self._relation_collection: Optional[StandardCollection] = None

    @property
    def arango_settings(self) -> ArangoProviderSettings:
        """Get properly typed ArangoDB settings."""
        return cast(ArangoProviderSettings, self.settings)

    def _parse_document_data_strict(self, doc_data: Dict[str, object]) -> Dict[str, object]:
        """Parse ArangoDB document data with strict validation.
        
        Args:
            doc_data: Raw document data from ArangoDB
            
        Returns:
            Validated ArangoDB document data
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields exist
        if "id" not in doc_data:
            raise ValueError("Document data missing required 'id' field")
        if "type" not in doc_data:
            raise ValueError("Document data missing required 'type' field")
        if "attributes" not in doc_data:
            raise ValueError("Document data missing required 'attributes' field")
        if "source" not in doc_data:
            raise ValueError("Document data missing required 'source' field")
        if "importance" not in doc_data:
            raise ValueError("Document data missing required 'importance' field")
        if "last_updated" not in doc_data:
            raise ValueError("Document data missing required 'last_updated' field")
        if "tags" not in doc_data:
            raise ValueError("Document data missing required 'tags' field")

        # Parse attributes strictly
        attributes_raw = doc_data["attributes"]
        attributes = {}

        if not isinstance(attributes_raw, dict):
            raise ValueError(f"Attributes must be a dict, got {type(attributes_raw)}")

        for attr_name, attr_data in attributes_raw.items():
            if not isinstance(attr_data, dict):
                raise ValueError(f"Attribute '{attr_name}' must be a dict, got {type(attr_data)}")

            # Validate all required attribute fields
            required_fields = ["value", "confidence", "source", "timestamp"]
            for field in required_fields:
                if field not in attr_data:
                    raise ValueError(f"Attribute '{attr_name}' missing required field '{field}'")

            attributes[attr_name] = attr_data

        return {
            "id": doc_data["id"],
            "type": doc_data["type"],
            "attributes": attributes,
            "tags": doc_data["tags"],
            "source": doc_data["source"],
            "importance": doc_data["importance"],
            "last_updated": doc_data["last_updated"]
        }

    async def initialize(self) -> None:
        """Initialize the ArangoDB connection.
        
        Creates the ArangoDB client instance and verifies the connection.
        Also ensures that required collections, indexes, and graph are created.
        
        Raises:
            ProviderError: If ArangoDB driver is not available or connection fails
        """
        settings = cast(ArangoProviderSettings, self.settings)

        try:
            # Create the ArangoDB client
            self._client = ArangoClient(
                hosts=settings.url,
                verify_override=settings.verify
            )

            # Connect to system database first to ensure target database exists
            sys_db = self._client.db(
                "_system",
                username=settings.username,
                password=settings.password
            )

            # Create database if it doesn't exist
            if not sys_db.has_database(settings.database):
                logger.info(f"Creating database '{settings.database}'")
                sys_db.create_database(settings.database)

            # Connect to the target database
            self._db = self._client.db(
                settings.database,
                username=settings.username,
                password=settings.password
            )

            # Create collections if they don't exist
            if not self._db.has_collection(settings.entity_collection):
                logger.info(f"Creating entity collection '{settings.entity_collection}'")
                self._db.create_collection(settings.entity_collection)

            if not self._db.has_collection(settings.relation_collection):
                logger.info(f"Creating relationship collection '{settings.relation_collection}'")
                self._db.create_collection(settings.relation_collection, edge=True)

            # Get collection references
            self._entity_collection = self._db.collection(settings.entity_collection)
            self._relation_collection = self._db.collection(settings.relation_collection)

            # Create graph if it doesn't exist
            if not self._db.has_graph(settings.graph_name):
                logger.info(f"Creating graph '{settings.graph_name}'")
                self._db.create_graph(
                    settings.graph_name,
                    edge_definitions=[
                        {
                            "edge_collection": settings.relation_collection,
                            "from_vertex_collections": [settings.entity_collection],
                            "to_vertex_collections": [settings.entity_collection]
                        }
                    ]
                )

            # Create indexes
            self._setup_indexes()

            self._initialized = True
            logger.info(f"ArangoDB provider '{self.name}' initialized successfully")

        except ArangoError as e:
            raise ProviderError(
                message=f"Failed to connect to ArangoDB: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="ConnectionError",
                    error_location="initialize",
                    component=self.name,
                    operation="connect_to_arango"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="initialize",
                    retry_count=0
                ),
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to initialize ArangoDB provider: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="InitializationError",
                    error_location="initialize",
                    component=self.name,
                    operation="initialize_provider"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="initialize",
                    retry_count=0
                ),
                cause=e
            )

    def _setup_indexes(self) -> None:
        """Set up indexes for entity and relationship collections."""
        if self._entity_collection is None:
            raise RuntimeError("Entity collection not initialized")
        if self._relation_collection is None:
            raise RuntimeError("Relation collection not initialized")

        # Create index on entity ID
        self._entity_collection.add_hash_index(["id"], unique=True)

        # Create index on entity type
        self._entity_collection.add_hash_index(["type"], unique=False)

        # Create index on relationship type
        self._relation_collection.add_hash_index(["relation_type"], unique=False)

    async def shutdown(self) -> None:
        """Shut down the ArangoDB connection."""
        self._client = None
        self._db = None
        self._entity_collection = None
        self._relation_collection = None
        self._initialized = False
        logger.info(f"ArangoDB provider '{self.name}' shut down")

    async def add_entity(self, entity: Entity) -> GraphStoreResult:
        """Add or update an entity node in ArangoDB.
        Args:
            entity: Entity to add or update (must be Entity model)
        Returns:
            GraphStoreResult with operation details
        Raises:
            ProviderError: If entity creation fails
        """
        if not self._initialized:
            raise ProviderError(
                message="ArangoDB provider not initialized",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="NotInitializedError",
                    error_location="add_entity",
                    component=self.name,
                    operation="check_initialization"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="add_entity",
                    retry_count=0
                )
            )
        try:
            doc = self._entity_to_document(entity)
            existing = None
            try:
                if self._entity_collection is None:
                    raise RuntimeError("Entity collection not initialized")
                existing = self._entity_collection.get({"id": entity.id})
            except Exception:
                pass
            if existing:
                if self._entity_collection is None:
                    raise RuntimeError("Entity collection not initialized")
                self._entity_collection.update_match({"id": entity.id}, doc)
            else:
                if self._entity_collection is None:
                    raise RuntimeError("Entity collection not initialized")
                self._entity_collection.insert(doc)
            for rel in entity.relationships:
                await self.add_relationship(
                    entity.id,
                    rel.target_entity,
                    rel.relation_type,
                    rel
                )
            return GraphStoreResult(
                success=True,
                stored_entities=[entity.id],
                stored_relationships=[],
                failed_entities=[],
                failed_relationships=[],
                error_details={},
                execution_time_ms=None
            )
        except ArangoError as e:
            raise ProviderError(
                message=f"Failed to add entity: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="EntityAddError",
                    error_location="add_entity",
                    component=self.name,
                    operation="add_entity"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="add_entity",
                    retry_count=0
                ),
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to add entity: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="EntityAddError",
                    error_location="add_entity",
                    component=self.name,
                    operation="add_entity"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="add_entity",
                    retry_count=0
                ),
                cause=e
            )

    def _entity_to_document(self, entity: Entity) -> Dict[str, object]:
        """Convert Entity object to ArangoDB document format."""
        # Convert attributes to serializable format
        attributes = {}
        for attr_name, attr in entity.attributes.items():
            attributes[attr_name] = {
                "name": attr.name,
                "value": attr.value,
                "confidence": attr.confidence,
                "source": attr.source,
                "timestamp": attr.timestamp
            }

        # Core entity properties
        doc = {
            "id": entity.id,
            "type": entity.type,
            "source": entity.source,
            "importance": entity.importance,
            "last_updated": entity.last_updated,
            "attributes": attributes,
            "tags": getattr(entity, 'tags', [])  # Include tags field with fallback to empty list
        }

        return doc

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID from ArangoDB.
        Args:
            entity_id: Unique identifier of the entity
        Returns:
            Entity object if found, None otherwise
        Raises:
            ProviderError: If retrieval fails
        """
        if not self._initialized:
            raise ProviderError(
                message="ArangoDB provider not initialized",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="NotInitializedError",
                    error_location="get_entity",
                    component=self.name,
                    operation="check_initialization"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="get_entity",
                    retry_count=0
                )
            )
        try:
            if self._entity_collection is None:
                raise RuntimeError("Entity collection not initialized")
            doc = self._entity_collection.get({"id": entity_id})
            if not doc:
                return None

            # Ensure doc is a dict, not an AsyncJob or BatchJob
            if not isinstance(doc, dict):
                raise ProviderError(
                    message=f"Unexpected document type from ArangoDB: {type(doc)}",
                    context=ErrorContext.create(
                        flow_name="arango_provider",
                        error_type="DataError",
                        error_location="get_entity",
                        component=self.name,
                        operation="get_document"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="graph_db",
                        operation="get_entity",
                        retry_count=0
                    )
                )
            relationships = []
            query = f"""
            FOR r IN {self.arango_settings.relation_collection}
            FILTER r._from == @from_key
            LET target = DOCUMENT(r._to)
            RETURN {{
                relation_type: r.relation_type,
                target_id: target.id,
                confidence: r.confidence,
                source: r.source,
                timestamp: r.timestamp
            }}
            """
            from_key = f"{self.arango_settings.entity_collection}/{self._get_doc_key(entity_id)}"
            if self._db is None:
                raise RuntimeError("Database not initialized")
            cursor = self._db.aql.execute(
                query,
                bind_vars={"from_key": from_key}
            )

            # Handle different cursor types - no fallbacks
            if cursor is None:
                raise RuntimeError("Query returned None cursor")

            # Type guard: only accept actual Cursor objects, not async/batch jobs
            from arango.cursor import Cursor
            if not isinstance(cursor, Cursor):
                raise RuntimeError("Async/batch jobs not supported - cursor must be synchronous")

            for rel in cursor:
                # Validate required relationship fields - no fallbacks
                if "target_id" not in rel or not rel["target_id"]:
                    continue  # Skip invalid relationships
                if "relation_type" not in rel:
                    raise ValueError("Relationship missing required 'relation_type' field")
                if "confidence" not in rel:
                    raise ValueError("Relationship missing required 'confidence' field")
                if "source" not in rel:
                    raise ValueError("Relationship missing required 'source' field")
                if "timestamp" not in rel:
                    raise ValueError("Relationship missing required 'timestamp' field")

                relationships.append(
                    EntityRelationship(
                        relation_type=rel["relation_type"],
                        target_entity=rel["target_id"],
                        confidence=rel["confidence"],
                        source=rel["source"],
                        timestamp=rel["timestamp"]
                    )
                )
            # Use strict parser for document data - no fallbacks
            arango_doc = self._parse_document_data_strict(doc)

            # Convert Arango attributes to Entity attributes
            entity_attributes = {}
            attributes_data = arango_doc["attributes"]
            if not isinstance(attributes_data, dict):
                raise ValueError(f"Expected dict for attributes, got {type(attributes_data)}")
            for attr_name, arango_attr in attributes_data.items():
                if not isinstance(arango_attr, dict):
                    raise ValueError(f"Expected dict for attribute {attr_name}, got {type(arango_attr)}")
                entity_attributes[attr_name] = EntityAttribute(
                    name=attr_name,
                    value=arango_attr["value"],
                    confidence=arango_attr["confidence"],
                    source=arango_attr["source"],
                    timestamp=arango_attr["timestamp"]
                )

            # Type guard for tags conversion
            tags_raw = arango_doc.get("tags", [])
            if isinstance(tags_raw, (list, tuple)):
                tags_list = list(tags_raw)
            else:
                tags_list = []

            entity = Entity(
                id=str(arango_doc["id"]),
                type=str(arango_doc["type"]),
                attributes=entity_attributes,
                relationships=relationships,
                tags=tags_list,
                source=str(arango_doc["source"]),
                importance=float(arango_doc["importance"]) if arango_doc["importance"] is not None and isinstance(arango_doc["importance"], (int, float, str)) else 0.7,
                vector_id=None,
                last_updated=str(arango_doc["last_updated"])
            )
            return entity
        except ArangoError as e:
            raise ProviderError(
                message=f"Failed to get entity: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="EntityRetrievalError",
                    error_location="get_entity",
                    component=self.name,
                    operation="get_entity"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="get_entity",
                    retry_count=0
                ),
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to get entity: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="EntityRetrievalError",
                    error_location="get_entity",
                    component=self.name,
                    operation="get_entity"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="get_entity",
                    retry_count=0
                ),
                cause=e
            )

    def _get_doc_key(self, entity_id: str) -> str:
        """Convert entity ID to a valid document key by replacing invalid characters."""
        # Replace characters that are not allowed in ArangoDB keys
        return entity_id.replace("/", "_").replace(" ", "_")

    async def add_relationship(
        self,
        source_id: str,
        target_entity: str,
        relation_type: str,
        relationship: EntityRelationship
    ) -> None:
        """Add a relationship between two entities.
        Args:
            source_id: ID of the source entity
            target_entity: ID of the target entity
            relation_type: Type of relationship
            relationship: EntityRelationship model
        Raises:
            ProviderError: If relationship creation fails
        """
        if not self._initialized:
            raise ProviderError(
                message="ArangoDB provider not initialized",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="NotInitializedError",
                    error_location="add_relationship",
                    component=self.name,
                    operation="check_initialization"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="add_relationship",
                    retry_count=0
                )
            )
        try:
            from_key = f"{self.arango_settings.entity_collection}/{self._get_doc_key(source_id)}"
            to_key = f"{self.arango_settings.entity_collection}/{self._get_doc_key(target_entity)}"
            rel_doc = {
                "_from": from_key,
                "_to": to_key,
                "relation_type": relation_type,
                "confidence": relationship.confidence,
                "source": relationship.source,
                "timestamp": relationship.timestamp
            }
            if self._relation_collection is None:
                raise RuntimeError("Relation collection not initialized")
            self._relation_collection.insert(rel_doc)
        except ArangoError as e:
            raise ProviderError(
                message=f"Failed to add relationship: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="RelationshipAddError",
                    error_location="add_relationship",
                    component=self.name,
                    operation="add_relationship"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="add_relationship",
                    retry_count=0
                ),
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to add relationship: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="RelationshipAddError",
                    error_location="add_relationship",
                    component=self.name,
                    operation="add_relationship"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="add_relationship",
                    retry_count=0
                ),
                cause=e
            )

    async def query_relationships(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        direction: str = "outgoing"
    ) -> RelationshipSearchResult:
        """Query relationships for an entity in ArangoDB.
        
        Args:
            entity_id: ID of the entity
            relation_type: Optional type to filter by
            direction: 'outgoing' or 'incoming'
            
        Returns:
            Structured result with relationships and metadata
            
        Raises:
            ProviderError: If query fails
        """
        if not self._initialized:
            raise ProviderError(
                message="ArangoDB provider not initialized",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="NotInitializedError",
                    error_location="query_relationships",
                    component=self.name,
                    operation="check_initialization"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="query_relationships",
                    retry_count=0
                )
            )

        try:
            start_time = datetime.now()

            # Get document key from entity ID
            if self._entity_collection is None:
                raise RuntimeError("Entity collection not initialized")
            doc = self._entity_collection.get({"id": entity_id})
            if not doc:
                return RelationshipSearchResult(
                    success=True,
                    relationships=[],
                    total_count=0,
                    source_entity=entity_id if direction == "outgoing" else None,
                    target_entity=entity_id if direction == "incoming" else None,
                    execution_time_ms=0.0,
                    metadata={"direction": direction, "relation_type": relation_type}
                )

            # Ensure doc is a dict (not AsyncJob or BatchJob)
            if not isinstance(doc, dict):
                raise ProviderError(
                    message=f"Unexpected document type from ArangoDB: {type(doc)}",
                    context=ErrorContext.create(
                        flow_name="arango_provider",
                        error_type="DataError",
                        error_location="query_relationships",
                        component=self.name,
                        operation="get_document"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="graph_db",
                        operation="query_relationships",
                        retry_count=0
                    )
                )

            entity_key = doc["_key"]
            collection_name = self.arango_settings.entity_collection

            # Construct AQL query based on direction
            if direction == "outgoing":
                query = f"""
                FOR v, e IN 1..1 OUTBOUND '{collection_name}/{entity_key}' {self.arango_settings.relation_collection}
                FILTER @relation_type IS NULL OR e.relation_type == @relation_type
                RETURN {{
                    source: '{entity_id}',
                    target: v.id,
                    type: e.relation_type,
                    properties: {{
                        confidence: e.confidence,
                        source: e.source,
                        timestamp: e.timestamp
                    }}
                }}
                """
            else:  # incoming
                query = f"""
                FOR v, e IN 1..1 INBOUND '{collection_name}/{entity_key}' {self.arango_settings.relation_collection}
                FILTER @relation_type IS NULL OR e.relation_type == @relation_type
                RETURN {{
                    source: v.id,
                    target: '{entity_id}',
                    type: e.relation_type,
                    properties: {{
                        confidence: e.confidence,
                        source: e.source,
                        timestamp: e.timestamp
                    }}
                }}
                """

            # Execute the query
            if self._db is None:
                raise RuntimeError("Database not initialized")
            cursor = self._db.aql.execute(
                query,
                bind_vars={"relation_type": relation_type}
            )

            # Type guard for cursor
            if cursor is None:
                raise RuntimeError("Query returned None cursor")

            from arango.cursor import Cursor
            if not isinstance(cursor, Cursor):
                raise RuntimeError("Async/batch jobs not supported - cursor must be synchronous")

            # Convert to EntityRelationship objects - strict validation
            relationships = []
            for rel in cursor:
                # Validate required relationship fields
                if "type" not in rel:
                    raise ValueError("Relationship missing required 'type' field")
                if "target" not in rel:
                    raise ValueError("Relationship missing required 'target' field")
                if "properties" not in rel:
                    raise ValueError("Relationship missing required 'properties' field")

                properties = rel["properties"]
                if not isinstance(properties, dict):
                    raise ValueError("Relationship properties must be a dict")

                # Validate required property fields
                if "confidence" not in properties:
                    raise ValueError("Relationship properties missing required 'confidence' field")
                if "source" not in properties:
                    raise ValueError("Relationship properties missing required 'source' field")
                if "timestamp" not in properties:
                    raise ValueError("Relationship properties missing required 'timestamp' field")

                relationships.append(
                    EntityRelationship(
                        relation_type=rel["type"],
                        target_entity=rel["target"],
                        confidence=properties["confidence"],
                        source=properties["source"],
                        timestamp=properties["timestamp"]
                    )
                )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return RelationshipSearchResult(
                success=True,
                relationships=relationships,
                total_count=len(relationships),
                source_entity=entity_id if direction == "outgoing" else None,
                target_entity=entity_id if direction == "incoming" else None,
                execution_time_ms=execution_time,
                metadata={"direction": direction, "relation_type": relation_type}
            )

        except ArangoError as e:
            raise ProviderError(
                message=f"Failed to query relationships: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="RelationshipQueryError",
                    error_location="query_relationships",
                    component=self.name,
                    operation="query_relationships"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="query_relationships",
                    retry_count=0
                ),
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to query relationships: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="RelationshipQueryError",
                    error_location="query_relationships",
                    component=self.name,
                    operation="query_relationships"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="query_relationships",
                    retry_count=0
                ),
                cause=e
            )

    async def traverse(
        self,
        start_id: str,
        relation_types: Optional[List[str]] = None,
        max_depth: int = 2
    ) -> List[Entity]:
        """Traverse the graph starting from an entity in ArangoDB.
        
        Args:
            start_id: ID of the starting entity
            relation_types: Optional list of relation types to traverse
            max_depth: Maximum traversal depth
            
        Returns:
            List of entities found in traversal
            
        Raises:
            ProviderError: If traversal fails
        """
        if not self._initialized:
            raise ProviderError(
                message="ArangoDB provider not initialized",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="NotInitializedError",
                    error_location="traverse",
                    component=self.name,
                    operation="check_initialization"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="traverse",
                    retry_count=0
                )
            )

        try:
            # Get document key from entity ID
            if self._entity_collection is None:
                raise RuntimeError("Entity collection not initialized")
            doc = self._entity_collection.get({"id": start_id})
            if not doc:
                return []

            # Ensure doc is a dict (not AsyncJob or BatchJob)
            if not isinstance(doc, dict):
                raise ProviderError(
                    message=f"Unexpected document type from ArangoDB: {type(doc)}",
                    context=ErrorContext.create(
                        flow_name="arango_provider",
                        error_type="DataError",
                        error_location="traverse",
                        component=self.name,
                        operation="get_document"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="graph_db",
                        operation="traverse",
                        retry_count=0
                    )
                )

            entity_key = doc["_key"]
            collection_name = self.arango_settings.entity_collection

            # Build relation type filter
            filter_clause = ""
            if relation_types:
                relation_list = [f'"{rel_type}"' for rel_type in relation_types]
                filter_clause = f"FILTER e.relation_type IN [{', '.join(relation_list)}]"

            # Construct AQL traversal query
            query = f"""
            FOR v, e, p IN 1..{max_depth} OUTBOUND '{collection_name}/{entity_key}' {self.arango_settings.relation_collection}
            {filter_clause}
            RETURN DISTINCT v.id
            """

            # Execute the query
            if self._db is None:
                raise RuntimeError("Database not initialized")
            cursor = self._db.aql.execute(query)

            # Type guard for cursor
            if cursor is None:
                raise RuntimeError("Query returned None cursor")

            from arango.cursor import Cursor
            if not isinstance(cursor, Cursor):
                raise RuntimeError("Async/batch jobs not supported - cursor must be synchronous")

            # Get the full entity for each ID
            entities = []
            ids = set()
            for result in cursor:
                if result not in ids:
                    ids.add(result)
                    entity = await self.get_entity(result)
                    if entity:
                        entities.append(entity)

            # Add start entity if not already included
            start_entity = await self.get_entity(start_id)
            if start_entity and start_id not in ids:
                entities.insert(0, start_entity)

            return entities

        except ArangoError as e:
            raise ProviderError(
                message=f"Failed to traverse graph: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="GraphTraversalError",
                    error_location="traverse",
                    component=self.name,
                    operation="traverse"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="traverse",
                    retry_count=0
                ),
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to traverse graph: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="GraphTraversalError",
                    error_location="traverse",
                    component=self.name,
                    operation="traverse"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="traverse",
                    retry_count=0
                ),
                cause=e
            )

    async def query(
        self,
        query: str,
        params: Optional[object] = None
    ) -> GraphQueryResult:
        """Execute a simple query language for ArangoDB.
        Args:
            query: Query string
            params: Optional query parameters (should be a Pydantic model or None)
        Returns:
            GraphQueryResult
        Raises:
            ProviderError: If query parsing fails
        """
        if not self._initialized:
            raise ProviderError(
                message="ArangoDB provider not initialized",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="NotInitializedError",
                    error_location="query",
                    component=self.name,
                    operation="check_initialization"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="query",
                    retry_count=0
                )
            )
        try:
            start_time = datetime.now()
            query_lower = query.strip().lower()
            if params and hasattr(params, 'model_dump'):
                param_dict = params.model_dump()
            elif params:
                # Type guard for dict conversion
                if isinstance(params, (dict, Mapping)):
                    param_dict = dict(params)
                else:
                    raise RuntimeError(f"Unsupported params type: {type(params)}. Expected dict, Mapping or Pydantic model")
            else:
                param_parts = [part.strip() for part in query_lower.split(" ") if "=" in part]
                param_dict = {}
                for part in param_parts:
                    key, value = part.split("=", 1)
                    param_dict[key] = value
            raw_results = None
            if query_lower.startswith("find_entities") and "type" in param_dict:
                entity_type = param_dict["type"]
                raw_results = await self._find_entities_by_type(entity_type)
            elif query_lower.startswith("find_entities") and "name" in param_dict:
                name = param_dict["name"]
                raw_results = await self._find_entities_by_name(name)
            elif query_lower.startswith("neighbors") and "id" in param_dict:
                entity_id = param_dict["id"]
                relation_type = param_dict["relation"] if "relation" in param_dict else None
                raw_results = await self._find_neighbors(entity_id, relation_type)
            elif query_lower.startswith("path") and "from" in param_dict and "to" in param_dict:
                from_id = param_dict["from"]
                to_id = param_dict["to"]
                max_depth = int(param_dict["max_depth"]) if "max_depth" in param_dict else 3
                raw_results = await self._find_path(from_id, to_id, max_depth)
            elif query_lower.startswith("aql") and "aql_query" in param_dict:
                aql_query = param_dict["aql_query"]
                aql_params = param_dict["params"] if "params" in param_dict else {}
                raw_results = self._execute_aql(aql_query, aql_params)
            else:
                raise ProviderError(
                    message=f"Unsupported query: {query}",
                    context=ErrorContext.create(
                        flow_name="arango_provider",
                        error_type="UnsupportedQueryError",
                        error_location="query",
                        component=self.name,
                        operation="parse_query"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="graph_db",
                        operation="query",
                        retry_count=0
                    )
                )
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            nodes = []
            edges = []
            if raw_results:
                for result in raw_results:
                    if isinstance(result, dict) and "entity" in result:
                        entity_data = result["entity"]
                        if not isinstance(entity_data, dict):
                            continue  # Skip invalid entity data
                        entity = self._document_to_entity(entity_data)
                        if entity:
                            nodes.append(entity.model_dump())
                    elif isinstance(result, dict) and "relation" in result:
                        edges.append({"type": result["relation"]})
                    elif isinstance(result, dict):
                        entity = self._document_to_entity(result)
                        if entity:
                            nodes.append(entity.model_dump())
            return GraphQueryResult(
                success=True,
                nodes=nodes,
                edges=edges,
                execution_time_ms=execution_time,
                total_count=len(nodes) + len(edges),
                metadata={"query": query, "params": param_dict}
            )
        except ArangoError as e:
            raise ProviderError(
                message=f"Failed to execute query: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="QueryExecutionError",
                    error_location="query",
                    component=self.name,
                    operation="execute_query"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="query",
                    retry_count=0
                ),
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute query: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="QueryExecutionError",
                    error_location="query",
                    component=self.name,
                    operation="execute_query"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="query",
                    retry_count=0
                ),
                cause=e
            )

    def _execute_aql(self, query: str, params: Optional[Dict[str, object]] = None) -> List[Dict[str, object]]:
        """Execute an AQL query with parameters."""
        if self._db is None:
            raise RuntimeError("Database not initialized")
        from typing import MutableMapping, cast
        bind_vars = cast(MutableMapping[str, Any], params or {})
        cursor = self._db.aql.execute(query, bind_vars=bind_vars)

        # Type guard for cursor
        if cursor is None:
            raise RuntimeError("Query returned None cursor")

        from arango.cursor import Cursor
        if not isinstance(cursor, Cursor):
            raise RuntimeError("Async/batch jobs not supported - cursor must be synchronous")

        return [doc for doc in cursor]

    async def _find_entities_by_type(self, entity_type: str) -> List[Dict[str, object]]:
        """Find entities by type in ArangoDB."""
        query = f"""
        FOR e IN {self.arango_settings.entity_collection}
        FILTER e.type == @type
        RETURN {{ id: e.id, entity: e }}
        """

        if self._db is None:
            raise RuntimeError("Database not initialized")
        cursor = self._db.aql.execute(query, bind_vars={"type": entity_type})

        # Type guard for cursor
        if cursor is None:
            raise RuntimeError("Query returned None cursor")

        from arango.cursor import Cursor
        if not isinstance(cursor, Cursor):
            raise RuntimeError("Async/batch jobs not supported - cursor must be synchronous")

        return [doc for doc in cursor]

    async def _find_entities_by_name(self, name: str) -> List[Dict[str, object]]:
        """Find entities by name in ArangoDB."""
        query = f"""
        FOR e IN {self.arango_settings.entity_collection}
        FILTER e.attributes.name.value LIKE @name
        RETURN {{ id: e.id, entity: e }}
        """

        if self._db is None:
            raise RuntimeError("Database not initialized")
        cursor = self._db.aql.execute(query, bind_vars={"name": f"%{name}%"})

        # Type guard for cursor
        if cursor is None:
            raise RuntimeError("Query returned None cursor")

        from arango.cursor import Cursor
        if not isinstance(cursor, Cursor):
            raise RuntimeError("Async/batch jobs not supported - cursor must be synchronous")

        return [doc for doc in cursor]

    async def _find_neighbors(self, entity_id: str, relation_type: Optional[str]) -> List[Dict[str, object]]:
        """Find neighboring entities in ArangoDB."""
        # Get document key from entity ID
        if self._entity_collection is None:
            raise RuntimeError("Entity collection not initialized")
        doc = self._entity_collection.get({"id": entity_id})
        if not doc:
            return []

        # Ensure doc is a dict (not AsyncJob or BatchJob)
        if not isinstance(doc, dict):
            raise ProviderError(
                message=f"Unexpected document type from ArangoDB: {type(doc)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="DataError",
                    error_location="get_neighbors",
                    component=self.name,
                    operation="get_document"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="get_neighbors",
                    retry_count=0
                )
            )

        entity_key = doc["_key"]
        collection_name = self.arango_settings.entity_collection

        # Build relation type filter
        filter_clause = ""
        if relation_type:
            filter_clause = f"FILTER e.relation_type == '{relation_type}'"

        query = f"""
        FOR v, e IN 1..1 OUTBOUND '{collection_name}/{entity_key}' {self.arango_settings.relation_collection}
        {filter_clause}
        RETURN {{
            id: v.id,
            relation: e.relation_type,
            entity: v
        }}
        """

        if self._db is None:
            raise RuntimeError("Database not initialized")
        cursor = self._db.aql.execute(query)

        # Type guard for cursor
        if cursor is None:
            raise RuntimeError("Query returned None cursor")

        from arango.cursor import Cursor
        if not isinstance(cursor, Cursor):
            raise RuntimeError("Async/batch jobs not supported - cursor must be synchronous")

        return [doc for doc in cursor]

    async def _find_path(self, from_id: str, to_id: str, max_depth: int) -> List[Dict[str, object]]:
        """Find path between entities in ArangoDB."""
        # Get document keys
        if self._entity_collection is None:
            raise RuntimeError("Entity collection not initialized")
        from_doc = self._entity_collection.get({"id": from_id})
        to_doc = self._entity_collection.get({"id": to_id})

        if not from_doc or not to_doc:
            return []

        # Type guard: ensure documents are dicts, not async jobs
        if not isinstance(from_doc, dict) or not isinstance(to_doc, dict):
            raise RuntimeError("Async/batch jobs not supported for document retrieval")

        from_key = from_doc["_key"]
        to_key = to_doc["_key"]
        collection_name = self.arango_settings.entity_collection

        query = f"""
        LET path = (
            FOR v, e, p IN 1..{max_depth} OUTBOUND 
            '{collection_name}/{from_key}' {self.arango_settings.relation_collection}
            FILTER v._key == '{to_key}'
            LIMIT 1
            RETURN p.vertices
        )
        
        RETURN LENGTH(path) > 0 ? (
            FOR vertex IN path[0]
            LET index = POSITION(path[0], vertex)
            RETURN {{
                position: index,
                id: vertex.id,
                entity: vertex
            }}
        ) : []
        """

        if self._db is None:
            raise RuntimeError("Database not initialized")
        cursor = self._db.aql.execute(query)

        # Type guard for cursor
        if cursor is None:
            raise RuntimeError("Query returned None cursor")

        from arango.cursor import Cursor
        if not isinstance(cursor, Cursor):
            raise RuntimeError("Async/batch jobs not supported - cursor must be synchronous")

        result = cursor.next()
        return result if result else []

    async def delete_entity(self, entity_id: str) -> GraphDeleteResult:
        """Delete an entity and its relationships from ArangoDB.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            Result of the delete operation
            
        Raises:
            ProviderError: If deletion fails
        """
        if not self._initialized:
            raise ProviderError(
                message="ArangoDB provider not initialized",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="NotInitializedError",
                    error_location="delete_entity",
                    component=self.name,
                    operation="check_initialization"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="delete_entity",
                    retry_count=0
                )
            )

        try:
            start_time = datetime.now()

            # Get document key from entity ID
            if self._entity_collection is None:
                raise RuntimeError("Entity collection not initialized")
            doc = self._entity_collection.get({"id": entity_id})
            if not doc:
                return GraphDeleteResult(
                    success=False,
                    deleted_entities=[],
                    deleted_relationships=[],
                    not_found_entities=[entity_id],
                    not_found_relationships=[],
                    error_details={},
                    execution_time_ms=0.0
                )

            # Type guard: ensure document is dict, not async job
            if not isinstance(doc, dict):
                raise RuntimeError("Async/batch jobs not supported for document retrieval")

            entity_key = doc["_key"]

            # Delete entity and related edges via AQL
            query = f"""
            LET vertex = DOCUMENT('{self.arango_settings.entity_collection}/{entity_key}')
            LET edges = (
                FOR v, e IN 1..1 ANY vertex {self.arango_settings.relation_collection}
                RETURN e._key
            )
            
            FOR edge_key IN edges
                REMOVE edge_key IN {self.arango_settings.relation_collection}
                
            REMOVE vertex IN {self.arango_settings.entity_collection}
            RETURN true
            """

            if self._db is None:
                raise RuntimeError("Database not initialized")
            self._db.aql.execute(query)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return GraphDeleteResult(
                success=True,
                deleted_entities=[entity_id],
                deleted_relationships=[],  # We don't track specific relationship IDs
                not_found_entities=[],
                not_found_relationships=[],
                error_details={},
                execution_time_ms=execution_time
            )

        except ArangoError as e:
            raise ProviderError(
                message=f"Failed to delete entity: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="EntityDeleteError",
                    error_location="delete_entity",
                    component=self.name,
                    operation="delete_entity"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="delete_entity",
                    retry_count=0
                ),
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete entity: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="EntityDeleteError",
                    error_location="delete_entity",
                    component=self.name,
                    operation="delete_entity"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="delete_entity",
                    retry_count=0
                ),
                cause=e
            )

    async def delete_relationship(
        self,
        source_id: str,
        target_entity: str,
        relation_type: Optional[str] = None
    ) -> bool:
        """Delete relationship(s) between entities in ArangoDB.
        
        Args:
            source_id: ID of the source entity
            target_entity: ID of the target entity
            relation_type: Optional type to filter by
            
        Returns:
            True if relationships were deleted, False if none found
            
        Raises:
            ProviderError: If deletion fails
        """
        if not self._initialized:
            raise ProviderError(
                message="ArangoDB provider not initialized",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="NotInitializedError",
                    error_location="delete_relationship",
                    component=self.name,
                    operation="check_initialization"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="delete_relationship",
                    retry_count=0
                )
            )

        try:
            # Get document keys
            if self._entity_collection is None:
                raise RuntimeError("Entity collection not initialized")
            source_doc = self._entity_collection.get({"id": source_id})
            target_doc = self._entity_collection.get({"id": target_entity})

            if not source_doc or not target_doc:
                return False

            # Type guard: ensure documents are dicts, not async jobs
            if not isinstance(source_doc, dict) or not isinstance(target_doc, dict):
                raise RuntimeError("Async/batch jobs not supported for document retrieval")

            source_key = source_doc["_key"]
            target_key = target_doc["_key"]
            collection_name = self.arango_settings.entity_collection

            # Build relation type filter
            type_filter = ""
            if relation_type:
                type_filter = f"FILTER e.relation_type == '{relation_type}'"

            # Delete edges via AQL
            query = f"""
            FOR e IN {self.arango_settings.relation_collection}
            FILTER e._from == '{collection_name}/{source_key}' AND e._to == '{collection_name}/{target_key}'
            {type_filter}
            REMOVE e IN {self.arango_settings.relation_collection}
            COLLECT WITH COUNT INTO deleted
            RETURN deleted
            """

            if self._db is None:
                raise RuntimeError("Database not initialized")
            cursor = self._db.aql.execute(query)

            # Type guard for cursor
            if cursor is None:
                raise RuntimeError("Query returned None cursor")

            from arango.cursor import Cursor
            if not isinstance(cursor, Cursor):
                raise RuntimeError("Async/batch jobs not supported - cursor must be synchronous")

            deleted = cursor.next()

            # Ensure we return a boolean by explicitly converting
            return bool(deleted > 0)

        except ArangoError as e:
            raise ProviderError(
                message=f"Failed to delete relationship: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="RelationshipDeleteError",
                    error_location="delete_relationship",
                    component=self.name,
                    operation="delete_relationship"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="delete_relationship",
                    retry_count=0
                ),
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete relationship: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="RelationshipDeleteError",
                    error_location="delete_relationship",
                    component=self.name,
                    operation="delete_relationship"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="delete_relationship",
                    retry_count=0
                ),
                cause=e
            )

    async def remove_relationship(
        self,
        source_id: str,
        target_entity: str,
        relation_type: str
    ) -> None:
        """Remove a relationship between two entities in ArangoDB.
        
        Args:
            source_id: ID of the source entity
            target_entity: ID of the target entity
            relation_type: Type of relationship
            
        Raises:
            ProviderError: If relationship removal fails
        """
        try:
            result = await self.delete_relationship(source_id, target_entity, relation_type)
            if not result:
                logger.warning(
                    f"No relationship found to remove: {source_id} -> {relation_type} -> {target_entity}"
                )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to remove relationship: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="RelationshipRemovalError",
                    error_location="remove_relationship",
                    component=self.name,
                    operation="remove_relationship"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="remove_relationship",
                    retry_count=0
                ),
                cause=e
            )

    async def bulk_add_entities(self, entities: List[Entity]) -> GraphStoreResult:
        """Bulk add entities to ArangoDB.
        Args:
            entities: List of Entity models
        Returns:
            GraphStoreResult
        Raises:
            ProviderError: If bulk operation fails
        """
        if not self._initialized:
            raise ProviderError(
                message="ArangoDB provider not initialized",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="NotInitializedError",
                    error_location="bulk_add_entities",
                    component=self.name,
                    operation="check_initialization"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="bulk_add_entities",
                    retry_count=0
                )
            )
        try:
            stored_entities = []
            failed_entities = []
            for entity in entities:
                try:
                    result = await self.add_entity(entity)
                    if result.success:
                        stored_entities.extend(result.stored_entities)
                    else:
                        failed_entities.append(entity.id)
                except Exception:
                    failed_entities.append(entity.id)
            return GraphStoreResult(
                success=len(failed_entities) == 0,
                stored_entities=stored_entities,
                failed_entities=failed_entities,
                stored_relationships=[],
                failed_relationships=[],
                error_details={"failed_entities": failed_entities} if failed_entities else {},
                execution_time_ms=None
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to bulk add entities: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="BulkAddError",
                    error_location="bulk_add_entities",
                    component=self.name,
                    operation="bulk_add_entities"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="bulk_add_entities",
                    retry_count=0
                ),
                cause=e
            )

    async def search_entities(
        self,
        query: Optional[str] = None,
        entity_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10
    ) -> EntitySearchResult:
        """Search for entities in ArangoDB based on criteria.
        
        Args:
            query: Optional text query to match against entity ID or attributes.
            entity_type: Optional entity type to filter by.
            tags: Optional list of tags to filter by.
            limit: Maximum number of entities to return.
            
        Returns:
            EntitySearchResult with matching entities and metadata.
            
        Raises:
            ProviderError: If the search operation fails.
        """
        if not self._initialized:
            raise ProviderError(
                message="ArangoDB provider not initialized",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="NotInitializedError",
                    error_location="search_entities",
                    component=self.name,
                    operation="check_initialization"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="search_entities",
                    retry_count=0
                )
            )

        try:
            start_time = datetime.now()

            # Build AQL query
            query_parts = [f"FOR e IN {self.arango_settings.entity_collection}"]
            filters = []
            bind_vars: Dict[str, Union[str, List[str]]] = {}

            if query and query.strip():
                filters.append("(e.id LIKE @query OR e.attributes.name.value LIKE @query)")
                bind_vars["query"] = f"%{query}%"

            if entity_type:
                filters.append("e.type == @entity_type")
                bind_vars["entity_type"] = entity_type

            if tags:
                filters.append("LENGTH(INTERSECTION(e.tags, @tags)) > 0")
                bind_vars["tags"] = tags

            if filters:
                query_parts.append("FILTER " + " AND ".join(filters))

            query_parts.append(f"LIMIT {limit}")
            query_parts.append("RETURN e")

            aql_query = " ".join(query_parts)
            from typing import Any, MutableMapping, cast

            if self._db is None:
                raise RuntimeError("Database not initialized")
            cursor = self._db.aql.execute(aql_query, bind_vars=cast(MutableMapping[str, Any], bind_vars))

            # Type guard for cursor
            if cursor is None:
                raise RuntimeError("Query returned None cursor")

            from arango.cursor import Cursor
            if not isinstance(cursor, Cursor):
                raise RuntimeError("Async/batch jobs not supported - cursor must be synchronous")

            entities = []
            for doc in cursor:
                # Convert document to Entity
                entity = self._document_to_entity(doc)
                if entity:
                    entities.append(entity)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return EntitySearchResult(
                success=True,
                entities=entities,
                total_count=len(entities),
                search_query=query or "",
                execution_time_ms=execution_time,
                metadata={
                    "entity_type": entity_type,
                    "tags": tags,
                    "limit": limit
                }
            )

        except ArangoError as e:
            raise ProviderError(
                message=f"Failed to search entities in ArangoDB: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="EntitySearchError",
                    error_location="search_entities",
                    component=self.name,
                    operation="search_entities"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="search_entities",
                    retry_count=0
                ),
                cause=e
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to search entities: {str(e)}",
                context=ErrorContext.create(
                    flow_name="arango_provider",
                    error_type="EntitySearchError",
                    error_location="search_entities",
                    component=self.name,
                    operation="search_entities"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="search_entities",
                    retry_count=0
                ),
                cause=e
            )

    def _document_to_entity(self, doc: Dict[str, object]) -> Optional[Entity]:
        """Convert ArangoDB document to Entity object using strict parsing."""
        try:
            # Use strict parser for all document data - no fallbacks
            arango_doc = self._parse_document_data_strict(doc)

            # Convert Arango attributes to Entity attributes
            entity_attributes = {}
            attributes_data = arango_doc["attributes"]
            if not isinstance(attributes_data, dict):
                raise ValueError(f"Expected dict for attributes, got {type(attributes_data)}")
            for attr_name, arango_attr in attributes_data.items():
                if not isinstance(arango_attr, dict):
                    raise ValueError(f"Expected dict for attribute {attr_name}, got {type(arango_attr)}")
                entity_attributes[attr_name] = EntityAttribute(
                    name=attr_name,
                    value=arango_attr["value"],
                    confidence=arango_attr["confidence"],
                    source=arango_attr["source"],
                    timestamp=arango_attr["timestamp"]
                )

            # Note: relationships are not included in document search
            # They would need to be fetched separately if needed

            # Validate and convert field types
            entity_id = arango_doc["id"]
            if not isinstance(entity_id, str):
                raise ValueError(f"Expected str for id, got {type(entity_id)}")

            entity_type = arango_doc["type"]
            if not isinstance(entity_type, str):
                raise ValueError(f"Expected str for type, got {type(entity_type)}")

            entity_tags = arango_doc["tags"]
            if not isinstance(entity_tags, list):
                raise ValueError(f"Expected list for tags, got {type(entity_tags)}")

            entity_source = arango_doc["source"]
            if not isinstance(entity_source, str):
                raise ValueError(f"Expected str for source, got {type(entity_source)}")

            entity = Entity(
                id=entity_id,
                type=entity_type,
                attributes=entity_attributes,
                relationships=[],  # Empty for search results
                tags=entity_tags,
                source=entity_source,
                importance=float(arango_doc["importance"]) if arango_doc["importance"] is not None and isinstance(arango_doc["importance"], (int, float, str)) else 0.7,
                vector_id=None,
                last_updated=str(arango_doc["last_updated"])
            )

            return entity
        except Exception as e:
            logger.error(f"Failed to convert document to entity: {e}")
            return None
