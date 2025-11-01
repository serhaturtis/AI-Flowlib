"""JanusGraph database provider implementation.

This module provides a concrete implementation of the GraphDBProvider
for JanusGraph, a distributed graph database based on Apache TinkerPop.
"""

import json
import logging
from datetime import datetime
from typing import Any, cast

from pydantic import Field

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.decorators import provider

# Removed ProviderType import - using config-driven provider access
from flowlib.providers.graph.base import GraphDBProvider, GraphDBProviderSettings
from flowlib.providers.graph.janus.models import JanusAttributeData, JanusEntityData
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


# Define default classes first (single source of truth)
class _DefaultDriverRemoteConnection:
    """Default DriverRemoteConnection stub."""

    pass


class _DefaultClient:
    """Default Client stub."""

    pass


class _DefaultGraphBinaryMessageSerializer:
    """Default GraphBinaryMessageSerializer stub."""

    pass


def _default_traversal() -> None:
    """Default traversal stub."""
    return None


class _DefaultTraversalStep:
    """Default traversal step stub."""

    pass


# Define dummy models for type annotations when gremlin-python is not installed
try:
    from gremlin_python.driver.client import Client  # type: ignore[import-untyped]
    from gremlin_python.driver.driver_remote_connection import (
        DriverRemoteConnection,  # type: ignore[import-untyped]
    )
    from gremlin_python.driver.protocol.graph_binary_message_serializer import (
        GraphBinaryMessageSerializer,  # type: ignore[import-untyped]
    )
    from gremlin_python.process.anonymous_traversal import (
        traversal,  # type: ignore[import-untyped]
    )
    from gremlin_python.process.graph_traversal import (
        __,  # type: ignore[import-untyped]
    )

    JANUS_AVAILABLE = True
except ImportError:
    logger.warning("gremlin-python package not found. Install with 'pip install gremlinpython'")
    JANUS_AVAILABLE = False

    # Assign default classes
    DriverRemoteConnection = _DefaultDriverRemoteConnection
    Client = _DefaultClient
    GraphBinaryMessageSerializer = _DefaultGraphBinaryMessageSerializer
    traversal = _default_traversal
    __ = _DefaultTraversalStep()


class JanusProviderSettings(GraphDBProviderSettings):
    """JanusGraph provider settings - direct inheritance, only JanusGraph-specific fields.

    JanusGraph requires:
    1. Gremlin server WebSocket connection
    2. Optional authentication
    3. Graph traversal configuration
    4. Connection pool and timeout settings
    """

    # JanusGraph connection settings
    url: str = Field(
        default="ws://localhost:8182/gremlin", description="JanusGraph Gremlin server WebSocket URL"
    )
    username: str = Field(
        default="", description="Username for authentication (if using authentication)"
    )
    password: str = Field(
        default="", description="Password for authentication (if using authentication)"
    )

    # Graph settings
    graph_name: str = Field(default="g", description="Name of the graph instance")
    traversal_source: str = Field(default="g", description="Name of the traversal source")

    # Connection settings
    connection_pool_size: int = Field(default=4, description="Size of the connection pool")
    max_batch_size: int = Field(
        default=100, description="Maximum number of entities to process in a single batch"
    )
    message_serializer: str = Field(
        default="graphbinary-1.0", description="Message serializer to use"
    )
    read_timeout: int = Field(default=30, description="Read timeout in seconds")
    write_timeout: int = Field(default=30, description="Write timeout in seconds")
    max_retry_count: int = Field(
        default=3, description="Maximum number of retry attempts for operations"
    )


@provider(provider_type="graph_db", name="janusgraph", settings_class=JanusProviderSettings)
class JanusGraphProvider(GraphDBProvider):
    """JanusGraph graph database provider implementation.

    This provider interfaces with JanusGraph using the Gremlin Python client,
    mapping entities and relationships to JanusGraph's property graph model.
    """

    def __init__(
        self,
        name: str = "janusgraph",
        provider_type: str = "graph_db",
        settings: JanusProviderSettings | None = None,
    ):
        """Initialize JanusGraph graph database provider.

        Args:
            name: Provider name
            provider_type: Provider type
            settings: Provider settings
        """
        # Create settings explicitly if not provided
        if settings is None:
            settings = JanusProviderSettings()

        super().__init__(name=name, provider_type=provider_type, settings=settings)
        self._client: Client | None = None
        self._g = None  # Traversal source

    def _parse_entity_data_strict(self, entity_data: dict[str, Any]) -> JanusEntityData:
        """Parse JanusGraph entity data with strict validation.

        Args:
            entity_data: Raw entity data from JanusGraph

        Returns:
            Validated JanusGraph entity data

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields exist
        if "id" not in entity_data:
            raise ValueError("Entity data missing required 'id' field")
        if "type" not in entity_data:
            raise ValueError("Entity data missing required 'type' field")
        if "attributes" not in entity_data:
            raise ValueError("Entity data missing required 'attributes' field")
        if "source" not in entity_data:
            raise ValueError("Entity data missing required 'source' field")
        if "importance" not in entity_data:
            raise ValueError("Entity data missing required 'importance' field")
        if "last_updated" not in entity_data:
            raise ValueError("Entity data missing required 'last_updated' field")

        # Parse attributes strictly - handle JSON string or dict
        attributes_raw = entity_data["attributes"]
        attributes = {}

        if isinstance(attributes_raw, str):
            try:
                import json

                attributes_dict = json.loads(attributes_raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in attributes: {e}") from e
        elif isinstance(attributes_raw, dict):
            attributes_dict = attributes_raw
        else:
            raise ValueError(f"Attributes must be dict or JSON string, got {type(attributes_raw)}")

        # Parse each attribute strictly
        for attr_name, attr_data in attributes_dict.items():
            if not isinstance(attr_data, dict):
                raise ValueError(f"Attribute '{attr_name}' must be a dict, got {type(attr_data)}")

            # Validate all required attribute fields
            required_fields = ["value", "confidence", "source", "timestamp"]
            for field in required_fields:
                if field not in attr_data:
                    raise ValueError(f"Attribute '{attr_name}' missing required field '{field}'")

            attributes[attr_name] = JanusAttributeData(**attr_data)

        return JanusEntityData(
            id=entity_data["id"],
            type=entity_data["type"],
            attributes=attributes,
            source=entity_data["source"],
            importance=entity_data["importance"],
            last_updated=entity_data["last_updated"],
        )

    async def initialize(self) -> None:
        """Initialize the JanusGraph connection.

        Creates the Gremlin client instance and verifies the connection.
        Also ensures that required indexes are created.

        Raises:
            ProviderError: If Gremlin Python driver is not available or connection fails
        """
        if not JANUS_AVAILABLE:
            raise ProviderError(
                message="Gremlin Python driver is not installed. Install with 'pip install gremlinpython'",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="DependencyError",
                    error_location="initialize",
                    component=self.name,
                    operation="initialize",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="initialize",
                    retry_count=0,
                ),
            )

        settings = cast(JanusProviderSettings, self.settings)

        try:
            # Set up serializer
            message_serializer = GraphBinaryMessageSerializer()

            # Create Gremlin client
            self._client = Client(
                settings.url,
                "g",
                pool_size=settings.connection_pool_size,
                message_serializer=message_serializer,
                username=settings.username if settings.username else None,
                password=settings.password if settings.password else None,
                read_timeout=settings.read_timeout,
                write_timeout=settings.write_timeout,
            )

            # Create remote connection for traversals
            connection = DriverRemoteConnection(settings.url, settings.traversal_source)
            self._g = traversal().withRemote(connection)

            # Verify connection by executing a simple query
            self._client.submit("g.V().limit(1).count()").all().result()

            # Create schema (indexes)
            await self._setup_schema()

            self._initialized = True
            logger.info(f"JanusGraph provider '{self.name}' initialized successfully")

        except Exception as e:
            raise ProviderError(
                message=f"Failed to connect to JanusGraph: {str(e)}",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="ConnectionError",
                    error_location="initialize",
                    component=self.name,
                    operation="connect",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="connect",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def shutdown(self) -> None:
        """Shut down the JanusGraph connection.

        Closes the Gremlin client instance and releases resources.
        """
        if self._client:
            self._client.close()
            self._client = None
            self._g = None
            self._initialized = False
            logger.info(f"JanusGraph provider '{self.name}' shut down")

    async def _setup_schema(self) -> None:
        """Set up JanusGraph schema (indexes).

        Creates indexes to optimize performance:
        - Create vertex label for Entity
        - Create edge label for RELATES_TO
        - Create property key for id, type, etc.
        - Create composite index on Entity(id)
        - Create index on Entity(type)
        """
        try:
            # JanusGraph schema management is typically done via the management API
            # This uses execute_query to send schema management commands

            # Create vertex label for Entity if it doesn't exist
            await self._execute_query("""
                mgmt = graph.openManagement()
                if (!mgmt.getVertexLabel('Entity')) {
                    mgmt.makeVertexLabel('Entity').make()
                }
                mgmt.commit()
            """)

            # Create edge label for RELATES_TO if it doesn't exist
            await self._execute_query("""
                mgmt = graph.openManagement()
                if (!mgmt.getEdgeLabel('RELATES_TO')) {
                    mgmt.makeEdgeLabel('RELATES_TO').make()
                }
                mgmt.commit()
            """)

            # Create property keys
            await self._execute_query("""
                mgmt = graph.openManagement()
                if (!mgmt.getPropertyKey('id')) {
                    mgmt.makePropertyKey('id').dataType(String.class).make()
                }
                if (!mgmt.getPropertyKey('type')) {
                    mgmt.makePropertyKey('type').dataType(String.class).make()
                }
                if (!mgmt.getPropertyKey('relation_type')) {
                    mgmt.makePropertyKey('relation_type').dataType(String.class).make()
                }
                if (!mgmt.getPropertyKey('attributes')) {
                    mgmt.makePropertyKey('attributes').dataType(String.class).make()
                }
                if (!mgmt.getPropertyKey('importance')) {
                    mgmt.makePropertyKey('importance').dataType(Double.class).make()
                }
                mgmt.commit()
            """)

            # Create composite index for Entity.id
            await self._execute_query("""
                mgmt = graph.openManagement()
                id = mgmt.getPropertyKey('id')
                if (id && !mgmt.getGraphIndex('entityById')) {
                    mgmt.buildIndex('entityById', Vertex.class).addKey(id).unique().buildCompositeIndex()
                }
                mgmt.commit()
            """)

            # Create index for Entity.type
            await self._execute_query("""
                mgmt = graph.openManagement()
                type = mgmt.getPropertyKey('type')
                if (type && !mgmt.getGraphIndex('entityByType')) {
                    mgmt.buildIndex('entityByType', Vertex.class).addKey(type).buildCompositeIndex()
                }
                mgmt.commit()
            """)

            # Create index for edge relation_type
            await self._execute_query("""
                mgmt = graph.openManagement()
                relType = mgmt.getPropertyKey('relation_type')
                if (relType && !mgmt.getGraphIndex('edgeByRelationType')) {
                    mgmt.buildIndex('edgeByRelationType', Edge.class).addKey(relType).buildCompositeIndex()
                }
                mgmt.commit()
            """)

        except Exception as e:
            logger.warning(f"Failed to set up JanusGraph schema: {str(e)}")
            # Continue initialization even if schema setup fails
            # as the schema might already exist or will be managed externally

    async def _execute_query(
        self, query: str, bindings: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Gremlin query against JanusGraph.

        Args:
            query: Gremlin query to execute
            bindings: Parameter bindings for the query

        Returns:
            List of records as dictionaries

        Raises:
            ProviderError: If query execution fails
        """
        if not self._client:
            raise ProviderError(
                message="JanusGraph provider not initialized",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="StateError",
                    error_location="_execute_query",
                    component=self.name,
                    operation="execute_query",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="execute_query",
                    retry_count=0,
                ),
            )

        try:
            # Execute query with bindings
            bindings = bindings or {}
            result = self._client.submit(query, bindings).all().result()

            # Convert result to dictionaries
            records = []
            for item in result:
                # Handle different result types
                if hasattr(item, "properties"):
                    # Convert vertex/edge to dict
                    record = self._element_to_dict(item)
                    records.append(record)
                elif isinstance(item, dict):
                    records.append(item)
                elif isinstance(item, (int, float, str, bool)):
                    records.append({"value": item})
                else:
                    # Try to convert to dict
                    try:
                        records.append(dict(item))
                    except (TypeError, ValueError) as conversion_error:
                        # Fallback to string representation for non-dict-convertible items
                        logger.debug(
                            f"Could not convert graph item to dict, using string representation: {conversion_error}"
                        )
                        records.append({"value": str(item)})

            return records

        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute JanusGraph query: {str(e)}",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="QueryError",
                    error_location="_execute_query",
                    component=self.name,
                    operation="execute_query",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="execute_query",
                    retry_count=0,
                ),
                cause=e,
) from e

    def _element_to_dict(self, element: Any) -> dict[str, Any]:
        """Convert a Gremlin graph element (vertex/edge) to a dictionary."""
        result: dict[str, Any] = {}

        # Extract element ID
        result["_id"] = str(element.id)

        # Extract element label
        if hasattr(element, "label"):
            result["_label"] = element.label

        # Extract properties
        if hasattr(element, "properties"):
            for key, value in element.properties.items():
                # Handle multi-valued properties
                if isinstance(value, list):
                    if len(value) == 1:
                        result[key] = value[0].value
                    else:
                        result[key] = [v.value for v in value]
                else:
                    result[key] = value

        return result

    def _element_to_entity(self, element: Any) -> Entity | None:
        """Convert a Gremlin element to an Entity model."""
        try:
            element_dict = self._element_to_dict(element)

            # Extract required fields for Entity
            entity_id = element_dict.get("_id", "")
            entity_type = element_dict.get("_label", "unknown")

            # Build attributes excluding internal fields
            raw_properties = {k: v for k, v in element_dict.items() if not k.startswith("_")}

            # Convert raw properties to EntityAttribute objects
            attributes = {}
            for key, value in raw_properties.items():
                attributes[key] = EntityAttribute(
                    name=key, value=str(value), confidence=1.0, source="janus", timestamp=""
                )

            return Entity(
                id=str(entity_id),
                type=str(entity_type),
                attributes=attributes,
                relationships=[],
                tags=[],
                source="janus",
                importance=1.0,
                vector_id=None,
                last_updated="",
            )
        except Exception:
            return None

    async def add_entity(self, entity: Entity) -> GraphStoreResult:
        """Add or update an entity node in JanusGraph.

        Args:
            entity: Entity to add or update

        Returns:
            ID of the created/updated entity

        Raises:
            ProviderError: If entity creation fails
        """
        if not self._initialized:
            raise ProviderError(
                message="JanusGraph provider not initialized",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="StateError",
                    error_location="add_entity",
                    component=self.name,
                    operation="add_entity",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="add_entity",
                    retry_count=0,
                ),
            )

        try:
            # Convert entity to JanusGraph compatible format
            entity_props = {
                "id": entity.id,
                "type": entity.type,
                "source": entity.source,
                "importance": entity.importance,
                "last_updated": entity.last_updated,
            }

            # Convert attributes to a serializable format
            attributes = {}
            for attr_name, attr in entity.attributes.items():
                attributes[attr_name] = {
                    "name": attr.name,
                    "value": attr.value,
                    "confidence": attr.confidence,
                    "source": attr.source,
                    "timestamp": attr.timestamp,
                }
            entity_props["attributes"] = json.dumps(attributes)

            # Check if entity exists (this is case-sensitive)
            existing = await self._execute_query(
                "g.V().has('Entity', 'id', id).hasNext()", {"id": entity.id}
            )

            if existing and len(existing) > 0 and "value" in existing[0] and existing[0]["value"]:
                # Update existing entity
                await self._execute_query(
                    """
                    g.V().has('Entity', 'id', id)
                      .property('type', type)
                      .property('source', source)
                      .property('importance', importance)
                      .property('last_updated', last_updated)
                      .property('attributes', attributes)
                    """,
                    {
                        "id": entity.id,
                        "type": entity.type,
                        "source": entity.source,
                        "importance": entity.importance,
                        "last_updated": entity.last_updated,
                        "attributes": json.dumps(attributes),
                    },
                )
            else:
                # Create new entity
                await self._execute_query(
                    """
                    g.addV('Entity')
                      .property('id', id)
                      .property('type', type)
                      .property('source', source)
                      .property('importance', importance)
                      .property('last_updated', last_updated)
                      .property('attributes', attributes)
                    """,
                    {
                        "id": entity.id,
                        "type": entity.type,
                        "source": entity.source,
                        "importance": entity.importance,
                        "last_updated": entity.last_updated,
                        "attributes": json.dumps(attributes),
                    },
                )

            # Add relationships
            for rel in entity.relationships:
                await self.add_relationship(entity.id, rel.target_entity, rel.relation_type, rel)

            return GraphStoreResult(
                success=True,
                stored_entities=[entity.id],
                stored_relationships=[],
                failed_entities=[],
                failed_relationships=[],
                error_details={},
                execution_time_ms=None,
            )

        except Exception as e:
            raise ProviderError(
                message=f"Failed to add entity: {str(e)}",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="EntityError",
                    error_location="add_entity",
                    component=self.name,
                    operation="add_entity",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="add_entity",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID from JanusGraph.

        Args:
            entity_id: Unique identifier of the entity

        Returns:
            Entity object if found, None otherwise

        Raises:
            ProviderError: If retrieval fails
        """
        if not self._initialized:
            raise ProviderError(
                message="JanusGraph provider not initialized",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="StateError",
                    error_location="get_entity",
                    component=self.name,
                    operation="get_entity",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="get_entity",
                    retry_count=0,
                ),
            )

        try:
            # Get entity with its properties
            result = await self._execute_query(
                """
                g.V().has('Entity', 'id', id)
                  .project('id', 'type', 'source', 'importance', 'last_updated', 'attributes')
                  .by('id')
                  .by('type')
                  .by('source')
                  .by('importance')
                  .by('last_updated')
                  .by('attributes')
                """,
                {"id": entity_id},
            )

            if not result:
                return None

            entity_data = result[0]

            # Parse entity data using strict parser - no fallbacks
            janus_entity = self._parse_entity_data_strict(entity_data)

            # Convert Janus attributes to Entity attributes
            entity_attributes = {}
            for attr_name, janus_attr in janus_entity.attributes.items():
                entity_attributes[attr_name] = EntityAttribute(
                    name=attr_name,
                    value=janus_attr.value,
                    confidence=janus_attr.confidence,
                    source=janus_attr.source,
                    timestamp=janus_attr.timestamp,
                )

            # Get relationships
            rel_result = await self._execute_query(
                """
                g.V().has('Entity', 'id', id)
                  .outE('RELATES_TO')
                  .project('relation_type', 'confidence', 'source', 'timestamp', 'target_id')
                  .by('relation_type')
                  .by('confidence')
                  .by('source')
                  .by('timestamp')
                  .by(inV().values('id'))
                """,
                {"id": entity_id},
            )

            relationships = []
            for rel in rel_result:
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
                        timestamp=rel["timestamp"],
                    )
                )

            # Create Entity object using strict parsed data
            entity = Entity(
                id=janus_entity.id,
                type=janus_entity.type,
                attributes=entity_attributes,
                relationships=relationships,
                tags=[],  # JanusGraph doesn't have tags in our model
                source=janus_entity.source,
                importance=janus_entity.importance,
                vector_id=None,
                last_updated=janus_entity.last_updated,
            )

            return entity

        except Exception as e:
            raise ProviderError(
                message=f"Failed to get entity: {str(e)}",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="EntityError",
                    error_location="get_entity",
                    component="JanusGraphProvider",
                    operation="get_entity",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="get_entity",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def add_relationship(
        self,
        source_id: str,
        target_entity: str,
        relation_type: str,
        relationship: EntityRelationship,
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
        try:
            await self._execute_query(
                """
                g.V().has('Entity', 'id', source_id)
                  .as_('a')
                  .V().has('Entity', 'id', target_id)
                  .addE('RELATES_TO')
                  .property('relation_type', relation_type)
                  .property('confidence', confidence)
                  .property('source', source)
                  .property('timestamp', timestamp)
                  .from_('a')
                """,
                {
                    "source_id": source_id,
                    "target_id": target_entity,
                    "relation_type": relation_type,
                    "confidence": relationship.confidence,
                    "source": relationship.source,
                    "timestamp": relationship.timestamp,
                },
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to add relationship: {str(e)}",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="RelationshipError",
                    error_location="add_relationship",
                    component=self.name,
                    operation="add_relationship",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="add_relationship",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def _entity_exists(self, entity_id: str) -> bool:
        """Check if an entity exists in JanusGraph.

        Args:
            entity_id: ID of the entity to check

        Returns:
            True if entity exists, False otherwise
        """
        result = await self._execute_query(
            "g.V().has('Entity', 'id', id).hasNext()", {"id": entity_id}
        )

        return bool(result and len(result) > 0 and "value" in result[0] and result[0]["value"])

    async def query_relationships(
        self, entity_id: str, relation_type: str | None = None, direction: str = "outgoing"
    ) -> RelationshipSearchResult:
        """Query relationships for an entity.
        Args:
            entity_id: ID of the entity
            relation_type: Optional type to filter by
            direction: 'outgoing' or 'incoming'
        Returns:
            EntitySearchResult
        Raises:
            ProviderError: If query fails
        """
        try:
            # Build Gremlin query
            if direction == "outgoing":
                gremlin = """
                g.V().has('Entity', 'id', entity_id)
                  .outE('RELATES_TO')
                  .as_('r')
                  .inV().as_('target')
                  .select('r', 'target')
                """
            else:
                gremlin = """
                g.V().has('Entity', 'id', entity_id)
                  .inE('RELATES_TO')
                  .as_('r')
                  .outV().as_('source')
                  .select('r', 'source')
                """
            params = {"entity_id": entity_id}
            if relation_type:
                gremlin += ".has('relation_type', relation_type)"
                params["relation_type"] = relation_type
            records = await self._execute_query(gremlin, params)
            relationships = []
            for rec in records:
                if "r" not in rec:
                    raise ValueError("Query result missing required 'r' field")
                r = rec["r"]

                # Validate target exists - no fallbacks
                target = None
                if "target" in rec:
                    target = rec["target"]
                elif "source" in rec:
                    target = rec["source"]
                else:
                    raise ValueError("Query result missing both 'target' and 'source' fields")

                # Validate all required relationship fields - no fallbacks
                if "relation_type" not in r:
                    raise ValueError("Relationship missing required 'relation_type' field")
                if "confidence" not in r:
                    raise ValueError("Relationship missing required 'confidence' field")
                if "source" not in r:
                    raise ValueError("Relationship missing required 'source' field")
                if "timestamp" not in r:
                    raise ValueError("Relationship missing required 'timestamp' field")
                if "id" not in target:
                    raise ValueError("Target entity missing required 'id' field")

                relationships.append(
                    EntityRelationship(
                        relation_type=r["relation_type"],
                        target_entity=target["id"],
                        confidence=r["confidence"],
                        source=r["source"],
                        timestamp=r["timestamp"],
                    )
                )
            return RelationshipSearchResult(
                success=True,
                relationships=relationships,
                total_count=len(relationships),
                source_entity=entity_id,
                target_entity=None,
                execution_time_ms=None,
                metadata={"direction": direction, "relation_type": relation_type},
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to query relationships: {str(e)}",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="QueryError",
                    error_location="query_relationships",
                    component=self.name,
                    operation="query_relationships",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="query_relationships",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def traverse(
        self, start_id: str, relation_types: list[str] | None = None, max_depth: int = 2
    ) -> list[Entity]:
        """Traverse the graph starting from an entity in JanusGraph.

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
                message="JanusGraph provider not initialized",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="StateError",
                    error_location="traverse",
                    component=self.name,
                    operation="traverse",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="traverse",
                    retry_count=0,
                ),
            )

        try:
            # Check if start entity exists
            if not await self._entity_exists(start_id):
                return []

            # Build relation type filter
            rel_filter = ""
            if relation_types:
                rel_types_str = ", ".join([f"'{rel}'" for rel in relation_types])
                rel_filter = f".has('relation_type', within({rel_types_str}))"

            # Perform traversal
            query = f"""
            g.V().has('Entity', 'id', startId)
              .repeat(outE('RELATES_TO'){rel_filter}.inV().dedup())
              .emit()
              .times({max_depth})
              .values('id')
            """

            # Execute traversal
            results = await self._execute_query(query, {"startId": start_id})

            # Collect entity IDs
            entity_ids = set()
            for result in results:
                if isinstance(result, dict) and "value" in result:
                    entity_ids.add(result["value"])

            # Add start entity ID
            entity_ids.add(start_id)

            # Retrieve full entity objects
            entities = []
            for entity_id in entity_ids:
                entity = await self.get_entity(entity_id)
                if entity:
                    entities.append(entity)

            return entities

        except Exception as e:
            raise ProviderError(
                message=f"Failed to traverse graph: {str(e)}",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="TraversalError",
                    error_location="traverse",
                    component=self.name,
                    operation="traverse",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="traverse",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def query(self, query: str, params: Any | None = None) -> GraphQueryResult:
        """Execute a query and return model-driven results.
        Args:
            query: Query string
            params: Optional query parameters (should be a Pydantic model or None)
        Returns:
            EntitySearchResult
        Raises:
            ProviderError: If query fails
        """
        try:
            # For demonstration, only support a simple entity search
            if query.startswith("find_entities"):
                entity_type = None
                if params and hasattr(params, "entity_type"):
                    entity_type = params.entity_type
                elif params and isinstance(params, dict):
                    entity_type = params["entity_type"] if "entity_type" in params else None
                if not entity_type:
                    raise ProviderError(
                        message="entity_type required for find_entities",
                        context=ErrorContext.create(
                            flow_name="janus_provider",
                            error_type="ValidationError",
                            error_location="query",
                            component=self.name,
                            operation="find_entities",
                        ),
                        provider_context=ProviderErrorContext(
                            provider_name=self.name,
                            provider_type="graph_db",
                            operation="find_entities",
                            retry_count=0,
                        ),
                    )
                entities = await self._find_entities_by_type(entity_type)
                entity_models = [self._element_to_entity(e) for e in entities]
                entity_models = [e for e in entity_models if e]
                # Convert entities to nodes format for GraphQueryResult
                nodes = []
                for entity in entity_models:
                    if entity:
                        # Convert attributes back to simple dict for GraphQueryResult
                        properties = {}
                        for attr_name, attr in entity.attributes.items():
                            properties[attr_name] = attr.value

                        nodes.append(
                            {
                                "id": entity.id,
                                "type": entity.type,
                                "properties": properties,
                                "labels": [entity.type],
                                "importance": entity.importance,
                            }
                        )
                return GraphQueryResult(
                    success=True,
                    nodes=nodes,
                    edges=[],
                    metadata={"query_type": "find_entities", "entity_type": entity_type},
                    execution_time_ms=None,
                    total_count=len(nodes),
                )
            raise ProviderError(
                message="Unsupported query",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="QueryError",
                    error_location="query",
                    component=self.name,
                    operation="query",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="query",
                    retry_count=0,
                ),
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute query: {str(e)}",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="QueryError",
                    error_location="query",
                    component="JanusGraphProvider",
                    operation="query",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="query",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def _find_entities_by_type(self, entity_type: str) -> list[dict[str, Any]]:
        """Find entities by type in JanusGraph."""
        results = await self._execute_query(
            """
            g.V().has('Entity', 'type', type)
              .project('id', 'entity')
              .by('id')
              .by(valueMap())
            """,
            {"type": entity_type},
        )

        return results

    async def _find_entities_by_name(self, name: str) -> list[dict[str, Any]]:
        """Find entities by name attribute in JanusGraph."""
        # This requires searching in the JSON attributes field
        results = await self._execute_query(
            """
            g.V().has('Entity', 'attributes', textContains('name'))
              .filter(values('attributes').is(textContains(name)))
              .project('id', 'entity')
              .by('id')
              .by(valueMap())
            """,
            {"name": name},
        )

        return results

    async def _find_neighbors(
        self, entity_id: str, relation_type: str | None
    ) -> list[dict[str, Any]]:
        """Find neighboring entities in JanusGraph."""
        # Construct filter for relation type
        rel_filter = ""
        if relation_type:
            rel_filter = ".has('relation_type', relType)"

        query = f"""
        g.V().has('Entity', 'id', id)
          .outE('RELATES_TO'){rel_filter}
          .as('rel')
          .inV()
          .project('id', 'relation', 'entity')
          .by('id')
          .by(select('rel').values('relation_type'))
          .by(valueMap())
        """

        results = await self._execute_query(query, {"id": entity_id, "relType": relation_type})

        return results

    async def _find_path(self, from_id: str, to_id: str, max_depth: int) -> list[dict[str, Any]]:
        """Find path between entities in JanusGraph."""
        query = """
        g.V().has('Entity', 'id', fromId)
          .until(has('id', toId))
          .repeat(out().simplePath())
          .limit(1)
          .path()
          .unfold()
          .project('position', 'id', 'entity')
          .by(constant(-1))  // Will update position later
          .by('id')
          .by(valueMap())
        """

        results = await self._execute_query(query, {"fromId": from_id, "toId": to_id})

        # Update positions
        for i, result in enumerate(results):
            result["position"] = i

        return results

    async def delete_entity(self, entity_id: str) -> GraphDeleteResult:
        """Delete an entity and its relationships from JanusGraph.

        Args:
            entity_id: ID of the entity to delete

        Returns:
            True if entity was deleted, False if not found

        Raises:
            ProviderError: If deletion fails
        """
        if not self._initialized:
            raise ProviderError(
                message="JanusGraph provider not initialized",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="StateError",
                    error_location="delete_entity",
                    component=self.name,
                    operation="delete_entity",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="delete_entity",
                    retry_count=0,
                ),
            )

        try:
            # Check if entity exists
            if not await self._entity_exists(entity_id):
                return GraphDeleteResult(
                    success=False,
                    deleted_entities=[],
                    deleted_relationships=[],
                    not_found_entities=[entity_id],
                    not_found_relationships=[],
                    error_details={},
                    execution_time_ms=None,
                )

            # Delete entity and its relationships
            await self._execute_query(
                """
                g.V().has('Entity', 'id', id).drop()
                """,
                {"id": entity_id},
            )

            return GraphDeleteResult(
                success=True,
                deleted_entities=[entity_id],
                deleted_relationships=[],
                not_found_entities=[],
                not_found_relationships=[],
                error_details={},
                execution_time_ms=None,
            )

        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete entity: {str(e)}",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="DeletionError",
                    error_location="delete_entity",
                    component="JanusGraphProvider",
                    operation="delete_entity",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="delete_entity",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def delete_relationship(
        self, source_id: str, target_entity: str, relation_type: str | None = None
    ) -> bool:
        """Delete relationship(s) between entities in JanusGraph.

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
                message="JanusGraph provider not initialized",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="StateError",
                    error_location="delete_relationship",
                    component=self.name,
                    operation="delete_relationship",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="delete_relationship",
                    retry_count=0,
                ),
            )

        try:
            # Construct relation type filter
            rel_filter = ""
            if relation_type:
                rel_filter = ".has('relation_type', relType)"

            # Delete matching edges
            result = await self._execute_query(
                f"""
                g.V().has('Entity', 'id', sourceId)
                  .outE('RELATES_TO'){rel_filter}
                  .where(inV().has('id', targetId))
                  .drop()
                  .count()
                """,
                {"sourceId": source_id, "targetId": target_entity, "relType": relation_type},
            )

            # Check if any edges were deleted
            if result and len(result) > 0 and "value" in result[0]:
                deleted_count = result[0]["value"]
            else:
                deleted_count = 0
            # Ensure we return a boolean by explicitly converting
            return bool(deleted_count > 0)

        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete relationship: {str(e)}",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="RelationshipError",
                    error_location="delete_relationship",
                    component=self.name,
                    operation="delete_relationship",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="delete_relationship",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def remove_relationship(
        self, source_id: str, target_entity: str, relation_type: str
    ) -> None:
        """Remove a relationship between two entities in JanusGraph.

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
                    flow_name="janus_provider",
                    error_type="RelationshipError",
                    error_location="remove_relationship",
                    component=self.name,
                    operation="remove_relationship",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="remove_relationship",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def bulk_add_entities(self, entities: list[Entity]) -> GraphStoreResult:
        """Add multiple entities in bulk to JanusGraph.

        This method optimizes bulk insertions by batching entities.

        Args:
            entities: List of entities to add

        Returns:
            List of IDs of added entities

        Raises:
            ProviderError: If bulk operation fails
        """
        if not self._initialized:
            raise ProviderError(
                message="JanusGraph provider not initialized",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="StateError",
                    error_location="bulk_add_entities",
                    component=self.name,
                    operation="bulk_add_entities",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="bulk_add_entities",
                    retry_count=0,
                ),
            )

        try:
            settings = cast(JanusProviderSettings, self.settings)
            added_ids = []

            # Process in batches
            batch_size = settings.max_batch_size
            for i in range(0, len(entities), batch_size):
                batch = entities[i : i + batch_size]

                # Add each entity in the batch
                for entity in batch:
                    result = await self.add_entity(entity)
                    if result.success:
                        added_ids.extend(result.stored_entities)

            return GraphStoreResult(
                success=len(added_ids) == len(entities),
                stored_entities=added_ids,
                stored_relationships=[],
                failed_entities=[e.id for e in entities if e.id not in added_ids],
                failed_relationships=[],
                error_details={},
                execution_time_ms=None,
            )

        except Exception as e:
            raise ProviderError(
                message=f"Failed to bulk add entities: {str(e)}",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="BulkOperationError",
                    error_location="bulk_add_entities",
                    component="JanusGraphProvider",
                    operation="bulk_add_entities",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="bulk_add_entities",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def search_entities(
        self,
        query: str | None = None,
        entity_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> EntitySearchResult:
        """Search for entities based on criteria.

        Args:
            query: Optional text query to match against entity ID or attributes
            entity_type: Optional entity type to filter by
            tags: Optional list of tags to filter by
            limit: Maximum number of entities to return

        Returns:
            Structured search result with entities and metadata

        Raises:
            ProviderError: If the search operation fails
        """
        if not self._initialized:
            raise ProviderError(
                message="JanusGraph provider not initialized",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="StateError",
                    error_location="search_entities",
                    component=self.name,
                    operation="search_entities",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="search_entities",
                    retry_count=0,
                ),
            )

        try:
            start_time = datetime.now()

            # Build Gremlin query based on search criteria
            query_parts = ["g.V().hasLabel('Entity')"]
            params = {}

            if entity_type:
                query_parts.append(".has('type', entity_type)")
                params["entity_type"] = entity_type

            if query:
                # Search in entity ID or attributes
                query_parts.append(
                    ".where(__.has('id', containing(query_text)).or(__.has('attributes', containing(query_text))))"
                )
                params["query_text"] = query

            # Add limit
            query_parts.append(f".limit({limit})")

            # Project the results
            query_parts.append("""
                .project('id', 'type', 'source', 'importance', 'last_updated', 'attributes')
                .by('id')
                .by('type')
                .by('source')
                .by('importance')
                .by('last_updated')
                .by('attributes')
            """)

            gremlin_query = "".join(query_parts)

            # Execute query
            results = await self._execute_query(gremlin_query, params)

            # Convert to Entity objects
            entities = []
            for result in results:
                try:
                    # Use strict parser for all entity data
                    janus_entity = self._parse_entity_data_strict(result)

                    # Convert Janus attributes to Entity attributes
                    entity_attributes = {}
                    for attr_name, janus_attr in janus_entity.attributes.items():
                        entity_attributes[attr_name] = EntityAttribute(
                            name=attr_name,
                            value=janus_attr.value,
                            confidence=janus_attr.confidence,
                            source=janus_attr.source,
                            timestamp=janus_attr.timestamp,
                        )

                    entity = Entity(
                        id=janus_entity.id,
                        type=janus_entity.type,
                        attributes=entity_attributes,
                        relationships=[],  # Relationships not included in search for performance
                        tags=[],  # JanusGraph doesn't have tags in our model
                        source=janus_entity.source,
                        importance=janus_entity.importance,
                        vector_id=None,
                        last_updated=janus_entity.last_updated,
                    )
                    entities.append(entity)

                except Exception as e:
                    logger.warning(f"Failed to convert search result to entity: {e}")
                    continue

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000

            return EntitySearchResult(
                success=True,
                entities=entities,
                total_count=len(entities),
                search_query=query or "",
                execution_time_ms=execution_time,
                metadata={"entity_type_filter": entity_type, "tags_filter": tags, "limit": limit},
            )

        except Exception as e:
            raise ProviderError(
                message=f"Failed to search entities: {str(e)}",
                context=ErrorContext.create(
                    flow_name="janus_provider",
                    error_type="SearchError",
                    error_location="search_entities",
                    component="JanusGraphProvider",
                    operation="search_entities",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="search_entities",
                    retry_count=0,
                ),
                cause=e,
) from e
