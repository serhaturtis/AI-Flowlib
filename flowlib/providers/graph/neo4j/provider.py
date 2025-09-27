"""Neo4j graph database provider implementation.

This module provides a concrete implementation of the GraphDBProvider 
for Neo4j, a popular open-source graph database.

Note: This implementation currently uses the synchronous neo4j-python-driver
wrapped in async methods. For better async performance in production environments,
consider using the neo4j-async driver when it becomes stable:
https://github.com/neo4j/neo4j-python-driver/issues/766

The current approach is suitable for most use cases but may have limitations
under high concurrent load. The synchronous driver is battle-tested and stable.
"""

import logging
from typing import Dict, List, Optional, Any, cast, Union, Literal

from flowlib.core.errors.errors import ProviderError, ErrorContext
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.decorators import provider
from flowlib.providers.graph.base import GraphDBProvider, GraphDBProviderSettings
from flowlib.providers.graph.models import (
    Entity, EntityAttribute, EntityRelationship, EntitySearchResult,
    RelationshipSearchResult, GraphStoreResult, GraphQueryResult, GraphQueryParams,
    GraphDeleteResult, Neo4jQueryParams
)
from flowlib.providers.graph.neo4j.models import Neo4jAttributeData
from flowlib.providers.graph.neo4j.models import Neo4jNodeData

# Import necessary types for Provider inheritance  
from pydantic import Field

logger = logging.getLogger(__name__)

# Define dummy models for type annotations when neo4j is not installed
try:
    from neo4j import GraphDatabase, Driver, Session, Transaction, Result
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    logger.warning("neo4j-python-driver package not found. Install with 'pip install neo4j'")
    NEO4J_AVAILABLE = False
    # Define dummy types for type annotations
    Driver = Any  # type: ignore
    Session = Any  # type: ignore
    Transaction = Any  # type: ignore
    Result = Any  # type: ignore


class Neo4jProviderSettings(GraphDBProviderSettings):
    """Neo4j provider settings - direct inheritance, only Neo4j-specific fields.
    
    Neo4j requires:
    1. Connection URI (bolt://host:port)
    2. Authentication (username, password)
    3. Database name and connection pooling
    4. Neo4j-specific settings (encryption, trust)
    """
    
    # Neo4j connection settings
    uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI (e.g., 'bolt://localhost:7687')")
    username: str = Field(default="neo4j", description="Neo4j username for authentication")
    password: str = Field(default="password", description="Neo4j password for authentication (should be overridden in production)")
    database: str = Field(default="neo4j", description="Neo4j database name")
    
    # Neo4j security settings
    encryption: bool = Field(default=False, description="Whether to use encrypted connection")
    trust: Literal["TRUST_ALL_CERTIFICATES", "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"] = Field(default="TRUST_SYSTEM_CA_SIGNED_CERTIFICATES", description="Trust level for certificates")
    
    # Neo4j connection pool settings
    connection_timeout: int = Field(default=30, description="Connection timeout in seconds")
    connection_acquisition_timeout: int = Field(default=60, description="Timeout for acquiring connection from pool")
    max_connection_lifetime: int = Field(default=3600, description="Maximum lifetime of a connection")
    max_connection_pool_size: int = Field(default=100, description="Maximum number of connections in the pool")
    max_batch_size: int = Field(default=100, description="Maximum number of entities to process in a single batch")



@provider(provider_type="graph_db", name="neo4j", settings_class=Neo4jProviderSettings)
class Neo4jProvider(GraphDBProvider):
    """Neo4j graph database provider implementation.
    
    This provider interfaces with Neo4j using the official Python driver,
    mapping entities and relationships to Neo4j's property graph model.
    """
    
    def __init__(self, name: str = "neo4j", provider_type: str = "graph_db", settings: Union[Neo4jProviderSettings, Dict[str, Any], None] = None):
        """Initialize Neo4j graph database provider.
        
        Args:
            name: Provider name
            provider_type: Provider type
            settings: Provider settings
        """
        # Create default settings if none provided or convert dict to settings
        if settings is None:
            settings = Neo4jProviderSettings()
        elif isinstance(settings, dict):
            settings = Neo4jProviderSettings(**settings)
        super().__init__(name=name, provider_type=provider_type, settings=settings)
        self._driver: Optional[Driver] = None
    
    def is_initialized(self) -> bool:
        """Check if provider is initialized (compatibility method)."""
        return self.initialized
    
    def _parse_node_data_strict(self, node_data_raw: Dict[str, Any]) -> Neo4jNodeData:
        """Parse Neo4j node data with strict validation.
        
        Args:
            node_data_raw: Raw node data from Neo4j
            
        Returns:
            Validated Neo4j node data
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields exist
        if "id" not in node_data_raw:
            raise ValueError("Node data missing required 'id' field")
        
        # Parse attributes strictly - no fallbacks, all fields required
        if "attributes" not in node_data_raw:
            raise ValueError("Node data missing required 'attributes' field")
        attributes_raw = node_data_raw["attributes"]
        attributes = {}
        
        if isinstance(attributes_raw, str):
            import json
            try:
                attributes_dict = json.loads(attributes_raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in attributes: {e}")
        elif isinstance(attributes_raw, dict):
            attributes_dict = attributes_raw
        else:
            raise ValueError(f"Attributes must be dict or JSON string, got {type(attributes_raw)}")
        
        # Parse each attribute strictly
        for attr_name, attr_data in attributes_dict.items():
            if not isinstance(attr_data, dict):
                raise ValueError(f"Attribute '{attr_name}' must be a dict, got {type(attr_data)}")
            
            # Validate all required attribute fields exist
            required_fields = ["value", "confidence", "source", "timestamp"]
            for field in required_fields:
                if field not in attr_data:
                    raise ValueError(f"Attribute '{attr_name}' missing required field '{field}'")
            
            attributes[attr_name] = Neo4jAttributeData(**attr_data)
        
        # Validate all required node fields
        if "type" not in node_data_raw:
            raise ValueError("Node data missing required 'type' field")
        if "tags" not in node_data_raw:
            raise ValueError("Node data missing required 'tags' field")
        if "source" not in node_data_raw:
            raise ValueError("Node data missing required 'source' field")
        if "importance" not in node_data_raw:
            raise ValueError("Node data missing required 'importance' field")
        if "created_at" not in node_data_raw:
            raise ValueError("Node data missing required 'created_at' field")
        if "updated_at" not in node_data_raw:
            raise ValueError("Node data missing required 'updated_at' field")
        
        return Neo4jNodeData(
            id=node_data_raw["id"],
            type=node_data_raw["type"],
            attributes=attributes,
            tags=node_data_raw["tags"],
            source=node_data_raw["source"],
            importance=node_data_raw["importance"],
            created_at=node_data_raw["created_at"],
            updated_at=node_data_raw["updated_at"]
        )
        
    async def initialize(self) -> None:
        """Initialize the Neo4j connection.
        
        Creates the Neo4j driver instance and verifies the connection.
        Also ensures that required indexes and constraints are created.
        
        Raises:
            ProviderError: If Neo4j driver is not available or connection fails
        """
        if not NEO4J_AVAILABLE:
            raise ProviderError(
                message="Neo4j driver is not installed. Install with 'pip install neo4j'",
                context=ErrorContext.create(
                    flow_name="neo4j_provider",
                    error_type="DependencyError",
                    error_location="initialize",
                    component=self.name,
                    operation="check_neo4j_dependency"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="initialize",
                    retry_count=0
                )
            )
        
        try:
            # Create the Neo4j driver with settings
            settings = cast(Neo4jProviderSettings, self.settings)
            self._driver = GraphDatabase.driver(
                settings.uri,
                auth=(settings.username, settings.password),
                encrypted=settings.encryption,
                trust=settings.trust,
                connection_timeout=settings.connection_timeout,
                connection_acquisition_timeout=settings.connection_acquisition_timeout,
                max_connection_lifetime=settings.max_connection_lifetime,
                max_connection_pool_size=settings.max_connection_pool_size
            )
            
            # Verify connection
            await self._execute_query("RETURN 1 AS test")
            
            # Create indexes and constraints
            await self._setup_schema()
            
            self._initialized = True
            logger.info(f"Neo4j provider '{self.name}' initialized successfully")
            
        except (ServiceUnavailable, AuthError) as e:
            raise ProviderError(
                message=f"Failed to connect to Neo4j: {str(e)}",
                context=ErrorContext.create(
                    flow_name="neo4j_provider",
                    error_type="ConnectionError",
                    error_location="initialize",
                    component=self.name,
                    operation="connect_to_neo4j"
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
                message=f"Failed to initialize Neo4j provider: {str(e)}",
                context=ErrorContext.create(
                    flow_name="neo4j_provider",
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
    
    async def shutdown(self) -> None:
        """Shut down the Neo4j connection.
        
        Closes the Neo4j driver instance and releases resources.
        """
        if self._driver:
            self._driver.close()
            self._driver = None
            self._initialized = False
            logger.info(f"Neo4j provider '{self.name}' shut down")
    
    async def _setup_schema(self) -> None:
        """Set up Neo4j schema (indexes and constraints).
        
        Creates indexes and constraints to optimize performance:
        - Unique constraint on Entity.id
        - Index on Entity.type
        - Index on EntityRelationship.relation_type
        """
        # Create constraint for unique entity IDs
        await self._execute_query(
            """
            CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) 
            REQUIRE e.id IS UNIQUE
            """
        )
        
        # Create index on entity type
        await self._execute_query(
            """
            CREATE INDEX IF NOT EXISTS FOR (e:Entity) 
            ON (e.type)
            """
        )
        
        # Create index on relationship types
        await self._execute_query(
            """
            CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATES_TO]-() 
            ON (r.relation_type)
            """
        )
    
    async def _execute_query(
        self, 
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query against Neo4j.
        
        Args:
            query: Cypher query to execute
            parameters: Parameters for the query
            
        Returns:
            List of records as dictionaries
            
        Raises:
            ProviderError: If query execution fails
        """
        if not self._driver:
            raise ProviderError(
                message="Neo4j provider not initialized",
                context=ErrorContext.create(
                    flow_name="neo4j_provider",
                    error_type="NotInitializedError",
                    error_location="_execute_query",
                    component=self.name,
                    operation="check_initialization"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="_execute_query",
                    retry_count=0
                )
            )
        
        try:
            settings = cast(Neo4jProviderSettings, self.settings)
            parameters = parameters or {}

            # Use async with statement when proper async driver is available
            # For now, we're using the synchronous driver methods
            with self._driver.session(database=settings.database) as session:
                result = session.run(query, parameters)
                records = [dict(record) for record in result]
                return records
                
        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute Neo4j query: {str(e)}",
                context=ErrorContext.create(
                    flow_name="neo4j_provider",
                    error_type="QueryExecutionError",
                    error_location="_execute_query",
                    component=self.name,
                    operation="execute_cypher_query"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="_execute_query",
                    retry_count=0
                ),
                cause=e
            )
    
    async def add_entity(self, entity: Entity) -> GraphStoreResult:
        """Add or update an entity node in Neo4j.
        
        Args:
            entity: Entity to add or update
            
        Returns:
            GraphStoreResult with operation details
            
        Raises:
            ProviderError: If entity creation fails
        """
        try:
            # Convert entity to Neo4j-compatible format
            entity_props = {
                "id": entity.id,
                "type": entity.type,
                "source": entity.source,
                "importance": entity.importance,
                "last_updated": entity.last_updated
            }
            
            # Convert attributes to a serializable format
            attributes = {}
            for attr_name, attr in entity.attributes.items():
                attributes[attr_name] = {
                    "name": attr.name,
                    "value": attr.value,
                    "confidence": attr.confidence,
                    "source": attr.source,
                    "timestamp": attr.timestamp
                }
            
            # Create or merge the entity node
            await self._execute_query(
                """
                MERGE (e:Entity {id: $id})
                SET e = $properties, e.attributes = $attributes
                RETURN e.id as id
                """,
                {
                    "id": entity.id,
                    "properties": entity_props,
                    "attributes": attributes
                }
            )
            
            # Add relationships
            for rel in entity.relationships:
                await self.add_relationship(
                    entity.id,
                    rel.target_entity,
                    rel.relation_type,
                    rel  # rel is already an EntityRelationship object
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
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to add entity: {str(e)}",
                context=ErrorContext.create(
                    flow_name="neo4j_provider",
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
    
    def _convert_node_to_entity(self, node_data: Dict[str, Any]) -> Entity:
        """Helper method to convert Neo4j node data dictionary to an Entity object.
        Assumes `node_data` contains the properties of the node itself.
        Does NOT handle relationships fetched separately.
        
        Raises:
            ValueError: If node data is invalid or missing required fields
        """
        # Use strict parser - no fallbacks, all fields required
        neo4j_node = self._parse_node_data_strict(node_data)
        
        # Convert Neo4j attributes to Entity attributes
        entity_attributes = {}
        for attr_name, neo4j_attr in neo4j_node.attributes.items():
            entity_attributes[attr_name] = EntityAttribute(
                name=attr_name,
                value=neo4j_attr.value,
                confidence=neo4j_attr.confidence,
                source=neo4j_attr.source,
                timestamp=neo4j_attr.timestamp
            )
        
        # Relationships are not handled by this helper
        relationships: List[EntityRelationship] = []
        
        entity = Entity(
            id=neo4j_node.id,
            type=neo4j_node.type,
            attributes=entity_attributes,
            relationships=relationships,
            tags=neo4j_node.tags,
            source=neo4j_node.source,
            importance=neo4j_node.importance,
            vector_id=None,
            last_updated=neo4j_node.updated_at
        )
        return entity

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID from Neo4j.
        
        Args:
            entity_id: Unique identifier of the entity
            
        Returns:
            Entity object if found, None otherwise
            
        Raises:
            ProviderError: If retrieval fails
        """
        try:
            # Query the entity WITH its relationships
            result = await self._execute_query(
                """
                MATCH (e:Entity {id: $id})
                OPTIONAL MATCH (e)-[r:RELATES_TO]->(target:Entity)
                RETURN 
                    e.id as id, 
                    e.type as type, 
                    e.source as source,
                    e.importance as importance,
                    e.last_updated as last_updated,
                    e.attributes as attributes,
                    e.tags as tags,
                    collect({
                        target_id: target.id,
                        relation_type: r.relation_type,
                        confidence: r.confidence,
                        source: r.source,
                        timestamp: r.timestamp
                    }) as relationships
                """,
                {"id": entity_id}
            )
            
            if not result:
                return None
                
            record = result[0]
            
            # --- REVERTED: Use original deserialization logic for get_entity --- 
            # Deserialize attributes
            attributes = {}
            if "attributes" not in record:
                raise ValueError(f"Entity '{entity_id}' missing required 'attributes' field")
            attributes_raw = record["attributes"]
            attributes_dict = {}
            if isinstance(attributes_raw, str):
                 try:
                     import json
                     attributes_dict = json.loads(attributes_raw)
                 except json.JSONDecodeError:
                     logger.warning(f"Could not decode attributes JSON for entity {entity_id}: {attributes_raw}")
            elif isinstance(attributes_raw, dict):
                 attributes_dict = attributes_raw
                 
            for attr_name, attr_data in attributes_dict.items():
                if isinstance(attr_data, dict):
                    # Validate all required attribute fields
                    if "value" not in attr_data:
                        raise ValueError(f"Attribute '{attr_name}' missing required 'value' field")
                    if "confidence" not in attr_data:
                        raise ValueError(f"Attribute '{attr_name}' missing required 'confidence' field")
                    if "source" not in attr_data:
                        raise ValueError(f"Attribute '{attr_name}' missing required 'source' field")
                    if "timestamp" not in attr_data:
                        raise ValueError(f"Attribute '{attr_name}' missing required 'timestamp' field")
                        
                    attributes[attr_name] = EntityAttribute(
                        name=attr_name,
                        value=attr_data["value"],
                        confidence=attr_data["confidence"],
                        source=attr_data["source"],
                        timestamp=attr_data["timestamp"]
                    )
                else:
                    attributes[attr_name] = EntityAttribute(
                        name=attr_name,
                        value=str(attr_data),
                        confidence=0.9,
                        source="neo4j_import"
                    )
            
            # Deserialize relationships
            relationships = []
            if "relationships" not in record:
                raise ValueError(f"Entity '{entity_id}' missing required 'relationships' field")
            relationships_raw = record["relationships"]
            if isinstance(relationships_raw, list):
                for rel_data in relationships_raw:
                    # Skip invalid relationship items (null target)
                    if isinstance(rel_data, dict) and "target_id" in rel_data and rel_data["target_id"]:
                        # Validate all required relationship fields
                        if "relation_type" not in rel_data:
                            raise ValueError("Relationship missing required 'relation_type' field")
                        if "confidence" not in rel_data:
                            raise ValueError("Relationship missing required 'confidence' field")
                        if "source" not in rel_data:
                            raise ValueError("Relationship missing required 'source' field")
                        if "timestamp" not in rel_data:
                            raise ValueError("Relationship missing required 'timestamp' field")
                            
                        relationships.append(
                            EntityRelationship(
                                relation_type=rel_data["relation_type"],
                                target_entity=rel_data["target_id"],
                                confidence=rel_data["confidence"],
                                source=rel_data["source"],
                                timestamp=rel_data["timestamp"]
                            )
                        )
            # --- End Reverted Section --- 

            # Create and return entity
            # Validate all required entity fields
            if "id" not in record:
                raise ValueError("Entity query result missing required 'id' field")
            if "type" not in record:
                raise ValueError(f"Entity '{entity_id}' missing required 'type' field")
            if "tags" not in record:
                raise ValueError(f"Entity '{entity_id}' missing required 'tags' field")
            if "source" not in record:
                raise ValueError(f"Entity '{entity_id}' missing required 'source' field")
            if "importance" not in record:
                raise ValueError(f"Entity '{entity_id}' missing required 'importance' field")
            if "last_updated" not in record:
                raise ValueError(f"Entity '{entity_id}' missing required 'last_updated' field")
                
            entity = Entity(
                id=record["id"],
                type=record["type"],
                attributes=attributes,
                relationships=relationships,
                tags=record["tags"],
                source=record["source"],
                importance=record["importance"],
                vector_id=None,
                last_updated=record["last_updated"]
            )
            
            return entity
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to get entity: {str(e)}",
                context=ErrorContext.create(
                    flow_name="neo4j_provider",
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
        try:
            await self._execute_query(
                """
                MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id})
                MERGE (a)-[r:RELATES_TO {relation_type: $relation_type}]->(b)
                SET r.confidence = $confidence, r.source = $source, r.timestamp = $timestamp
                RETURN r
                """,
                {
                    "source_id": source_id,
                    "target_id": target_entity,
                    "relation_type": relation_type,
                    "confidence": relationship.confidence,
                    "source": relationship.source,
                    "timestamp": relationship.timestamp
                }
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to add relationship: {str(e)}",
                context=ErrorContext.create(
                    flow_name="neo4j_provider",
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
    
    async def _entity_exists(self, entity_id: str) -> bool:
        """Check if an entity exists.
        
        Args:
            entity_id: Entity ID to check
            
        Returns:
            True if entity exists, False otherwise
        """
        result = await self._execute_query(
            "MATCH (e:Entity {id: $id}) RETURN count(e) as count",
            {"id": entity_id}
        )
        
        if result and result[0]["count"] > 0:
            return True
        return False
        
    async def query_relationships(
        self, 
        entity_id: str, 
        relation_type: Optional[str] = None, 
        direction: str = "outgoing"
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
            # Build Cypher query
            if direction == "outgoing":
                cypher = """
                MATCH (e:Entity {id: $entity_id})-[r:RELATES_TO]->(target:Entity)
                RETURN r, target
                """
            else:
                cypher = """
                MATCH (source:Entity)-[r:RELATES_TO]->(e:Entity {id: $entity_id})
                RETURN r, source
                """
            params = {"entity_id": entity_id}
            if relation_type:
                cypher = cypher.replace("RETURN r,", "WHERE r.relation_type = $relation_type RETURN r,")
                params["relation_type"] = relation_type
            records = await self._execute_query(cypher, params)
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
                    
                relationships.append(EntityRelationship(
                    relation_type=r["relation_type"],
                    target_entity=target["id"],
                    confidence=r["confidence"],
                    source=r["source"],
                    timestamp=r["timestamp"]
                ))
            return RelationshipSearchResult(
                success=True,
                relationships=relationships,
                total_count=len(relationships),
                source_entity=entity_id if direction == "outgoing" else None,
                target_entity=entity_id if direction == "incoming" else None,
                execution_time_ms=None,
                metadata={"direction": direction, "relation_type": relation_type},
            )
        except Exception as e:
            raise ProviderError(
                message=f"Failed to query relationships: {str(e)}",
                context=ErrorContext.create(
                    flow_name="neo4j_provider",
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
        """Traverse the graph starting from an entity in Neo4j.
        
        Args:
            start_id: ID of the starting entity
            relation_types: Optional list of relation types to traverse
            max_depth: Maximum traversal depth
            
        Returns:
            List of entities found in traversal
            
        Raises:
            ProviderError: If traversal fails
        """
        try:
            # Check if start entity exists
            if not await self._entity_exists(start_id):
                return []
            
            # Create relationship filter
            rel_filter = ""
            if relation_types:
                types_list = [f"r.relation_type = '{type}'" for type in relation_types]
                rel_filter = f"WHERE {' OR '.join(types_list)}"
            
            # Use variable length path for traversal
            query = f"""
            MATCH path = (start:Entity {{id: $start_id}})-[r:RELATES_TO*1..{max_depth}]->(e:Entity)
            {rel_filter}
            RETURN DISTINCT e.id as entity_id
            """
            
            # Execute the query
            results = await self._execute_query(
                query,
                {"start_id": start_id}
            )
            
            # Fetch complete entity objects
            entities = []
            for result in results:
                entity_id = result["entity_id"]
                entity = await self.get_entity(entity_id)
                if entity:
                    entities.append(entity)
                    
            # Add the start entity
            start_entity = await self.get_entity(start_id)
            if start_entity and start_entity not in entities:
                entities.insert(0, start_entity)
                
            return entities
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to traverse graph: {str(e)}",
                context=ErrorContext.create(
                    flow_name="neo4j_provider",
                    error_type="GraphTraversalError",
                    error_location="traverse",
                    component=self.name,
                    operation="traverse_graph"
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
        params: Optional[GraphQueryParams] = None
    ) -> GraphQueryResult:
        """Execute a native Cypher query in Neo4j.
        
        Args:
            query: Cypher query string
            params: Optional Neo4j-specific query parameters
            
        Returns:
            GraphQueryResult with nodes, edges, and metadata
            
        Examples:
            >>> # Simple query
            >>> result = await provider.query("MATCH (n:Person) RETURN n LIMIT 10")
            
            >>> # Query with Neo4j-specific parameters
            >>> params = Neo4jQueryParams(
            ...     limit=100,
            ...     database="analytics",
            ...     access_mode="READ"
            ... )
            >>> result = await provider.query(
            ...     "MATCH (n) RETURN count(n) as total",
            ...     params=params
            ... )
            
        Raises:
            ProviderError: If query execution fails
        """
        try:
            # Convert params to dict for query execution
            query_params = {}
            if params:
                # Check if params is a Neo4jQueryParams model
                if isinstance(params, Neo4jQueryParams):
                    # Extract standard parameters from Neo4jQueryParams
                    if params.limit is not None:
                        query_params['limit'] = params.limit
                    if params.offset is not None:
                        query_params['offset'] = params.offset
                    if params.timeout_ms is not None:
                        query_params['timeout'] = int(params.timeout_ms / 1000)  # Convert to seconds
                    
                    # Add extra params
                    query_params.update(params.extra_params)
                    
                    # Neo4j-specific params are handled at session/transaction level
                    # - params.database: handled when creating session
                    # - params.access_mode: handled when creating transaction
                    # - params.bookmarks: handled for causal consistency
                elif isinstance(params, GraphQueryParams):
                    # Handle base GraphQueryParams
                    if hasattr(params, 'limit') and params.limit is not None:
                        query_params['limit'] = params.limit
                    if hasattr(params, 'offset') and params.offset is not None:
                        query_params['offset'] = params.offset
            
            # Execute the query
            results = await self._execute_query(query, query_params)
            
            # Process results into nodes and edges
            nodes = []
            edges = []
            
            for record in results:
                for key, value in record.items():
                    if isinstance(value, dict):
                        # Check if it's a node (has 'id' property)
                        if 'id' in value:
                            nodes.append(value)
                        # Check if it's an edge/relationship
                        elif 'source' in value or 'target' in value:
                            edges.append(value)
                    elif isinstance(value, list):
                        # Handle lists of nodes/edges
                        for item in value:
                            if isinstance(item, dict):
                                if 'id' in item:
                                    nodes.append(item)
                                elif 'source' in item or 'target' in item:
                                    edges.append(item)
            
            return GraphQueryResult(
                success=True,
                nodes=nodes,
                edges=edges,
                metadata={
                    "query": query,
                    "record_count": len(results)
                },
                execution_time_ms=None,
                total_count=len(results)
            )
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to execute query: {str(e)}",
                context=ErrorContext.create(
                    flow_name="neo4j_provider",
                    error_type="QueryExecutionError",
                    error_location="query",
                    component=self.name,
                    operation="execute_native_query"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="graph_db",
                    operation="query",
                    retry_count=0
                ),
                cause=e
            )
        
    async def _find_entities_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Find entities by name in Neo4j."""
        results = await self._execute_query(
            """
            MATCH (e:Entity)
            WHERE e.attributes.name.value CONTAINS $name
            RETURN e.id as id, e as entity
            """,
            {"name": name}
        )
        
        return [{"id": r["id"], "entity": r["entity"]} for r in results]
        
    async def _find_neighbors(self, entity_id: str, relation_type: Optional[str]) -> List[Dict[str, Any]]:
        """Find neighboring entities in Neo4j."""
        query = """
        MATCH (e:Entity {id: $id})-[r:RELATES_TO]->(neighbor:Entity)
        WHERE $relation_type IS NULL OR r.relation_type = $relation_type
        RETURN 
            neighbor.id as id, 
            r.relation_type as relation, 
            neighbor as entity
        """
        
        results = await self._execute_query(
            query,
            {
                "id": entity_id,
                "relation_type": relation_type
            }
        )
        
        return [
            {
                "id": r["id"],
                "relation": r["relation"],
                "entity": r["entity"]
            } for r in results
        ]
        
    async def _find_path(self, from_id: str, to_id: str, max_depth: int) -> List[Dict[str, Any]]:
        """Find path between entities in Neo4j."""
        query = f"""
        MATCH path = shortestPath((source:Entity {{id: $from_id}})-[r:RELATES_TO*1..{max_depth}]->(target:Entity {{id: $to_id}}))
        UNWIND nodes(path) as node
        WITH node, index(nodes(path), node) as position
        RETURN position, node.id as id, node as entity
        ORDER BY position
        """
        
        results = await self._execute_query(
            query,
            {
                "from_id": from_id,
                "to_id": to_id
            }
        )
        
        return [
            {
                "position": r["position"],
                "id": r["id"],
                "entity": r["entity"]
            } for r in results
        ]
    
    async def delete_entity(self, entity_id: str) -> GraphDeleteResult:
        """Delete an entity and its relationships from Neo4j.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if entity was deleted, False if not found
            
        Raises:
            ProviderError: If deletion fails
        """
        try:
            # Check if entity exists
            entity_exists = await self._entity_exists(entity_id)
            if not entity_exists:
                return GraphDeleteResult(
                    success=False,
                    not_found_entities=[entity_id],
                    execution_time_ms=None
                )
            
            # Delete entity and all relationships
            result = await self._execute_query(
                """
                MATCH (e:Entity {id: $id})
                DETACH DELETE e
                RETURN count(e) as deleted
                """,
                {"id": entity_id}
            )

            if result and result[0]["deleted"] > 0:
                return GraphDeleteResult(
                    success=True,
                    deleted_entities=[entity_id],
                    execution_time_ms=None
                )

            return GraphDeleteResult(
                success=False,
                not_found_entities=[entity_id],
                execution_time_ms=None
            )
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete entity: {str(e)}",
                context=ErrorContext.create(
                    flow_name="neo4j_provider",
                    error_type="EntityDeletionError",
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
        """Delete relationship(s) between entities in Neo4j.
        
        Args:
            source_id: ID of the source entity
            target_entity: ID of the target entity
            relation_type: Optional type to filter by
            
        Returns:
            True if relationships were deleted, False if none found
            
        Raises:
            ProviderError: If deletion fails
        """
        try:
            # Construct relation type filter
            relation_filter = ""
            if relation_type:
                relation_filter = "AND r.relation_type = $relation_type"
            
            # Delete matching relationships
            result = await self._execute_query(
                f"""
                MATCH (source:Entity {{id: $source_id}})-[r:RELATES_TO]->(target:Entity {{id: $target_id}})
                WHERE true {relation_filter}
                WITH r
                DELETE r
                RETURN count(r) as deleted
                """,
                {
                    "source_id": source_id,
                    "target_id": target_entity,
                    "relation_type": relation_type
                }
            )
            
            if result and result[0]["deleted"] > 0:
                return True
            
            return False
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete relationship: {str(e)}",
                context=ErrorContext.create(
                    flow_name="neo4j_provider",
                    error_type="RelationshipDeletionError",
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
        """Remove a relationship between two entities in Neo4j.
        
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
                    flow_name="neo4j_provider",
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
        """Add multiple entities in bulk to Neo4j.
        
        This method optimizes bulk insertions by batching entities.
        
        Args:
            entities: List of entities to add
            
        Returns:
            GraphStoreResult with bulk operation details
            
        Raises:
            ProviderError: If bulk operation fails
        """
        try:
            settings = self.settings
            added_ids = []
            
            # Process in batches
            batch_size = settings.max_batch_size
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i+batch_size]
                
                # Add each entity in the batch
                for entity in batch:
                    result = await self.add_entity(entity)
                    if result.success and result.stored_entities:
                        added_ids.extend(result.stored_entities)
                    
            return GraphStoreResult(
                success=True,
                stored_entities=added_ids,
                stored_relationships=[],
                failed_entities=[],
                failed_relationships=[],
                error_details={},
                execution_time_ms=None
            )
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to bulk add entities: {str(e)}",
                context=ErrorContext.create(
                    flow_name="neo4j_provider",
                    error_type="BulkEntityAddError",
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
        """Search for entities in Neo4j based on criteria.
        
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
        try:
            cypher_parts: List[str] = []
            params: Dict[str, Any] = {}
            where_clauses: List[str] = []

            cypher_parts.append("MATCH (e:Entity)")

            if query and query.strip():
                 # Adjusted query to handle potential non-string attributes safely
                 where_clauses.append("(e.id CONTAINS $query OR ANY(prop_key IN keys(e) WHERE prop_key <> 'relationships' AND toString(e[prop_key]) CONTAINS $query))")
                 params['query'] = query

            if entity_type:
                where_clauses.append("e.type = $type")
                params['type'] = entity_type

            if tags:
                where_clauses.append("ANY(tag IN e.tags WHERE tag IN $tags)")
                params['tags'] = tags

            if where_clauses:
                cypher_parts.append("WHERE " + " AND ".join(where_clauses))

            # Return node properties directly
            cypher_parts.append(f"RETURN e LIMIT {limit}")

            final_query = " ".join(cypher_parts)
            logger.debug(f"Executing Neo4j search query: {final_query} with params: {params}")
            query_results = await self._execute_query(final_query, params)

            entities = []
            for record in query_results:
                if 'e' in record and isinstance(record['e'], dict):
                    # Pass the node properties dictionary to the helper
                    entity = self._convert_node_to_entity(record['e']) 
                    if entity:
                       entities.append(entity)
                else:
                    logger.warning(f"Unexpected record format in search_entities: {record}")
            
            logger.debug(f"Neo4j search found {len(entities)} entities.")
            return EntitySearchResult(
                success=True,
                entities=entities,
                total_count=len(entities),
                search_query=query or "",
                execution_time_ms=None,
                metadata={
                    "entity_type": entity_type,
                    "tags": tags,
                    "limit": limit
                }
            )

        except Exception as e:
            raise ProviderError(
                message=f"Failed to search entities in Neo4j: {str(e)}",
                context=ErrorContext.create(
                    flow_name="neo4j_provider",
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

    async def _find_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Find entities by type in Neo4j."""
        results = await self._execute_query(
            """
            MATCH (e:Entity)
            WHERE e.type = $type
            RETURN e.id as id, e as entity
            """,
            {"type": entity_type}
        )

        return [{"id": r["id"], "entity": r["entity"]} for r in results] 