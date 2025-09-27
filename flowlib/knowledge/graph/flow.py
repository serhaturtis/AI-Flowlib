"""Graph storage flow for knowledge base."""

import logging
from typing import Dict, Any, cast
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.graph.models import Entity as ProviderEntity
from flowlib.providers.graph.models import EntityRelationship

from flowlib.knowledge.models import (
    Entity,
    Relationship,
    DocumentProcessingResult,
)
from flowlib.providers.graph.base import GraphDBProvider, GraphDBProviderSettings
from flowlib.providers.graph.models import RelationshipSearchResult
from flowlib.knowledge.graph.models import (
    GraphStoreInput,
    GraphStoreOutput,
    GraphNode,
    GraphEdge,
    GraphStatistics,
)
from flowlib.knowledge.graph.models import (
    GraphEntity,
    GraphEntityAttribute,
    GraphRelationship,
)

logger = logging.getLogger(__name__)


@flow(name="graph-storage-flow", description="Store entities and relationships in graph database")  # type: ignore[arg-type]
class GraphStorageFlow:
    """Flow for creating knowledge graph in Neo4j."""
    
    async def stream_upsert(self, extraction_result: DocumentProcessingResult, db_path: str) -> None:
        """Stream upsert entities and relationships to persistent graph database."""
        
        logger.info(f"Streaming upsert: {len(extraction_result.entities)} entities, {len(extraction_result.relationships)} relationships")
        
        # Get provider (reuse connection if available)
        if not hasattr(self, '_streaming_graph_provider'):
            self._streaming_graph_provider = await self._get_streaming_provider(db_path)
        
        try:
            # Convert and stream entities
            if extraction_result.entities:
                # Convert knowledge entities to provider entities
                provider_entities = [self._convert_to_provider_entity(entity) for entity in extraction_result.entities]
                await self._streaming_graph_provider.bulk_add_entities(provider_entities)
                logger.debug(f"Streamed {len(provider_entities)} entities to graph DB")
            
            # Convert and stream relationships
            if extraction_result.relationships:
                for rel in extraction_result.relationships:
                    provider_rel = self._convert_to_provider_relationship(rel)
                    await self._streaming_graph_provider.add_relationship(
                        source_id=rel.source_entity_id,
                        target_entity=rel.target_entity_id,
                        relation_type=rel.relationship_type.value,
                        relationship=provider_rel
                    )
                logger.debug(f"Streamed {len(extraction_result.relationships)} relationships to graph DB")
                
        except Exception as e:
            logger.error(f"Failed to stream upsert to graph DB: {e}")
            raise

    async def finalize_streaming_graph(self, db_path: str) -> dict:
        """Finalize streaming graph database."""
        
        logger.info("Finalizing streaming graph database")
        
        try:
            # Get final statistics
            stats = {"total_nodes": 0, "total_edges": 0}
            if hasattr(self, '_streaming_graph_provider'):
                graph_stats = await self._streaming_graph_provider.get_stats()
                # Convert GraphStats to dict format expected by return type
                stats = {
                    "total_nodes": graph_stats.total_entities,
                    "total_edges": graph_stats.total_relationships
                }
                
                # Cleanup streaming provider
                await self._streaming_graph_provider.shutdown()
                delattr(self, '_streaming_graph_provider')
            
            return {
                "status": "finalized",
                "db_path": db_path,
                "statistics": stats
            }
            
        except Exception as e:
            logger.error(f"Failed to finalize streaming graph: {e}")
            raise

    async def _get_streaming_provider(self, db_path: str) -> GraphDBProvider:
        """Get streaming-optimized graph provider with persistent connection."""

        logger.debug(f"Initializing streaming graph provider at: {db_path}")

        # Use memory graph provider for testing/development
        from flowlib.providers.graph.memory_graph import MemoryGraphProvider

        provider = MemoryGraphProvider(
            name="streaming_memory_graph",
            provider_type="memory_graph",
            settings=GraphDBProviderSettings()
        )
        await provider.initialize()

        return provider

    async def _get_provider(self, config: GraphStoreInput) -> GraphDBProvider:
        """Get initialized graph database provider."""
        logger.info(f"Initializing graph provider: {config.graph_provider_name}")
        
        # Use memory graph provider for testing/development
        from flowlib.providers.graph.memory_graph import MemoryGraphProvider

        provider = MemoryGraphProvider(
            name="mock_memory_graph",
            provider_type="memory_graph",
            settings=GraphDBProviderSettings()
        )
        await provider.initialize()

        return provider
    
    def _convert_to_provider_entity(self, knowledge_entity: Entity) -> 'ProviderEntity':
        """Convert knowledge Entity to provider Entity."""
        from flowlib.providers.graph.models import Entity as ProviderEntity, EntityAttribute

        # Convert entity attributes
        attributes = {}
        for key, value in knowledge_entity.properties.items():
            attributes[key] = EntityAttribute(
                name=key,
                value=str(value),
                confidence=knowledge_entity.confidence,
                source="knowledge_extraction"
            )

        return ProviderEntity(
            id=knowledge_entity.entity_id,
            type=knowledge_entity.entity_type.value,
            attributes=attributes,
            relationships=[],  # Will be populated separately
            tags=list(knowledge_entity.properties.keys()) if knowledge_entity.properties else [],
            source="knowledge_extraction",
            importance=knowledge_entity.confidence,
            vector_id=None
        )

    def _convert_to_provider_relationship(self, knowledge_rel: Relationship) -> 'EntityRelationship':
        """Convert knowledge Relationship to provider EntityRelationship."""
        from flowlib.providers.graph.models import EntityRelationship

        return EntityRelationship(
            relation_type=knowledge_rel.relationship_type.value,
            target_entity=knowledge_rel.target_entity_id,
            confidence=knowledge_rel.confidence,
            source="knowledge_extraction"
        )

    def _convert_to_graph_entity(self, entity: Entity) -> GraphEntity:
        """Convert knowledge Entity to GraphEntity."""
        # Create attributes from entity properties
        attributes = []
        
        # Add basic attributes
        attributes.append(GraphEntityAttribute(
            name="entity_type",
            value=entity.entity_type.value,
            type="string"
        ))
        
        if entity.description:
            attributes.append(GraphEntityAttribute(
                name="description",
                value=entity.description,
                type="string"
            ))
        
        # Add confidence and frequency
        attributes.append(GraphEntityAttribute(
            name="confidence",
            value=entity.confidence,
            type="float"
        ))
        
        attributes.append(GraphEntityAttribute(
            name="frequency",
            value=entity.frequency,
            type="integer"
        ))
        
        # Add document count
        attributes.append(GraphEntityAttribute(
            name="document_count",
            value=len(entity.documents),
            type="integer"
        ))
        
        # Add source documents as comma-separated list
        if entity.documents:
            attributes.append(GraphEntityAttribute(
                name="source_documents",
                value=",".join(entity.documents[:5]),  # Limit to first 5
                type="string"
            ))
        
        # Convert tags to list
        tags = list(entity.properties.keys()) if entity.properties else []
        
        return GraphEntity(
            id=entity.entity_id,
            type=entity.entity_type.value,
            label=entity.name,
            attributes=attributes,
            relationships=[],  # Will be populated separately
            tags=tags
        )
    
    def _create_graph_relationship(self, relationship: Relationship) -> GraphRelationship:
        """Create graph relationship from knowledge relationship."""
        # Calculate confidence string
        confidence = relationship.confidence
        if confidence >= 0.8:
            confidence_str = "high"
        elif confidence >= 0.5:
            confidence_str = "medium"
        else:
            confidence_str = "low"
        
        return GraphRelationship(
            id=relationship.relationship_id,
            source_id=relationship.source_entity_id,
            target_id=relationship.target_entity_id,
            type=relationship.relationship_type.value,
            label=relationship.description or relationship.relationship_type.value,
            attributes={
                "confidence": confidence,
                "frequency": relationship.frequency,
                "document_count": len(relationship.documents),
                "confidence_level": confidence_str
            },
            confidence=confidence
        )
    
    async def _build_graph(self, input_data: GraphStoreInput) -> GraphStoreOutput:
        """Build knowledge graph from entities and relationships."""
        logger.info(f"Building graph with {len(input_data.entities)} entities and {len(input_data.relationships)} relationships")
        
        # Get provider
        graph_provider = await self._get_provider(input_data)
        
        try:
            # Create nodes for documents
            doc_nodes = []
            for doc in input_data.documents:
                GraphEntity(
                    id=doc.document_id,
                    type="DOCUMENT",
                    label=doc.metadata.file_name,
                    attributes=[
                        GraphEntityAttribute(name="file_type", value=doc.metadata.file_type, type="string"),
                        GraphEntityAttribute(name="file_size", value=doc.metadata.file_size, type="integer"),
                        GraphEntityAttribute(name="created_date", value=doc.metadata.created_date or "", type="string"),
                        GraphEntityAttribute(name="language", value=doc.language_detected, type="string"),
                        GraphEntityAttribute(name="status", value=doc.status.value, type="string"),
                        GraphEntityAttribute(name="chunk_count", value=len(doc.chunks), type="integer"),
                        GraphEntityAttribute(name="reading_time", value=doc.reading_time_minutes, type="float"),
                    ],
                    tags=["document", doc.metadata.file_type.lower()]
                )
                
                doc_nodes.append(GraphNode(
                    node_id=doc.document_id,
                    node_type="document",
                    name=doc.metadata.file_name,
                    properties={
                        "file_type": doc.metadata.file_type,
                        "created_date": doc.metadata.created_date or ""
                    }
                ))
            
            # Convert and create entity nodes
            entity_nodes = []
            provider_entities = []
            for entity in input_data.entities:
                provider_entity = self._convert_to_provider_entity(entity)
                provider_entities.append(provider_entity)
                
                entity_nodes.append(GraphNode(
                    node_id=entity.entity_id,
                    node_type=entity.entity_type.value,
                    name=entity.name,
                    properties={
                        "confidence": entity.confidence,
                        "frequency": entity.frequency
                    }
                ))
            
            # Create all entities in graph (skip doc_nodes as they're not Entity objects)
            if provider_entities:
                await graph_provider.bulk_add_entities(provider_entities)
            
            # Convert and create relationships
            provider_relationships = []
            edges = []
            for relationship in input_data.relationships:
                provider_rel = self._convert_to_provider_relationship(relationship)
                provider_relationships.append(provider_rel)
                
                edges.append(GraphEdge(
                    edge_id=relationship.relationship_id,
                    source_node_id=relationship.source_entity_id,
                    target_node_id=relationship.target_entity_id,
                    relationship_type=relationship.relationship_type.value,
                    properties={
                        "confidence": relationship.confidence,
                        "frequency": relationship.frequency,
                        "description": relationship.description or relationship.relationship_type.value
                    }
                ))
            
            # Create relationships in graph
            if provider_relationships:
                for i, relationship in enumerate(input_data.relationships):
                    provider_rel = provider_relationships[i]
                    await graph_provider.add_relationship(
                        source_id=relationship.source_entity_id,
                        target_entity=relationship.target_entity_id,
                        relation_type=relationship.relationship_type.value,
                        relationship=provider_rel
                    )
            
            # Get statistics
            await graph_provider.get_stats()
            
            graph_stats = GraphStatistics(
                total_nodes=len(doc_nodes) + len(entity_nodes),
                total_edges=len(edges),
                node_types={
                    "documents": len(doc_nodes),
                    "entities": len(entity_nodes)
                },
                relationship_types={rel.relationship_type: 1 for rel in edges},  # Simplified count
                average_degree=2.0 * len(edges) / max(len(entity_nodes), 1) if entity_nodes else 0.0
            )
            
            return GraphStoreOutput(
                graph_name=input_data.graph_name,
                nodes_created=len(doc_nodes) + len(entity_nodes),
                edges_created=len(edges),
                graph_stats=graph_stats
            )
            
        finally:
            # Always shutdown provider
            await graph_provider.shutdown()
    
    async def _query_graph(self, input_data: GraphStoreInput) -> GraphStoreOutput:
        """Query existing graph and extract subgraphs."""
        logger.info(f"Querying graph: {input_data.graph_name}")
        
        # Get provider
        graph_provider = await self._get_provider(input_data)
        
        try:
            nodes = []
            edges = []
            
            # Get entities by query parameters
            if input_data.query_entity_id:
                # Get specific entity and its neighborhood
                entity_search_result = await graph_provider.search_entities(
                    query=input_data.query_entity_id
                )
                relationships = await graph_provider.query_relationships(
                    entity_id=input_data.query_entity_id
                )
            elif input_data.query_entity_type:
                # Get all entities of a specific type
                entity_search_result = await graph_provider.search_entities(
                    entity_type=input_data.query_entity_type,
                    limit=input_data.query_limit or 100
                )
                # Create empty RelationshipSearchResult
                relationships = RelationshipSearchResult(
                    success=True,
                    total_count=0,
                    source_entity=None,
                    target_entity=None,
                    execution_time_ms=None
                )
            else:
                # Get a sample of the graph
                entity_search_result = await graph_provider.search_entities(
                    limit=input_data.query_limit or 50
                )
                # Create empty RelationshipSearchResult
                relationships = RelationshipSearchResult(
                    success=True,
                    total_count=0,
                    source_entity=None,
                    target_entity=None,
                    execution_time_ms=None
                )

            # Convert to output format
            for entity in entity_search_result.entities:
                # Fail-fast approach - require essential fields
                if not entity.id:
                    raise ValueError("Entity missing required 'id' field")
                if not entity.type:
                    raise ValueError("Entity missing required 'type' field")

                # Use type for name if no specific label available
                entity_name = getattr(entity, 'label', entity.type)
                if not entity_name:
                    entity_name = entity.id

                nodes.append(GraphNode(
                    node_id=entity.id,
                    node_type=entity.type,
                    name=entity_name,
                    properties={"importance": entity.importance, "source": entity.source}
                ))
            
            for rel in relationships:
                # Ensure rel is a dictionary
                rel_dict = cast(Dict[str, object], rel)

                # Fail-fast approach - require essential relationship fields
                if "source_id" not in rel_dict:
                    raise ValueError("Relationship missing required 'source_id' field")
                if "target_id" not in rel_dict:
                    raise ValueError("Relationship missing required 'target_id' field")
                if "id" not in rel_dict:
                    raise ValueError("Relationship missing required 'id' field")
                if "type" not in rel_dict:
                    raise ValueError("Relationship missing required 'type' field")

                edges.append(GraphEdge(
                    edge_id=str(rel_dict["id"]),
                    source_node_id=str(rel_dict["source_id"]),
                    target_node_id=str(rel_dict["target_id"]),
                    relationship_type=str(rel_dict["type"]),
                    properties=cast(Dict[str, Any], rel_dict.get("properties", {}))
                ))
            
            # Create basic statistics
            basic_stats = GraphStatistics(
                total_nodes=len(nodes),
                total_edges=len(edges)
            )

            return GraphStoreOutput(
                graph_name=input_data.graph_name,
                nodes_created=len(nodes),
                edges_created=len(edges),
                graph_stats=basic_stats
            )
            
        finally:
            # Always shutdown provider
            await graph_provider.shutdown()
    
    async def _find_connections(self, input_data: GraphStoreInput) -> GraphStoreOutput:
        """Find connections between entities."""
        if not input_data.query_source_id or not input_data.query_target_id:
            raise ValueError("Both source and target entity IDs required")
        
        logger.info(f"Finding connections between {input_data.query_source_id} and {input_data.query_target_id}")
        
        # Get provider
        graph_provider = await self._get_provider(input_data)
        
        try:
            # Query for paths between entities - start with source entity relationships
            relationships = await graph_provider.query_relationships(
                entity_id=input_data.query_source_id,
                direction="outgoing"
            )
            
            # Extract unique nodes and edges from paths
            nodes = []
            edges = []
            seen_nodes = set()
            
            for rel in relationships:
                # Ensure rel is a dictionary
                rel_dict = cast(Dict[str, object], rel)

                # Fail-fast approach - require essential relationship fields for path finding
                if "source_id" not in rel_dict:
                    raise ValueError("Path relationship missing required 'source_id' field")
                if "target_id" not in rel_dict:
                    raise ValueError("Path relationship missing required 'target_id' field")
                if "id" not in rel_dict:
                    raise ValueError("Path relationship missing required 'id' field")
                if "type" not in rel_dict:
                    raise ValueError("Path relationship missing required 'type' field")

                source_id = str(rel_dict["source_id"])
                target_id = str(rel_dict["target_id"])
                
                # Add source node if not seen
                if source_id not in seen_nodes:
                    source_label = str(rel_dict.get("source_label", source_id))
                    nodes.append(GraphNode(
                        node_id=source_id,
                        node_type="entity",
                        name=source_label,
                        properties={}
                    ))
                    seen_nodes.add(source_id)

                # Add target node if not seen
                if target_id not in seen_nodes:
                    target_label = str(rel_dict.get("target_label", target_id))
                    nodes.append(GraphNode(
                        node_id=target_id,
                        node_type="entity",
                        name=target_label,
                        properties={}
                    ))
                    seen_nodes.add(target_id)
                
                # Add edge
                edges.append(GraphEdge(
                    edge_id=str(rel_dict["id"]),
                    source_node_id=source_id,
                    target_node_id=target_id,
                    relationship_type=str(rel_dict["type"]),
                    properties=cast(Dict[str, Any], rel_dict.get("properties", {}))
                ))
            
            # Create path statistics
            path_stats = GraphStatistics(
                total_nodes=len(nodes),
                total_edges=len(edges)
            )

            return GraphStoreOutput(
                graph_name=input_data.graph_name,
                nodes_created=len(nodes),
                edges_created=len(edges),
                graph_stats=path_stats
            )
            
        finally:
            # Always shutdown provider
            await graph_provider.shutdown()
    
    @pipeline(input_model=GraphStoreInput, output_model=GraphStoreOutput)
    async def run_pipeline(self, input_data: GraphStoreInput) -> GraphStoreOutput:
        """Execute graph storage pipeline."""
        # Determine which operation to perform
        if input_data.entities and input_data.relationships:
            # Build new graph
            return await self._build_graph(input_data)
        elif input_data.query_entity_id or input_data.query_entity_type:
            # Query existing graph
            return await self._query_graph(input_data)
        elif input_data.query_source_id and input_data.query_target_id:
            # Find connections between entities
            return await self._find_connections(input_data)
        else:
            raise ValueError("No valid operation specified in input")