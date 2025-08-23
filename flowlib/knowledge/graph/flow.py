"""Graph storage flow for knowledge base."""

import logging
from typing import List, Dict, Optional, Set
from datetime import datetime
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry

from flowlib.knowledge.models.models import (
    Entity,
    Relationship,
    DocumentContent,
    EntityType as KnowledgeEntityType,
    RelationType as KnowledgeRelationType,
)
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


@flow(name="graph-storage-flow", description="Store entities and relationships in graph database")
class GraphStorageFlow:
    """Flow for creating knowledge graph in Neo4j."""
    
    async def stream_upsert(self, extraction_result, db_path: str) -> None:
        """Stream upsert entities and relationships to persistent graph database."""
        
        logger.info(f"Streaming upsert: {len(extraction_result.entities)} entities, {len(extraction_result.relationships)} relationships")
        
        # Get provider (reuse connection if available)
        if not hasattr(self, '_streaming_graph_provider'):
            self._streaming_graph_provider = await self._get_streaming_provider(db_path)
        
        try:
            # Convert and stream entities
            if extraction_result.entities:
                graph_entities = [self._convert_to_graph_entity(entity) for entity in extraction_result.entities]
                await self._streaming_graph_provider.create_entities(graph_entities)
                logger.debug(f"Streamed {len(graph_entities)} entities to graph DB")
            
            # Convert and stream relationships  
            if extraction_result.relationships:
                graph_relationships = [self._create_graph_relationship(rel) for rel in extraction_result.relationships]
                await self._streaming_graph_provider.create_relationships(graph_relationships)
                logger.debug(f"Streamed {len(graph_relationships)} relationships to graph DB")
                
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
                stats = await self._streaming_graph_provider.get_graph_stats()
                
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

    async def _get_streaming_provider(self, db_path: str):
        """Get streaming-optimized graph provider with persistent connection."""
        
        logger.debug(f"Initializing streaming graph provider at: {db_path}")
        
        # For now, use mock provider optimized for streaming
        class StreamingMockGraphProvider:
            def __init__(self, db_path: str):
                self.db_path = db_path
                self.connection_count = 0
                self.total_entities = 0
                self.total_relationships = 0
                
            async def initialize(self):
                self.connection_count += 1
                logger.debug(f"Streaming graph provider initialized (connection #{self.connection_count})")
                
            async def create_entities(self, entities: List):
                logger.debug(f"Streaming create: {len(entities)} entities")
                self.total_entities += len(entities)
                # In real implementation, this would maintain persistent Neo4j connection
                return len(entities)
                
            async def create_relationships(self, relationships: List):
                logger.debug(f"Streaming create: {len(relationships)} relationships")
                self.total_relationships += len(relationships)
                return len(relationships)
                
            async def get_graph_stats(self):
                return {
                    "total_nodes": self.total_entities,
                    "total_edges": self.total_relationships
                }
                
            async def shutdown(self):
                logger.debug(f"Streaming graph provider shutdown (processed {self.total_entities} entities, {self.total_relationships} relationships)")
        
        provider = StreamingMockGraphProvider(db_path)
        await provider.initialize()
        
        return provider

    async def _get_provider(self, config: GraphStoreInput):
        """Get initialized graph database provider."""
        logger.info(f"Initializing graph provider: {config.graph_provider_name}")
        
        # TODO: Update to config-driven provider access
        logger.warning("Using mock graph provider. Update to config-driven pattern.")
        
        class MockGraphProvider:
            async def initialize(self): pass
            async def create_entities(self, entities: List[GraphEntity]): 
                logger.info(f"Mock: Creating {len(entities)} entities")
                return len(entities)
            async def create_relationships(self, relationships: List[GraphRelationship]):
                logger.info(f"Mock: Creating {len(relationships)} relationships")
                return len(relationships)
            async def query_entities(self, entity_type=None, limit=100):
                return []
            async def query_relationships(self, source_id=None, target_id=None):
                return []
            async def get_graph_stats(self):
                return {"total_nodes": 0, "total_edges": 0}
            async def shutdown(self): pass
        
        return MockGraphProvider()
    
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
                doc_entity = GraphEntity(
                    id=doc.document_id,
                    type="DOCUMENT",
                    label=doc.metadata.file_name,
                    attributes=[
                        GraphEntityAttribute(name="file_type", value=doc.metadata.file_type, type="string"),
                        GraphEntityAttribute(name="file_size", value=doc.metadata.file_size, type="integer"),
                        GraphEntityAttribute(name="created_at", value=doc.metadata.created_at.isoformat(), type="string"),
                        GraphEntityAttribute(name="language", value=doc.language_detected, type="string"),
                        GraphEntityAttribute(name="status", value=doc.status.value, type="string"),
                        GraphEntityAttribute(name="chunk_count", value=len(doc.chunks), type="integer"),
                        GraphEntityAttribute(name="reading_time", value=doc.reading_time_minutes, type="float"),
                    ],
                    tags=["document", doc.metadata.file_type.lower()]
                )
                
                doc_nodes.append(GraphNode(
                    id=doc.document_id,
                    type="document",
                    label=doc.metadata.file_name,
                    properties={
                        "file_type": doc.metadata.file_type,
                        "created_at": doc.metadata.created_at.isoformat()
                    }
                ))
            
            # Convert and create entity nodes
            entity_nodes = []
            graph_entities = []
            for entity in input_data.entities:
                graph_entity = self._convert_to_graph_entity(entity)
                graph_entities.append(graph_entity)
                
                entity_nodes.append(GraphNode(
                    id=entity.entity_id,
                    type=entity.entity_type.value,
                    label=entity.name,
                    properties={
                        "confidence": entity.confidence,
                        "frequency": entity.frequency
                    }
                ))
            
            # Create all entities in graph
            if doc_nodes:
                await graph_provider.create_entities([doc_entity for doc_entity in doc_nodes])
            if graph_entities:
                await graph_provider.create_entities(graph_entities)
            
            # Convert and create relationships
            graph_relationships = []
            edges = []
            for relationship in input_data.relationships:
                graph_rel = self._create_graph_relationship(relationship)
                graph_relationships.append(graph_rel)
                
                edges.append(GraphEdge(
                    id=relationship.relationship_id,
                    source_id=relationship.source_entity_id,
                    target_id=relationship.target_entity_id,
                    type=relationship.relationship_type.value,
                    label=relationship.description or relationship.relationship_type.value,
                    properties={
                        "confidence": relationship.confidence,
                        "frequency": relationship.frequency
                    }
                ))
            
            # Create relationships in graph
            if graph_relationships:
                await graph_provider.create_relationships(graph_relationships)
            
            # Get statistics
            stats = await graph_provider.get_graph_stats()
            
            graph_stats = GraphStatistics(
                total_nodes=len(doc_nodes) + len(entity_nodes),
                total_edges=len(edges),
                node_types={
                    "documents": len(doc_nodes),
                    "entities": len(entity_nodes)
                },
                edge_types={rel.type: 1 for rel in edges},  # Simplified count
                avg_node_degree=2.0 * len(edges) / max(len(entity_nodes), 1) if entity_nodes else 0
            )
            
            return GraphStoreOutput(
                graph_name=input_data.graph_name,
                nodes_created=len(doc_nodes) + len(entity_nodes),
                edges_created=len(edges),
                nodes=doc_nodes + entity_nodes,
                edges=edges,
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
                entities = await graph_provider.query_entities(
                    entity_id=input_data.query_entity_id
                )
                relationships = await graph_provider.query_relationships(
                    source_id=input_data.query_entity_id
                )
            elif input_data.query_entity_type:
                # Get all entities of a specific type
                entities = await graph_provider.query_entities(
                    entity_type=input_data.query_entity_type,
                    limit=input_data.query_limit or 100
                )
                relationships = []
            else:
                # Get a sample of the graph
                entities = await graph_provider.query_entities(
                    limit=input_data.query_limit or 50
                )
                relationships = []
            
            # Convert to output format
            for entity in entities:
                # Fail-fast approach - require essential fields
                if "id" not in entity:
                    raise ValueError("Entity missing required 'id' field")
                if "type" not in entity:
                    raise ValueError("Entity missing required 'type' field")
                    
                nodes.append(GraphNode(
                    id=entity["id"],
                    type=entity["type"],
                    label=entity["label"] if "label" in entity else None,
                    properties=entity["properties"] if "properties" in entity else {}
                ))
            
            for rel in relationships:
                # Fail-fast approach - require essential relationship fields
                if "source_id" not in rel:
                    raise ValueError("Relationship missing required 'source_id' field")
                if "target_id" not in rel:
                    raise ValueError("Relationship missing required 'target_id' field")
                    
                edges.append(GraphEdge(
                    id=rel["id"] if "id" in rel else None,
                    source_id=rel["source_id"],
                    target_id=rel["target_id"],
                    type=rel["type"] if "type" in rel else None,
                    label=rel["label"] if "label" in rel else None,
                    properties=rel["properties"] if "properties" in rel else {}
                ))
            
            return GraphStoreOutput(
                graph_name=input_data.graph_name,
                nodes_created=0,
                edges_created=0,
                nodes=nodes,
                edges=edges
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
            # Query for paths between entities
            relationships = await graph_provider.query_relationships(
                source_id=input_data.query_source_id,
                target_id=input_data.query_target_id,
                max_hops=input_data.query_max_hops or 3
            )
            
            # Extract unique nodes and edges from paths
            nodes = []
            edges = []
            seen_nodes = set()
            
            for rel in relationships:
                # Fail-fast approach - require essential relationship fields for path finding
                if "source_id" not in rel:
                    raise ValueError("Path relationship missing required 'source_id' field")
                if "target_id" not in rel:
                    raise ValueError("Path relationship missing required 'target_id' field")
                    
                source_id = rel["source_id"]
                target_id = rel["target_id"]
                
                # Add source node if not seen
                if source_id not in seen_nodes:
                    source_label = rel["source_label"] if "source_label" in rel else source_id
                    nodes.append(GraphNode(
                        id=source_id,
                        type="entity",
                        label=source_label,
                        properties={}
                    ))
                    seen_nodes.add(source_id)
                
                # Add target node if not seen
                if target_id not in seen_nodes:
                    target_label = rel["target_label"] if "target_label" in rel else target_id
                    nodes.append(GraphNode(
                        id=target_id,
                        type="entity",
                        label=target_label,
                        properties={}
                    ))
                    seen_nodes.add(target_id)
                
                # Add edge
                edges.append(GraphEdge(
                    id=rel["id"] if "id" in rel else None,
                    source_id=source_id,
                    target_id=target_id,
                    type=rel["type"] if "type" in rel else None,
                    label=rel["label"] if "label" in rel else None,
                    properties=rel["properties"] if "properties" in rel else {}
                ))
            
            return GraphStoreOutput(
                graph_name=input_data.graph_name,
                nodes_created=0,
                edges_created=0,
                nodes=nodes,
                edges=edges,
                path_found=len(edges) > 0,
                path_length=len(edges) if edges else None
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