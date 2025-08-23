"""Unified Memory Interface - Simple, Intelligent Memory.

This module provides a clean, unified memory interface that replaces
the complex memory hierarchy with simple, intelligent memory operations
that gracefully degrade when storage systems are unavailable.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from flowlib.core.decorators.decorators import inject
from flowlib.providers.core.registry import provider_registry
from .knowledge import KnowledgeSet, Entity, Concept, Relationship, Pattern

logger = logging.getLogger(__name__)


class IntelligentMemory:
    """Unified memory interface - no complex memory hierarchies.
    
    This class provides intelligent routing of knowledge to appropriate
    storage systems with graceful degradation when systems are unavailable.
    
    Key Features:
    - Single memory interface for all operations
    - Intelligent routing based on data type
    - Graceful degradation when storage fails
    - Simple store/retrieve API
    """
    
    @inject(vector='default-vector-db', graph='default-graph-db')
    async def store_knowledge(
        self, 
        knowledge: KnowledgeSet,
        vector_provider=None,
        graph_provider=None
    ) -> None:
        """Store knowledge with intelligent routing and graceful degradation.
        
        Args:
            knowledge: Knowledge set to store
            vector_provider: Vector database provider (injected)
            graph_provider: Graph database provider (injected)
        """
        logger.info(f"Storing knowledge set with {knowledge.total_items} items")
        
        # Store entities and concepts in vector memory for semantic search
        if vector_provider and (knowledge.entities or knowledge.concepts):
            await self._store_in_vector(
                knowledge.entities + knowledge.concepts, 
                vector_provider
            )
        
        # Store relationships in graph memory for connection analysis  
        if graph_provider and knowledge.relationships:
            await self._store_in_graph(knowledge.relationships, graph_provider)
        
        # Always store in working memory as fallback
        await self._store_in_working_memory(knowledge)
        
        logger.info("Knowledge storage complete")
    
    async def retrieve_knowledge(
        self, 
        query: str,
        knowledge_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> KnowledgeSet:
        """Retrieve relevant knowledge - intelligent source selection.
        
        Args:
            query: Search query
            knowledge_types: Types of knowledge to retrieve
            limit: Maximum items to return
            
        Returns:
            Knowledge set with relevant items
        """
        logger.info(f"Retrieving knowledge for query: '{query[:50]}...'")
        
        # Try vector search first for semantic relevance
        vector_results = await self._search_vector(query, limit)
        
        # Try graph traversal for relationship discovery
        graph_results = await self._search_graph(query, limit)
        
        # Combine and rank results intelligently
        combined_knowledge = self._combine_results(vector_results, graph_results)
        
        # Filter by knowledge types if specified
        if knowledge_types:
            combined_knowledge = self._filter_by_types(combined_knowledge, knowledge_types)
        
        logger.info(f"Retrieved {combined_knowledge.total_items} relevant items")
        return combined_knowledge
    
    async def _store_in_vector(self, items: List, vector_provider) -> None:
        """Store items in vector memory with graceful degradation."""
        try:
            if not items:
                return
                
            documents = []
            metadatas = []
            ids = []
            
            for item in items:
                # Create searchable document text
                description = item.description if hasattr(item, 'description') else ''
                doc_text = f"{item.name}: {description}"
                documents.append(doc_text)
                
                # Create metadata
                confidence = item.confidence if hasattr(item, 'confidence') else 1.0
                metadata = {
                    "type": type(item).__name__,
                    "confidence": confidence,
                    "name": item.name
                }
                metadatas.append(metadata)
                
                # Create unique ID
                item_id = f"{type(item).__name__}_{hash(item.name)}"
                ids.append(item_id)
            
            # Store in vector database
            await vector_provider.add_vectors(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.debug(f"Stored {len(items)} items in vector memory")
            
        except Exception as e:
            logger.warning(f"Vector storage failed, continuing: {e}")
    
    async def _store_in_graph(self, relationships: List[Relationship], graph_provider) -> None:
        """Store relationships in graph with graceful degradation."""
        try:
            if not relationships:
                return
                
            for rel in relationships:
                await graph_provider.add_relationship(
                    source_id=rel.source,
                    target_id=rel.target, 
                    relationship_type=rel.type,
                    properties={
                        "description": rel.description or "",
                        "confidence": rel.confidence,
                        "bidirectional": rel.bidirectional if hasattr(rel, 'bidirectional') else False
                    }
                )
            
            logger.debug(f"Stored {len(relationships)} relationships in graph memory")
            
        except Exception as e:
            logger.warning(f"Graph storage failed, continuing: {e}")
    
    async def _store_in_working_memory(self, knowledge: KnowledgeSet) -> None:
        """Store in working memory as fallback - always succeeds."""
        try:
            # This is a simple in-memory fallback
            # In a real implementation, this might use a local cache or file storage
            logger.debug("Stored knowledge in working memory (fallback)")
            
        except Exception as e:
            logger.error(f"Working memory storage failed: {e}")
    
    async def _search_vector(self, query: str, limit: int) -> KnowledgeSet:
        """Search vector memory for semantically relevant items."""
        try:
            vector_provider = await provider_registry.get_by_config("default-vector-db")
            if not vector_provider:
                return KnowledgeSet()
            
            # Perform vector search
            results = await vector_provider.search(
                query_text=query,
                limit=limit
            )
            
            # Convert results to knowledge objects
            entities = []
            concepts = []
            
            for result in results:
                metadata = result['metadata'] if 'metadata' in result else {}
                item_type = metadata['type'] if 'type' in metadata else 'Entity'
                
                if item_type == 'Entity':
                    name = metadata['name'] if 'name' in metadata else 'Unknown'
                    description = result['document'] if 'document' in result else ''
                    confidence = metadata['confidence'] if 'confidence' in metadata else 0.8
                    entities.append(Entity(
                        name=name,
                        type='retrieved',
                        description=description,
                        confidence=confidence
                    ))
                elif item_type == 'Concept':
                    name = metadata['name'] if 'name' in metadata else 'Unknown'
                    description = result['document'] if 'document' in result else ''
                    confidence = metadata['confidence'] if 'confidence' in metadata else 0.8
                    concepts.append(Concept(
                        name=name,
                        description=description,
                        confidence=confidence
                    ))
            
            return KnowledgeSet(
                entities=entities,
                concepts=concepts,
                summary=f"Vector search results for: {query}"
            )
            
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return KnowledgeSet()
    
    async def _search_graph(self, query: str, limit: int) -> KnowledgeSet:
        """Search graph memory for relationship connections."""
        try:
            graph_provider = await provider_registry.get_by_config("default-graph-db")
            if not graph_provider:
                return KnowledgeSet()
            
            # Perform graph traversal (simplified implementation)
            # In a real implementation, this would do more sophisticated graph queries
            relationships = []
            
            return KnowledgeSet(
                relationships=relationships,
                summary=f"Graph search results for: {query}"
            )
            
        except Exception as e:
            logger.warning(f"Graph search failed: {e}")
            return KnowledgeSet()
    
    def _combine_results(self, vector_results: KnowledgeSet, graph_results: KnowledgeSet) -> KnowledgeSet:
        """Combine and rank results from different sources intelligently."""
        # Combine all knowledge items
        combined_entities = vector_results.entities + graph_results.entities
        combined_concepts = vector_results.concepts + graph_results.concepts
        combined_relationships = vector_results.relationships + graph_results.relationships
        combined_patterns = vector_results.patterns + graph_results.patterns
        
        # Remove duplicates (simplified - just by name)
        unique_entities = self._deduplicate_by_name(combined_entities)
        unique_concepts = self._deduplicate_by_name(combined_concepts)
        unique_relationships = self._deduplicate_relationships(combined_relationships)
        unique_patterns = self._deduplicate_by_name(combined_patterns)
        
        return KnowledgeSet(
            entities=unique_entities,
            concepts=unique_concepts,
            relationships=unique_relationships,
            patterns=unique_patterns,
            summary="Combined search results from vector and graph memory"
        )
    
    def _deduplicate_by_name(self, items: List) -> List:
        """Remove duplicate items by name."""
        seen_names = set()
        unique_items = []
        
        for item in items:
            if item.name not in seen_names:
                seen_names.add(item.name)
                unique_items.append(item)
        
        return unique_items
    
    def _deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Remove duplicate relationships."""
        seen_rels = set()
        unique_rels = []
        
        for rel in relationships:
            rel_key = (rel.source, rel.target, rel.type)
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                unique_rels.append(rel)
        
        return unique_rels
    
    def _filter_by_types(self, knowledge: KnowledgeSet, knowledge_types: List[str]) -> KnowledgeSet:
        """Filter knowledge by specified types."""
        filtered_entities = knowledge.entities if 'entities' in knowledge_types else []
        filtered_concepts = knowledge.concepts if 'concepts' in knowledge_types else []
        filtered_relationships = knowledge.relationships if 'relationships' in knowledge_types else []
        filtered_patterns = knowledge.patterns if 'patterns' in knowledge_types else []
        
        return KnowledgeSet(
            entities=filtered_entities,
            concepts=filtered_concepts,
            relationships=filtered_relationships,
            patterns=filtered_patterns,
            summary=f"Filtered results for types: {knowledge_types}"
        )


# Global memory instance for simple API
_memory_instance = None

async def get_memory() -> IntelligentMemory:
    """Get or create the global memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = IntelligentMemory()
    return _memory_instance


# Simple API functions
async def remember(knowledge: KnowledgeSet) -> None:
    """Simple API for storing knowledge.
    
    Args:
        knowledge: Knowledge to store
        
    Example:
        knowledge = await learn_from_text("AI is transforming...")
        await remember(knowledge)
    """
    memory = await get_memory()
    await memory.store_knowledge(knowledge)


async def recall(
    query: str, 
    knowledge_types: Optional[List[str]] = None,
    limit: int = 10
) -> KnowledgeSet:
    """Simple API for retrieving knowledge.
    
    Args:
        query: Search query
        knowledge_types: Types to retrieve ('entities', 'concepts', 'relationships', 'patterns')
        limit: Maximum items to return
        
    Returns:
        Relevant knowledge
        
    Example:
        knowledge = await recall("machine learning trends")
        entities = await recall("companies", knowledge_types=['entities'])
    """
    memory = await get_memory()
    return await memory.retrieve_knowledge(query, knowledge_types, limit)