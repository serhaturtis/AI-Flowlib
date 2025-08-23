"""Knowledge retrieval flow for querying domain-specific knowledge."""

import logging
from typing import List, Optional

from ....flows.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry
# Removed ProviderType import - using config-driven provider access
from flowlib.resources.registry.registry import resource_registry
from flowlib.resources.models.constants import ResourceType

from .models import (
    KnowledgeRetrievalInput,
    KnowledgeRetrievalOutput,
    RetrievedKnowledge
)

logger = logging.getLogger(__name__)


@flow(
    name="knowledge-retrieval",
    description="Retrieve knowledge from plugins and agent memory using intelligent search",
    is_infrastructure=False
)
class KnowledgeRetrievalFlow:
    """Flow that retrieves knowledge from multiple sources for answering queries.
    
    This flow searches across the agent's memory system and knowledge plugins
    to find relevant information for a given query. It synthesizes results
    from multiple sources and provides a comprehensive response.
    """

    @pipeline(
        input_model=KnowledgeRetrievalInput,
        output_model=KnowledgeRetrievalOutput
    )
    async def run_pipeline(self, input_data: KnowledgeRetrievalInput) -> KnowledgeRetrievalOutput:
        """Retrieve knowledge for the given query.
        
        Args:
            input_data: Query and search parameters
            
        Returns:
            Retrieved knowledge from various sources with synthesis
        """
        logger.info(f"Starting knowledge retrieval for query: '{input_data.query}'")
        
        try:
            retrieved_knowledge = []
            sources_searched = []
            
            # Search agent memory if requested
            if input_data.include_memory:
                memory_results = await self._search_agent_memory(input_data)
                retrieved_knowledge.extend(memory_results)
                sources_searched.append("agent_memory")
                logger.debug(f"Found {len(memory_results)} results from agent memory")
            
            # Search knowledge plugins if requested
            if input_data.include_plugins:
                plugin_results = await self._search_knowledge_plugins(input_data)
                retrieved_knowledge.extend(plugin_results)
                sources_searched.append("knowledge_plugins")
                logger.debug(f"Found {len(plugin_results)} results from knowledge plugins")
            
            # Synthesize and rank results
            synthesized_results = await self._synthesize_results(
                retrieved_knowledge, 
                input_data,
                sources_searched
            )
            
            # Limit results to requested maximum
            final_results = synthesized_results[:input_data.max_results]
            
            result = KnowledgeRetrievalOutput(
                retrieved_knowledge=final_results,
                search_summary=f"Searched {len(sources_searched)} sources and found {len(retrieved_knowledge)} total results. "
                             f"Returning top {len(final_results)} most relevant items.",
                sources_searched=sources_searched,
                total_results=len(retrieved_knowledge)
            )
            
            logger.info(f"Knowledge retrieval completed: {len(final_results)} results returned")
            return result
            
        except Exception as e:
            logger.error(f"Knowledge retrieval failed: {e}")
            return KnowledgeRetrievalOutput(
                retrieved_knowledge=[],
                search_summary=f"Knowledge retrieval failed: {str(e)}",
                sources_searched=[],
                total_results=0
            )
    
    async def _search_agent_memory(self, input_data: KnowledgeRetrievalInput) -> List[RetrievedKnowledge]:
        """Search the agent's memory system.
        
        Args:
            input_data: Search parameters
            
        Returns:
            Retrieved knowledge from agent memory
        """
        try:
            # Note: This would typically be injected as a dependency
            # For now, we'll simulate memory search
            logger.debug("Searching agent memory...")
            
            # In a real implementation, this would use the AgentMemory
            # search functionality that we just integrated
            
            return []  # Placeholder - would be implemented with actual memory integration
            
        except Exception as e:
            logger.warning(f"Agent memory search failed: {e}")
            return []
    
    async def _search_knowledge_plugins(self, input_data: KnowledgeRetrievalInput) -> List[RetrievedKnowledge]:
        """Search knowledge plugins.
        
        Args:
            input_data: Search parameters
            
        Returns:
            Retrieved knowledge from plugins
        """
        try:
            # Import the global plugin manager instance
            from ....providers.knowledge.plugin_manager import plugin_manager
            
            logger.debug("Searching knowledge plugins...")
            
            # Determine domains to search
            domains_to_search = []
            if input_data.domain:
                # Search specific domain
                domains_to_search = [input_data.domain]
            else:
                # Search all available domains
                domains_to_search = plugin_manager.get_available_domains()
            
            if not domains_to_search:
                logger.info("No domains available for plugin search")
                return []
            
            all_results = []
            
            # Search each domain
            for domain in domains_to_search:
                try:
                    # Use plugin manager to query the domain
                    knowledge_results = await plugin_manager.query_domain(
                        domain=domain,
                        query=input_data.query,
                        limit=input_data.max_results
                    )
                    
                    # Convert Knowledge objects to RetrievedKnowledge
                    for knowledge in knowledge_results:
                        retrieved = RetrievedKnowledge(
                            content=knowledge.content,
                            source=f"plugin:{knowledge.source}",
                            domain=knowledge.domain,
                            confidence=knowledge.confidence,
                            metadata={
                                "plugin_source": knowledge.source,
                                "original_metadata": knowledge.metadata
                            }
                        )
                        all_results.append(retrieved)
                    
                    logger.debug(f"Domain '{domain}' returned {len(knowledge_results)} results")
                    
                except Exception as e:
                    logger.warning(f"Failed to search domain '{domain}': {e}")
                    continue
            
            logger.info(f"Plugin search completed: {len(all_results)} total results")
            return all_results
            
        except Exception as e:
            logger.warning(f"Knowledge plugin search failed: {e}")
            return []
    
    async def _synthesize_results(
        self, 
        results: List[RetrievedKnowledge], 
        input_data: KnowledgeRetrievalInput,
        sources_searched: List[str]
    ) -> List[RetrievedKnowledge]:
        """Synthesize and rank results from multiple sources.
        
        Args:
            results: Raw results from all sources
            input_data: Original query parameters
            sources_searched: List of sources that were searched
            
        Returns:
            Ranked and potentially merged results
        """
        try:
            if not results:
                return []
            
            # Simple ranking by confidence score
            # In a more sophisticated implementation, this could use:
            # - LLM-based relevance scoring
            # - Semantic similarity
            # - Source reliability weighting
            sorted_results = sorted(results, key=lambda x: x.confidence, reverse=True)
            
            # Remove duplicates based on content similarity
            unique_results = []
            seen_content = set()
            
            for result in sorted_results:
                # Simple duplicate detection based on first 100 characters
                content_key = result.content[:100].lower().strip()
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_results.append(result)
            
            logger.debug(f"Deduplicated {len(results)} results to {len(unique_results)} unique items")
            return unique_results
            
        except Exception as e:
            logger.warning(f"Result synthesis failed: {e}")
            return results  # Return original results if synthesis fails