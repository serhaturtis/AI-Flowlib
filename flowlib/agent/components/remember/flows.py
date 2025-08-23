from typing import Optional
from ....flows.base import Flow
from ....flows.decorators import flow, pipeline
from ...core.errors import ExecutionError
from flowlib.providers.core.registry import provider_registry
# Removed ProviderType import - using config-driven provider access
from flowlib.resources.registry.registry import resource_registry
from flowlib.resources.models.constants import ResourceType
from .models import RecallRequest, RecallResponse, RecallStrategy, MemoryMatch, ContextAnalysis

@flow(
    name="BaseRecallFlow",
    description="Base flow for memory recall operations that provides core recall functionality",
    is_infrastructure=True
)
class BaseRecallFlow(Flow):
    """Base flow for memory recall operations"""
    
    def __init__(self):
        """Initialize the base recall flow."""
        super().__init__(
            name_or_instance="BaseRecallFlow",
            input_schema=None,
            output_schema=None,
            metadata={"is_infrastructure": True}
        )
    
    async def validate_request(self, request: RecallRequest) -> RecallRequest:
        """Validate the recall request"""
        if request.strategy == RecallStrategy.ENTITY and not request.entity_id:
            raise ExecutionError("Entity ID required for entity-based recall", flow="BaseRecallFlow")
        return request
    
    async def execute_recall(self, request: RecallRequest) -> RecallResponse:
        """Execute the recall strategy - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement execute_recall")
    
    @pipeline(input_model=RecallRequest, output_model=RecallResponse)
    async def run_pipeline(self, input_data: RecallRequest) -> RecallResponse:
        """Run the recall pipeline.
        
        Args:
            input_data: The recall request
            
        Returns:
            Recall response
        """
        # Execute validation
        validated_request = await self.validate_request(input_data)
        
        # Execute recall
        result = await self.execute_recall(validated_request)
        
        return result

@flow(
    name="ContextualRecallFlow",
    description="Flow for context-aware memory recall that considers surrounding context",
    is_infrastructure=True
)
class ContextualRecallFlow(BaseRecallFlow):
    """Flow for context-based memory recall"""
    
    def __init__(self):
        """Initialize the contextual recall flow."""
        super().__init__()
    
    async def analyze_context(self, request: RecallRequest) -> ContextAnalysis:
        """Analyze the context to determine recall strategy"""
        # Get LLM provider using config-driven approach
        llm = await provider_registry.get_by_config("default-llm")
        
        # Get prompt template
        prompt = resource_registry.get("context_analysis")
        if not prompt:
            raise ExecutionError("Could not find context_analysis prompt", flow="ContextualRecallFlow")
        
        # Prepare prompt variables
        prompt_vars = {
            "context": request.context or "No specific context provided",
            "query": request.query
        }
        
        # Use generate_structured following conventions
        result = await llm.generate_structured(
            prompt=prompt,
            output_type=ContextAnalysis,
            prompt_variables=prompt_vars
        )
        
        return result
    
    async def execute_recall(self, request: RecallRequest) -> RecallResponse:
        """Execute contextual recall"""
        # Get memory provider through registry - fail fast if not available
        from flowlib.providers.core.registry import provider_registry
        
        try:
            memory_provider = await provider_registry.get_by_config("default-vector-db")
            if not memory_provider:
                raise RuntimeError("No default-vector-db provider configured for memory recall")
            
            # Search for relevant memories using vector similarity
            # This assumes the vector provider can store and search memories
            search_results = await memory_provider.search(
                query_text=request.query,
                limit=request.context_limit if hasattr(request, 'context_limit') else 10
            )
            
            # Convert search results to memory objects
            memories = []
            for result in search_results:
                # Extract required fields with strict validation
                if 'id' not in result:
                    raise ValueError("Search result missing required 'id' field")
                if 'content' not in result:
                    raise ValueError("Search result missing required 'content' field")
                    
                memories.append(type('Memory', (), {
                    'id': result['id'],
                    'content': result['content'],
                    'type': result['type'] if 'type' in result else 'contextual',
                    'relevance': result['score'] if 'score' in result else 0.0,
                    'metadata': result['metadata'] if 'metadata' in result else {}
                })())
                
        except Exception as e:
            # Log error but don't mask it - fail fast according to CLAUDE.md
            logger.error(f"Memory recall failed: {str(e)}")
            raise RuntimeError(f"Memory provider access failed: {str(e)}") from e
        
        # Convert memories to matches
        matches = [
            MemoryMatch(
                memory_id=m.id,
                content=m.content,
                memory_type=m.type,
                relevance_score=m.relevance,
                metadata=m.metadata
            )
            for m in memories
        ]
        
        return RecallResponse(
            matches=matches,
            strategy_used=RecallStrategy.CONTEXTUAL,
            total_matches=len(matches),
            query_analysis={"context_analysis": request.context}
        )
    
    @pipeline(input_model=RecallRequest, output_model=RecallResponse)
    async def run_pipeline(self, input_data: RecallRequest) -> RecallResponse:
        """Run the contextual recall pipeline.
        
        Args:
            input_data: The recall request
            
        Returns:
            Recall response with matches based on context
        """
        # Execute validation
        validated_request = await self.validate_request(input_data)
        
        # Execute context analysis
        analysis_result = await self.analyze_context(validated_request)
        
        # Execute recall
        result = await self.execute_recall(validated_request)
        
        return result 