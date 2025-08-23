"""Semantic recall flow for meaning-based memory retrieval."""

from .....flows.decorators import flow, pipeline
from ....core.errors import ExecutionError, ErrorContext
from flowlib.providers.core.registry import provider_registry
# Removed ProviderType import - using config-driven provider access
from flowlib.resources.registry.registry import resource_registry
from flowlib.resources.models.constants import ResourceType
from ..flows import BaseRecallFlow
from ..models import RecallRequest, RecallResponse, RecallStrategy, MemoryMatch
from .models import SemanticAnalysis


@flow(
    name="semantic-recall",
    description="Flow for semantic-based memory recall that matches based on meaning",
    is_infrastructure=True
)
class SemanticRecallFlow(BaseRecallFlow):
    """Flow for semantic-based memory recall"""
    
    def __init__(self):
        """Initialize the semantic recall flow."""
        super().__init__()
    
    async def analyze_semantic_query(self, request: RecallRequest) -> SemanticAnalysis:
        """Analyze the query for semantic understanding"""
        # Get LLM provider
        llm = await provider_registry.get_by_config("default-llm")
        if not llm:
            raise ExecutionError(
                "Could not get LLM provider 'llamacpp'",
                ErrorContext.create(
                    flow_name="semantic_recall",
                    error_type="ProviderError",
                    error_location="analyze_semantic_query",
                    component="SemanticRecallFlow",
                    operation="provider_access"
                )
            )
        
        # Get prompt template
        prompt = resource_registry.get("semantic_analysis")
        if not prompt:
            raise ExecutionError(
                "Could not find semantic_analysis prompt",
                ErrorContext.create(
                    flow_name="semantic_recall",
                    error_type="ResourceError",
                    error_location="analyze_semantic_query",
                    component="SemanticRecallFlow",
                    operation="resource_access"
                )
            )
        
        # Prepare prompt variables
        prompt_vars = {
            "query": request.query,
            "context": request.context or "No specific context provided"
        }
        
        # Use generate_structured following conventions
        result = await llm.generate_structured(
            prompt=prompt,
            output_type=SemanticAnalysis,
            model_name="agent-model-small",
            prompt_variables=prompt_vars
        )
        
        return result
    
    async def execute_semantic_recall(self, request: RecallRequest, semantic_analysis: SemanticAnalysis) -> RecallResponse:
        """Execute semantic recall based on analysis"""
        # TODO: Implement proper semantic search through provider registry
        # For now, return empty results as placeholder
        matches = []
        
        return RecallResponse(
            matches=matches,
            strategy_used=RecallStrategy.SEMANTIC,
            total_matches=len(matches),
            query_analysis={
                "semantic_concepts": semantic_analysis.key_concepts,
                "confidence": semantic_analysis.confidence
            }
        )
    
    @pipeline(input_model=RecallRequest, output_model=RecallResponse)
    async def run_pipeline(self, input_data: RecallRequest) -> RecallResponse:
        """Run the semantic recall pipeline.
        
        Args:
            input_data: The recall request
            
        Returns:
            Recall response with semantically matched memories
        """
        # Execute validation
        validated_request = await self.validate_request(input_data)
        
        # Analyze semantic aspects
        semantic_analysis = await self.analyze_semantic_query(validated_request)
        
        # Execute semantic recall
        result = await self.execute_semantic_recall(validated_request, semantic_analysis)
        
        return result