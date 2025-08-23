"""Entity recall flow for retrieving entity-specific information."""

from .....flows.decorators import flow, pipeline
from ....core.errors import ExecutionError, ErrorContext
from ..flows import BaseRecallFlow
from ..models import RecallRequest, RecallResponse, RecallStrategy, MemoryMatch


@flow(
    name="entity-recall",
    description="Flow for entity-based memory recall that retrieves information about specific entities",
    is_infrastructure=True
)
class EntityRecallFlow(BaseRecallFlow):
    """Flow for entity-based memory recall"""
    
    def __init__(self):
        """Initialize the entity recall flow."""
        super().__init__()
    
    async def validate_entity(self, request: RecallRequest) -> RecallRequest:
        """Validate entity ID and existence"""
        if not request.entity_id:
            raise ExecutionError(
                message="Entity ID is required for entity recall",
                context=ErrorContext.create(
                    flow_name="entity-recall",
                    error_type="ValidationError",
                    error_location="validate_entity",
                    component="EntityRecallFlow",
                    operation="validate_entity_id"
                )
            )
            
        # TODO: Implement proper entity validation through provider registry
        # entity = await memory_provider.get_entity(request.entity_id)
        # if not entity:
        #     raise ExecutionError(f"Entity {request.entity_id} not found in knowledge memory")
            
        return request
    
    async def recall_entity_knowledge(self, request: RecallRequest) -> RecallResponse:
        """Recall knowledge about the entity"""
        # TODO: Implement proper entity recall through provider registry
        # For now, return empty results as placeholder
        matches = []
        
        return RecallResponse(
            matches=matches,
            strategy_used=RecallStrategy.ENTITY,
            total_matches=len(matches),
            query_analysis={"entity_id": request.entity_id}
        )
    
    @pipeline(input_model=RecallRequest, output_model=RecallResponse)
    async def run_pipeline(self, input_data: RecallRequest) -> RecallResponse:
        """Run the entity recall pipeline.
        
        Args:
            input_data: The recall request
            
        Returns:
            Recall response with entity knowledge
        """
        # Execute validation stages
        validated_request = await self.validate_request(input_data)
        entity_validated_request = await self.validate_entity(validated_request)
        
        # Execute entity recall
        result = await self.recall_entity_knowledge(entity_validated_request)
        
        return result