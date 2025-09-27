"""Knowledge flows for workflow integration.

These flows provide workflow-specific interfaces to the KnowledgeComponent,
following flowlib's flow patterns and strict validation.
"""

import logging
from typing import cast

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.core.models import StrictBaseModel
from pydantic import Field

from .models import KnowledgeSet, RetrievalRequest
from .component import KnowledgeComponent

logger = logging.getLogger(__name__)


# Flow Input/Output Models

class AgentKnowledgeExtractionInput(StrictBaseModel):
    """Input for agent knowledge extraction flow."""
    
    text: str = Field(..., min_length=1, description="Text to extract knowledge from")
    context: str = Field(..., description="Context information")
    domain_hint: str = Field(default="general", description="Domain hint for extraction")


class AgentKnowledgeExtractionOutput(StrictBaseModel):
    """Output from agent knowledge extraction flow."""
    
    success: bool = Field(..., description="Whether extraction succeeded")
    knowledge: KnowledgeSet = Field(..., description="Extracted knowledge")
    message: str = Field(..., description="Result message")
    processing_time_seconds: float = Field(..., ge=0.0, description="Processing time")


class AgentKnowledgeRetrievalInput(StrictBaseModel):
    """Input for agent knowledge retrieval flow."""
    
    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")
    context_filter: str = Field(default="", description="Context filter")


class AgentKnowledgeRetrievalOutput(StrictBaseModel):
    """Output from agent knowledge retrieval flow."""
    
    knowledge: KnowledgeSet = Field(..., description="Retrieved knowledge")
    query: str = Field(..., description="Original query")
    total_found: int = Field(..., ge=0, description="Total items found")


# Knowledge Flows

@flow(  # type: ignore[arg-type]
    name="agent-knowledge-extraction",
    description="Extract knowledge from text for agent learning workflows",
    is_infrastructure=False
)
class AgentKnowledgeExtractionFlow:
    """Flow for extracting knowledge in agent workflows."""
    
    @pipeline(
        input_model=AgentKnowledgeExtractionInput,
        output_model=AgentKnowledgeExtractionOutput
    )
    async def run_pipeline(self, input_data: AgentKnowledgeExtractionInput) -> AgentKnowledgeExtractionOutput:
        """Extract knowledge using the knowledge component.
        
        Args:
            input_data: Extraction input data
            
        Returns:
            Extraction output with knowledge
        """
        logger.info(f"Starting agent knowledge extraction for {len(input_data.text)} characters")
        
        try:
            # Get knowledge component from registry
            knowledge_component = self._get_knowledge_component()
            
            # Perform knowledge learning
            learning_result = await knowledge_component.learn_from_content(
                content=input_data.text,
                context=input_data.context,
                domain_hint=input_data.domain_hint
            )
            
            return AgentKnowledgeExtractionOutput(
                success=learning_result.success,
                knowledge=learning_result.knowledge,
                message=learning_result.message,
                processing_time_seconds=learning_result.processing_time_seconds
            )
            
        except Exception as e:
            logger.error(f"Agent knowledge extraction failed: {e}")
            return AgentKnowledgeExtractionOutput(
                success=False,
                knowledge=KnowledgeSet(),
                message=f"Extraction failed: {str(e)}",
                processing_time_seconds=0.0
            )
    
    def _get_knowledge_component(self) -> KnowledgeComponent:
        """Get knowledge component from registry."""
        # This would be injected by the flow runner
        # For now, placeholder following flowlib patterns
        from flowlib.agent.core.component_registry import component_registry
        component = component_registry.get("knowledge")
        if not component:
            raise RuntimeError("Knowledge component not available in registry")
        return cast(KnowledgeComponent, component)


@flow(  # type: ignore[arg-type]
    name="agent-knowledge-retrieval",
    description="Retrieve knowledge for agent workflows",
    is_infrastructure=False
)
class AgentKnowledgeRetrievalFlow:
    """Flow for retrieving knowledge in agent workflows."""
    
    @pipeline(
        input_model=AgentKnowledgeRetrievalInput,
        output_model=AgentKnowledgeRetrievalOutput
    )
    async def run_pipeline(self, input_data: AgentKnowledgeRetrievalInput) -> AgentKnowledgeRetrievalOutput:
        """Retrieve knowledge using the knowledge component.
        
        Args:
            input_data: Retrieval input data
            
        Returns:
            Retrieval output with knowledge
        """
        logger.info(f"Starting agent knowledge retrieval for query: '{input_data.query}'")
        
        try:
            # Get knowledge component from registry
            knowledge_component = self._get_knowledge_component()
            
            # Create retrieval request
            retrieval_request = RetrievalRequest(
                query=input_data.query,
                knowledge_types=[],  # All types
                limit=input_data.limit,
                context_filter=input_data.context_filter if input_data.context_filter else None
            )
            
            # Perform knowledge retrieval
            retrieval_result = await knowledge_component.retrieve_knowledge(retrieval_request)
            
            return AgentKnowledgeRetrievalOutput(
                knowledge=retrieval_result.knowledge,
                query=retrieval_result.query,
                total_found=retrieval_result.total_found
            )
            
        except Exception as e:
            logger.error(f"Agent knowledge retrieval failed: {e}")
            return AgentKnowledgeRetrievalOutput(
                knowledge=KnowledgeSet(),
                query=input_data.query,
                total_found=0
            )
    
    def _get_knowledge_component(self) -> KnowledgeComponent:
        """Get knowledge component from registry."""
        # This would be injected by the flow runner
        # For now, placeholder following flowlib patterns
        from flowlib.agent.core.component_registry import component_registry
        component = component_registry.get("knowledge")
        if not component:
            raise RuntimeError("Knowledge component not available in registry")
        return cast(KnowledgeComponent, component)