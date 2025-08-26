"""
Agent learning management component.

This module handles agent learning and intelligence operations
that were previously in BaseAgent.
"""

import logging
from typing import Any, Dict, List, Optional

from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import NotInitializedError
from flowlib.agent.models.config import AgentConfig
from flowlib.agent.components.intelligence.knowledge import LearningResult, Entity, Relationship
from flowlib.agent.components.intelligence.learning import IntelligentLearningFlow
from flowlib.agent.components.intelligence.models import LearningWorthinessEvaluation

logger = logging.getLogger(__name__)


class AgentLearningManager(AgentComponent):
    """Handles agent learning operations.
    
    This component is responsible for:
    - Learning capability initialization
    - Entity extraction and relationship learning
    - Knowledge integration and concept formation
    - Learning from conversations
    """
    
    def __init__(self, name: str = "learning_manager"):
        """Initialize the learning manager.
        
        Args:
            name: Component name
        """
        super().__init__(name)
        self._learning_flow: Optional[IntelligentLearningFlow] = None
    
    async def _initialize_impl(self) -> None:
        """Initialize the learning manager."""
        logger.info("Learning manager initialized")
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the learning manager."""
        if self._learning_flow:
            await self._learning_flow.shutdown()
        logger.info("Learning manager shutdown")
    
    async def evaluate_learning_worthiness(self, 
                                         content: str, 
                                         context: Optional[str] = None) -> LearningWorthinessEvaluation:
        """Evaluate if content is worth learning from.
        
        Args:
            content: Content to evaluate
            context: Optional context for evaluation
            
        Returns:
            Learning worthiness evaluation result
        """
        from flowlib.resources.registry.registry import resource_registry
        
        try:
            # Get the learning worthiness prompt resource
            prompt_resource = resource_registry.get("learning-worthiness-evaluation")
            if not prompt_resource:
                logger.warning("Learning worthiness prompt not found, defaulting to worth learning")
                return LearningWorthinessEvaluation(
                    worth_learning=True,
                    reasoning="Prompt resource not available, defaulting to learning",
                    confidence=0.5
                )
            
            # Get LLM provider using config-driven approach (same as learning flow)
            from flowlib.providers.core.registry import provider_registry
            
            llm = await provider_registry.get_by_config("default-llm")
            if not llm:
                logger.warning("No LLM provider available for learning worthiness evaluation")
                return LearningWorthinessEvaluation(
                    worth_learning=True,
                    reasoning="No LLM provider available, defaulting to learning",
                    confidence=0.5
                )
            
            # Prepare prompt variables
            prompt_vars = {
                "content": content
            }
            if context:
                prompt_vars["context"] = context
            
            # Generate evaluation using structured generation (same as learning flow)
            result = await llm.generate_structured(
                prompt=prompt_resource,
                output_type=LearningWorthinessEvaluation,
                model_name="default-model",
                prompt_variables=prompt_vars
            )
            
            return result
                
        except Exception as e:
            logger.error(f"Error evaluating learning worthiness: {e}")
            return LearningWorthinessEvaluation(
                worth_learning=True,
                reasoning=f"Evaluation error: {str(e)}, defaulting to learning",
                confidence=0.1
            )
    
    async def initialize_learning_capability(self, config: AgentConfig) -> None:
        """Initialize learning capability from configuration.
        
        Args:
            config: Agent configuration
        """
        if not config.enable_learning:
            logger.info("Learning capability disabled in configuration")
            return
        
        try:
            # Initialize the intelligent learning flow
            self._learning_flow = IntelligentLearningFlow()
            # Learning flow no longer uses parent relationships
            if hasattr(self._learning_flow, 'initialize'):
                await self._learning_flow.initialize()
            
            logger.info("Learning capability initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize learning capability: {e}")
    
    async def learn(self, 
                   content: str, 
                   context: Optional[str] = None, 
                   focus_areas: Optional[List[str]] = None) -> Any:
        """Learn from content using the intelligent learning system.
        
        Args:
            content: Content to learn from
            context: Optional context for the learning
            focus_areas: Optional areas to focus learning on
            
        Returns:
            Learning result
            
        Raises:
            NotInitializedError: If learning is not initialized
        """
        if not self._learning_flow:
            raise NotInitializedError(
                component_name=self._name,
                operation="learn"
            )
        
        try:
            # First evaluate if the content is worth learning from
            worthiness = await self.evaluate_learning_worthiness(content, context)
            
            if not worthiness.worth_learning:
                logger.info(f"Skipping learning - not worth it: {worthiness.reasoning}")
                return None
            
            logger.debug(f"Content worth learning (confidence: {worthiness.confidence}): {worthiness.reasoning}")
            
            # Use the intelligent learning flow  
            from flowlib.agent.components.intelligence.learning import LearningInput
            
            learning_input = LearningInput(
                content=content,
                context=context or "general", 
                focus_areas=focus_areas or []
            )
            
            result = await self._learning_flow.run_pipeline(learning_input)
            
            # Store learned knowledge in memory if available
            memory_manager = self.get_component("memory_manager")
            if memory_manager and result:
                await self._store_learning_result(content, result, context, memory_manager)
            
            return result
        except Exception as e:
            logger.error(f"Error during learning: {e}")
            raise
    
    async def extract_entities(self, 
                              content: str, 
                              context: Optional[str] = None) -> List[Entity]:
        """Extract entities from content.
        
        Args:
            content: Content to extract entities from
            context: Optional context for extraction
            
        Returns:
            List of extracted entities
        """
        if not self._learning_flow:
            logger.warning("Learning flow not initialized, cannot extract entities")
            return []
        
        try:
            # Use the learning flow's entity extraction capability
            result = await self.learn(content, context, focus_areas=["entities"])
            
            # Extract entities from the result
            entities = []
            if hasattr(result, 'entities'):
                entities = result.entities
            elif isinstance(result, dict) and 'entities' in result:
                entities = [Entity(**e) if isinstance(e, dict) else e for e in result['entities']]
            
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    async def learn_relationships(self, 
                                 content: str, 
                                 entity_ids: List[str]) -> List[Relationship]:
        """Learn relationships from content.
        
        Args:
            content: Content to learn relationships from
            entity_ids: List of entity IDs to focus on
            
        Returns:
            List of learned relationships
        """
        if not self._learning_flow:
            logger.warning("Learning flow not initialized, cannot learn relationships")
            return []
        
        try:
            # Use the learning flow's relationship learning capability
            result = await self.learn(content, focus_areas=["relationships"])
            
            # Extract relationships from the result
            relationships = []
            if hasattr(result, 'relationships'):
                relationships = result.relationships
            elif isinstance(result, dict) and 'relationships' in result:
                relationships = [Relationship(**r) if isinstance(r, dict) else r for r in result['relationships']]
            
            return relationships
        except Exception as e:
            logger.error(f"Error learning relationships: {e}")
            return []
    
    async def integrate_knowledge(self, 
                                 content: str, 
                                 entity_ids: List[str]) -> Any:
        """Integrate knowledge from content.
        
        Args:
            content: Content to integrate
            entity_ids: List of entity IDs for integration
            
        Returns:
            Integration result
        """
        if not self._learning_flow:
            logger.warning("Learning flow not initialized, cannot integrate knowledge")
            return None
        
        try:
            # Use the learning flow's knowledge integration capability
            result = await self.learn(content, focus_areas=["integration", "knowledge"])
            return result
        except Exception as e:
            logger.error(f"Error integrating knowledge: {e}")
            return None
    
    async def form_concepts(self, content: str) -> List[Any]:
        """Form concepts from content.
        
        Args:
            content: Content to form concepts from
            
        Returns:
            List of formed concepts
        """
        if not self._learning_flow:
            logger.warning("Learning flow not initialized, cannot form concepts")
            return []
        
        try:
            # Use the learning flow's concept formation capability
            result = await self.learn(content, focus_areas=["concepts", "formation"])
            
            # Extract concepts from the result
            concepts = []
            if hasattr(result, 'concepts'):
                concepts = result.concepts
            elif isinstance(result, dict) and 'concepts' in result:
                concepts = result['concepts']
            
            return concepts or []
        except Exception as e:
            logger.error(f"Error forming concepts: {e}")
            return []
    
    async def learn_from_conversation(self, 
                                    user_message: str, 
                                    assistant_response: str, 
                                    context: Optional[Dict[str, Any]] = None) -> None:
        """Learn from a conversation exchange.
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
            context: Optional conversation context
        """
        try:
            # Combine the conversation for learning
            conversation_content = f"User: {user_message}\nAssistant: {assistant_response}"
            
            # Learn from the conversation
            await self.learn(
                content=conversation_content,
                context="conversation",
                focus_areas=["dialogue", "interaction", "user_intent"]
            )
            
            logger.debug("Learned from conversation exchange")
        except Exception as e:
            logger.error(f"Error learning from conversation: {e}")
    
    async def _store_learning_result(self, 
                                   content: str, 
                                   result: Any, 
                                   context: Optional[str],
                                   memory_manager: Any) -> None:
        """Store learning result in memory.
        
        Args:
            content: Original content that was learned from
            result: Learning result to store
            context: Context of the learning
            memory_manager: Memory manager component
        """
        if not memory_manager:
            return
        
        try:
            # Store the learning result
            learning_key = f"learning_{hash(content)}"
            learning_data = {
                "content": content,
                "result": result.model_dump() if hasattr(result, 'model_dump') else str(result),
                "context": context,
                "timestamp": str(logger.handlers[0].formatter.formatTime(logger.makeRecord(
                    name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
                ))) if logger.handlers else None
            }
            
            await memory_manager.store_memory(
                key=learning_key,
                value=learning_data,
                context="learning",
                metadata={"type": "learning_result", "context": context}
            )
        except Exception as e:
            logger.error(f"Error storing learning result: {e}")
    
    @property
    def learning_enabled(self) -> bool:
        """Check if learning is enabled.
        
        Returns:
            True if learning flow is initialized
        """
        return self._learning_flow is not None