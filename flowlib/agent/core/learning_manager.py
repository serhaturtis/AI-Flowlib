"""
Agent learning management component.

This module handles agent learning and intelligence operations
that were previously in AgentCore.
"""

import logging
from typing import Any, Dict, List, Optional

from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import NotInitializedError
from flowlib.agent.models.config import AgentConfig
from flowlib.agent.components.intelligence.knowledge import LearningResult, Entity, Relationship
from flowlib.agent.components.intelligence.learning import IntelligentLearningFlow

logger = logging.getLogger(__name__)


class AgentLearningManager(AgentComponent):
    """Handles agent learning operations.
    
    This component is responsible for:
    - Learning capability initialization
    - Entity extraction and relationship learning
    - Knowledge integration and concept formation
    - Learning from conversations
    """
    
    def __init__(self, 
                 activity_stream=None,
                 memory_manager=None,
                 name: str = "learning_manager"):
        """Initialize the learning manager.
        
        Args:
            activity_stream: Activity stream for logging
            memory_manager: Memory manager for storing learned knowledge
            name: Component name
        """
        super().__init__(name)
        self._activity_stream = activity_stream
        self._memory_manager = memory_manager
        self._learning_flow: Optional[IntelligentLearningFlow] = None
    
    async def _initialize_impl(self) -> None:
        """Initialize the learning manager."""
        logger.info("Learning manager initialized")
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the learning manager."""
        if self._learning_flow:
            await self._learning_flow.shutdown()
        logger.info("Learning manager shutdown")
    
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
            self._learning_flow = IntelligentLearningFlow(
                activity_stream=self._activity_stream
            )
            self._learning_flow.set_parent(self)
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
            # Use the intelligent learning flow
            learning_input = {
                "content": content,
                "context": context or "general",
                "focus_areas": focus_areas or []
            }
            
            result = await self._learning_flow.run_pipeline(learning_input)
            
            # Store learned knowledge in memory if available
            if self._memory_manager and result:
                await self._store_learning_result(content, result, context)
            
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
                                   context: Optional[str]) -> None:
        """Store learning result in memory.
        
        Args:
            content: Original content that was learned from
            result: Learning result to store
            context: Context of the learning
        """
        if not self._memory_manager:
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
            
            await self._memory_manager.store_memory(
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