"""
Thread management for agents.

This module manages agent threads and their message queues.
"""

import asyncio
import threading
import logging
from typing import Dict, Optional
from flowlib.agent.core.base_agent import BaseAgent
from flowlib.agent.core.response_router import ResponseRouter
from flowlib.agent.core.models.messages import AgentMessage, AgentResponse
from flowlib.agent.models.config import AgentConfig

logger = logging.getLogger(__name__)


class AgentThreadPoolManager:
    """Manages agent threads and their queues."""
    
    def __init__(self) -> None:
        """Initialize thread pool manager."""
        self.agents: Dict[str, BaseAgent] = {}
        self.response_routers: Dict[str, ResponseRouter] = {}
        self._lock = threading.Lock()
        
    def create_agent(self, agent_id: str, config: AgentConfig) -> BaseAgent:
        """Create and start an agent in its own thread.
        
        Args:
            agent_id: Unique agent identifier
            config: Agent configuration
            
        Returns:
            Created agent instance
            
        Raises:
            ValueError: If agent already exists
        """
        with self._lock:
            if agent_id in self.agents:
                raise ValueError(f"Agent {agent_id} already exists")
            
            # Create agent with task description from config
            agent = BaseAgent(
                config=config,
                task_description=config.task_description or ""
            )
            
            # Store agent
            self.agents[agent_id] = agent
            
            # Start agent thread
            agent.start()
            
            # Create response router for this agent
            router = ResponseRouter(agent_id, agent.output_queue)
            self.response_routers[agent_id] = router
            
            logger.info(f"Created and started agent {agent_id}")
            return agent
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(agent_id)
    
    def get_response_router(self, agent_id: str) -> Optional[ResponseRouter]:
        """Get response router for agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Response router or None if not found
        """
        return self.response_routers.get(agent_id)
    
    async def send_message(self, agent_id: str, message: AgentMessage) -> str:
        """Send message to agent.
        
        Args:
            agent_id: Target agent ID
            message: Message to send
            
        Returns:
            Message ID
            
        Raises:
            ValueError: If agent not found
        """
        logger.debug(f"[ThreadManager] Looking for agent {agent_id}")
        agent = self.agents.get(agent_id)
        if not agent:
            logger.error(f"[ThreadManager] Agent {agent_id} not found")
            raise ValueError(f"Agent {agent_id} not found")
        
        logger.debug(f"[ThreadManager] Checking if agent {agent_id} is running")
        if not agent.is_running():
            logger.error(f"[ThreadManager] Agent {agent_id} is not running")
            raise RuntimeError(f"Agent {agent_id} is not running")
        
        logger.debug(f"[ThreadManager] Scheduling message {message.message_id} to agent's queue")
        # Send to agent's thread-safe input queue
        agent.input_queue.put(message)
        
        logger.info(f"[ThreadManager] Successfully sent message {message.message_id} to agent {agent_id}")
        return message.message_id
    
    async def wait_for_response(self, agent_id: str, message_id: str, timeout: Optional[float] = None) -> AgentResponse:
        """Wait for response from agent.
        
        Args:
            agent_id: Agent ID
            message_id: Message ID to wait for
            timeout: Timeout in seconds
            
        Returns:
            Agent response
            
        Raises:
            ValueError: If agent not found
            TimeoutError: If response times out
        """
        logger.debug(f"[ThreadManager] Getting router for agent {agent_id}")
        router = self.response_routers.get(agent_id)
        if not router:
            logger.error(f"[ThreadManager] No router found for agent {agent_id}")
            raise ValueError(f"Agent {agent_id} not found")
        
        logger.debug(f"[ThreadManager] Waiting for response to message {message_id}")
        response = await router.wait_for_response(message_id, timeout)
        logger.debug(f"[ThreadManager] Received response for message {message_id}")
        return response
    
    def shutdown_agent(self, agent_id: str) -> None:
        """Shutdown an agent.
        
        Args:
            agent_id: Agent to shutdown
        """
        with self._lock:
            agent = self.agents.get(agent_id)
            if not agent:
                logger.warning(f"Agent {agent_id} not found for shutdown")
                return
            
            # Stop agent
            agent.stop()
            
            # Stop response router
            router = self.response_routers.get(agent_id)
            if router and router.routing_task:
                # Check if we're in an async context
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context, create task to stop router
                    asyncio.create_task(router.stop())
                except RuntimeError:
                    # Not in async context, run synchronously
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(router.stop())
                    loop.close()
            
            # Clean up
            del self.agents[agent_id]
            del self.response_routers[agent_id]
            
            logger.info(f"Shutdown agent {agent_id}")
    
    def shutdown_all(self) -> None:
        """Shutdown all agents."""
        agent_ids = list(self.agents.keys())
        for agent_id in agent_ids:
            self.shutdown_agent(agent_id)
        
        logger.info("Shutdown all agents")