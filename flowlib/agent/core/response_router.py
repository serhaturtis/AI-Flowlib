"""
Response router for agent queue-based communication.

This module handles routing of agent responses back to waiting clients.
"""

import asyncio
import logging
from typing import Dict, Optional
from flowlib.agent.core.models.messages import AgentResponse

logger = logging.getLogger(__name__)


class ResponseRouter:
    """Routes agent responses to waiting clients."""
    
    def __init__(self, agent_id: str, agent_output_queue: asyncio.Queue):
        """Initialize response router.
        
        Args:
            agent_id: ID of the agent this router serves
            agent_output_queue: Queue to read responses from
        """
        self.agent_id = agent_id
        self.agent_output_queue = agent_output_queue
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.routing_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start routing responses."""
        if self._running:
            raise RuntimeError(f"Response router for agent {self.agent_id} already running")
            
        self._running = True
        self.routing_task = asyncio.create_task(self._routing_loop())
        logger.info(f"Started response router for agent {self.agent_id}")
        
    async def stop(self):
        """Stop routing responses."""
        self._running = False
        
        if self.routing_task:
            self.routing_task.cancel()
            try:
                await self.routing_task
            except asyncio.CancelledError:
                pass
            self.routing_task = None
            
        # Cancel all pending responses
        for future in self.pending_responses.values():
            if not future.done():
                future.cancel()
        self.pending_responses.clear()
        
        logger.info(f"Stopped response router for agent {self.agent_id}")
        
    async def _routing_loop(self):
        """Route responses from agent to waiting clients."""
        while self._running:
            try:
                # Wait for response from agent
                response = await self.agent_output_queue.get()
                
                # Find waiting future
                future = self.pending_responses.get(response.message_id)
                if future and not future.done():
                    future.set_result(response)
                    del self.pending_responses[response.message_id]
                    logger.debug(f"Routed response for message {response.message_id}")
                else:
                    logger.warning(f"No waiting client for message {response.message_id}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error routing response: {e}")
                
    async def wait_for_response(self, message_id: str, timeout: Optional[float] = None) -> AgentResponse:
        """Wait for response for specific message ID.
        
        Args:
            message_id: Message ID to wait for
            timeout: Timeout in seconds (None for no timeout)
            
        Returns:
            Agent response
            
        Raises:
            TimeoutError: If response times out
            RuntimeError: If router is not running
        """
        if not self._running:
            raise RuntimeError(f"Response router for agent {self.agent_id} is not running")
            
        # Create future for this message
        future = asyncio.Future()
        self.pending_responses[message_id] = future
        
        try:
            if timeout:
                return await asyncio.wait_for(future, timeout)
            else:
                return await future
        except asyncio.TimeoutError:
            # Clean up pending response
            self.pending_responses.pop(message_id, None)
            raise TimeoutError(f"Response timeout for message {message_id} after {timeout}s")
        except Exception:
            # Clean up on any error
            self.pending_responses.pop(message_id, None)
            raise