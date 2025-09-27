"""
Response router for agent queue-based communication.

This module handles routing of agent responses back to waiting clients.
"""

import asyncio
import logging
import queue
from typing import Dict, Optional, Union
from flowlib.agent.core.models.messages import AgentResponse

logger = logging.getLogger(__name__)


class ResponseRouter:
    """Routes agent responses to waiting clients."""
    
    def __init__(self, agent_id: str, agent_output_queue: Union[asyncio.Queue[AgentResponse], queue.Queue[AgentResponse]]):
        """Initialize response router.
        
        Args:
            agent_id: ID of the agent this router serves
            agent_output_queue: Queue to read responses from (thread-safe or async)
        """
        self.agent_id = agent_id
        self.agent_output_queue = agent_output_queue
        self.pending_responses: Dict[str, asyncio.Future[AgentResponse]] = {}
        self.routing_task: Optional[asyncio.Task[None]] = None
        self._running = False
        
    async def start(self) -> None:
        """Start routing responses."""
        if self._running:
            raise RuntimeError(f"Response router for agent {self.agent_id} already running")

        self._running = True
        self.routing_task = asyncio.create_task(self._routing_loop())
        logger.info(f"Started response router for agent {self.agent_id}")

    async def stop(self) -> None:
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
        
    async def _routing_loop(self) -> None:
        """Route responses from agent to waiting clients."""
        logger.debug(f"[Router] Starting routing loop for agent {self.agent_id}")
        while self._running:
            try:
                # Wait for response from agent
                logger.debug("[Router] Waiting for response from agent output queue...")
                
                # Handle both thread-safe and async queues
                if isinstance(self.agent_output_queue, queue.Queue):
                    # Thread-safe queue - poll with timeout
                    try:
                        response = self.agent_output_queue.get(timeout=0.1)
                        logger.debug(f"[Router] Got response from output queue for message {response.message_id}")
                    except queue.Empty:
                        await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                        continue
                else:
                    # Async queue
                    response = await self.agent_output_queue.get()
                    logger.debug(f"[Router] Got response from output queue for message {response.message_id}")
                
                # Find waiting future
                future = self.pending_responses.get(response.message_id)
                if future and not future.done():
                    logger.debug(f"[Router] Setting result for waiting future {response.message_id}")
                    future.set_result(response)
                    del self.pending_responses[response.message_id]
                    logger.info(f"[Router] Successfully routed response for message {response.message_id}")
                else:
                    logger.warning(f"[Router] No waiting client for message {response.message_id}")
                    
            except asyncio.CancelledError:
                logger.debug(f"[Router] Routing loop cancelled for agent {self.agent_id}")
                break
            except Exception as e:
                logger.error(f"[Router] Error routing response: {e}")
                
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
        logger.debug("[Router] Checking if router is running...")
        if not self._running:
            logger.error(f"[Router] Router for agent {self.agent_id} is not running")
            raise RuntimeError(f"Response router for agent {self.agent_id} is not running")
            
        # Create future for this message
        logger.debug(f"[Router] Creating future for message {message_id}")
        future: asyncio.Future[AgentResponse] = asyncio.Future()
        self.pending_responses[message_id] = future
        
        try:
            if timeout:
                logger.debug(f"[Router] Waiting for response with timeout {timeout}s")
                return await asyncio.wait_for(future, timeout)
            else:
                logger.debug("[Router] Waiting for response without timeout")
                return await future
        except asyncio.TimeoutError:
            logger.error(f"[Router] Response timeout for message {message_id} after {timeout}s")
            # Clean up pending response
            self.pending_responses.pop(message_id, None)
            raise TimeoutError(f"Response timeout for message {message_id} after {timeout}s")
        except Exception as e:
            logger.error(f"[Router] Error waiting for response: {e}")
            # Clean up on any error
            self.pending_responses.pop(message_id, None)
            raise