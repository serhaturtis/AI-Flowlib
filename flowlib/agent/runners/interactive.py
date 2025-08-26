"""
Interactive command-line runner for agents.
"""

import logging
import asyncio
from typing import TYPE_CHECKING, Optional

# Avoid circular import, only type hint BaseAgent
if TYPE_CHECKING:
    from ..core.agent import BaseAgent
    from flowlib.agent.models.state import AgentState # For type hint in run_autonomous

# Import exceptions
from ..core.errors import NotInitializedError, ExecutionError

logger = logging.getLogger(__name__)

# Sentinel value to signal the worker task to stop
_SENTINEL = object()

async def _agent_worker(
    agent: 'BaseAgent',
    input_queue: asyncio.Queue,
    output_queue: asyncio.Queue
):
    """Background task that processes messages from the input queue."""
    logger.info(f"Agent worker started for {agent.name}.")
    max_iterations = 1000  # Safety limit to prevent infinite loops
    iteration_count = 0
    
    while iteration_count < max_iterations:
        input_item = None
        try:
            # Wait for an item from the input queue with timeout
            try:
                input_item = await asyncio.wait_for(input_queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Worker timeout waiting for input, continuing...")
                iteration_count += 1
                continue

            # Check for sentinel to stop the worker
            if input_item is _SENTINEL:
                logger.info("Sentinel received, agent worker stopping.")
                input_queue.task_done()  # Mark sentinel as done
                break

            # Process the input using the agent's core logic with timeout
            result = None
            if isinstance(input_item, str):
                try:
                    # Use the refactored internal method if available
                    if hasattr(agent, '_handle_single_input'):
                        result = await asyncio.wait_for(
                            agent._handle_single_input(input_item), 
                            timeout=60.0
                        )
                    elif hasattr(agent, 'process_message'): # Fallback to public method
                        result = await asyncio.wait_for(
                            agent.process_message(input_item), 
                            timeout=60.0
                        )
                    else:
                        logger.error("Agent has no suitable method (_handle_single_input or process_message) to handle input.")
                        result = f"AGENT_ERROR: No suitable processing method available"
                except asyncio.TimeoutError:
                    logger.error(f"Agent processing timed out for input: {input_item[:50]}...")
                    result = f"TIMEOUT_ERROR: Agent processing took too long"
            else:
                logger.warning(f"Received non-string item in input queue: {type(input_item)}")
                result = f"TYPE_ERROR: Expected string, got {type(input_item)}"

            # Put the result onto the output queue (non-blocking)
            try:
                await asyncio.wait_for(output_queue.put(result), timeout=5.0)
            except asyncio.TimeoutError:
                logger.error("Failed to put result on output queue - queue may be full")
            
            # Mark the processed task as done (important for queue management)
            input_queue.task_done()
            iteration_count += 1

        except asyncio.CancelledError:
            logger.info("Agent worker task cancelled.")
            # Only call task_done if we actually got an item
            if input_item is not None:
                input_queue.task_done()
            break
        except Exception as e:
            logger.error(f"Error in agent worker: {e}", exc_info=True)
            # Put an error indicator onto the output queue
            try:
                await asyncio.wait_for(
                    output_queue.put(f"WORKER_ERROR: {e}"), 
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.error("Failed to put error on output queue")
            
            # Mark the task as done even if there was an error
            if input_item is not None:
                input_queue.task_done()
            iteration_count += 1

    if iteration_count >= max_iterations:
        logger.error(f"Worker exceeded maximum iterations ({max_iterations}), stopping")
    
    logger.info("Agent worker finished.")

async def run_interactive_session(agent: 'BaseAgent'):
    """Runs a standard interactive command-line session for an agent.

    Uses input/output queues to decouple I/O from agent processing.
    Handles user input, puts it on a queue, gets results from another queue,
    and manages agent saving/shutdown on exit.

    Args:
        agent: An initialized BaseAgent instance.
    """
    if not agent or not agent.initialized:
        logger.error("Agent must be initialized before running interactive session.")
        return

    task_id = agent._state_manager.current_state.task_id
    print("\n=== Interactive Agent Session ===")
    print(f"Agent: {agent.name}")
    print(f"Persona: {getattr(agent, 'persona', 'N/A')}") # Display persona if available
    print(f"Session ID: {task_id}")
    print("Type 'exit' or 'quit' to end the session.")

    # Display recent history if available (modify as needed)
    if agent._state_manager.current_state.system_messages:
        print("\n=== Recent History ===")
        max_history = min(3, len(agent._state_manager.current_state.user_messages))
        if max_history > 0:
             for i in range(max_history):
                 user_idx = -(i + 1)
                 sys_idx = -(i + 1)
                 if abs(user_idx) <= len(agent._state_manager.current_state.user_messages):
                     print(f"\nYou: {agent._state_manager.current_state.user_messages[user_idx]}")
                 if abs(sys_idx) <= len(agent._state_manager.current_state.system_messages):
                     print(f"Assistant: {agent._state_manager.current_state.system_messages[sys_idx]}")
        print("\n======================")

    # Main conversation loop - Moved from dual_path_main
    input_queue = asyncio.Queue(maxsize=100)  # Limit queue size to prevent memory issues
    output_queue = asyncio.Queue(maxsize=100)

    # Start the agent worker task in the background
    worker_task = asyncio.create_task(
        _agent_worker(agent, input_queue, output_queue),
        name=f"agent_worker_{agent.name}"
    )

    max_interactions = 10000  # Safety limit for testing
    interaction_count = 0
    
    while interaction_count < max_interactions:
        try:
            user_message = await asyncio.to_thread(input, "\nYou: ")

            if user_message.lower().strip() in ['exit', 'quit', 'bye']:
                logger.info("Exit command received. Shutting down agent.")
                # Signal the worker to stop
                await input_queue.put(_SENTINEL)
                break

            # Put the user message onto the input queue
            await input_queue.put(user_message)

            # Wait for the result from the output queue with timeout
            try:
                result = await asyncio.wait_for(output_queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for agent response")
                result = None
                print(f"\nAssistant: [Timeout] I'm taking too long to respond. Please try again.")

            # Display response (prefer result, fallback to state)
            response_displayed = False
            if result and hasattr(result, 'status') and result.status == "SUCCESS" and hasattr(result.data, "response"):
                print(f"\nAssistant: {result.data.response}")
                response_displayed = True
            elif agent._state_manager.current_state and agent._state_manager.current_state.system_messages:
                # Check if a new system message was added corresponding to this turn
                # This logic might need refinement depending on exact state updates
                print(f"\nAssistant: {agent._state_manager.current_state.system_messages[-1]}")
                response_displayed = True
            
            if not response_displayed:
                 logger.warning("No response generated or found in state for the last turn.")
            
            interaction_count += 1

        except EOFError:
            logger.info("EOF received. Shutting down agent.")
            await input_queue.put(_SENTINEL) # Signal worker to stop
            break # Exit loop on EOF (e.g., piped input ends)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down agent.")
            await input_queue.put(_SENTINEL) # Signal worker to stop
            # Worker cancellation might be handled automatically on main loop exit
            # but sending sentinel is cleaner.
            break # Exit loop on Ctrl+C
        except Exception as e:
            logger.error(f"Error during interactive loop: {e}", exc_info=True)
            print(f"\nAssistant: I encountered an error processing that request: {e}")
            # Decide whether to continue or break on error (currently continues)
            # break
            interaction_count += 1

    if interaction_count >= max_interactions:
        logger.warning(f"Reached maximum interactions limit ({max_interactions}), ending session")
        await input_queue.put(_SENTINEL)

    # Wait for the worker task to finish processing remaining items + sentinel
    try:
        # Wait for queue to be empty with timeout to prevent infinite hang
        await asyncio.wait_for(input_queue.join(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.warning("Input queue cleanup timed out")
    
    # Ensure worker task is properly cleaned up
    if not worker_task.done():
        worker_task.cancel()
        try:
            await asyncio.wait_for(worker_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            logger.warning("Worker task cleanup timed out or was cancelled")
    
    # Drain any remaining items from output queue to prevent memory leaks
    try:
        while not output_queue.empty():
            try:
                output_queue.get_nowait()
                output_queue.task_done()
            except asyncio.QueueEmpty:
                break
    except Exception as e:
        logger.warning(f"Error draining output queue: {e}")

    # Shutdown the agent gracefully
    try:
        logger.info(f"Saving final state for task {agent._state_manager.current_state.task_id}...")
        await agent.save_state()
        logger.info("Shutting down agent...")
        await agent.shutdown()
        logger.info("Agent shutdown complete.")
    except Exception as e:
        logger.error(f"Error during agent save/shutdown: {e}", exc_info=True)

    print("\nSession ended.") 