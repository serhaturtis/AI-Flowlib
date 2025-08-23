"""
Conversational Agent using FlowLib

This script creates and runs a conversational agent that can interact with users.
It uses the agent architecture from FlowLib and integrates with LLM models for natural language response generation.
"""

import asyncio
import logging
import datetime
import os
import sys
import uuid
from typing import Dict, Any, Optional

# Add the project to the Python path (accounting for apps/ directory)
apps_dir = os.path.dirname(os.path.abspath(__file__))
flowlib_root = os.path.dirname(apps_dir)  # Go up from apps/ to flowlib/
project_root = os.path.dirname(flowlib_root)  # Go up to AI-Flowlib/
sys.path.insert(0, project_root)

from flowlib.agent.models.config import (
    AgentConfig,
    StatePersistenceConfig, 
    PlannerConfig, 
    ReflectionConfig, 
    EngineConfig, 
    MemoryConfig
)
from flowlib.agent.core.agent import Agent as AgentCore
from flowlib.agent.persistence.factory import create_state_persister

# Import flow components
from flowlib.agent.components.conversation.flow import ConversationFlow
from flowlib.agent.components.conversation.models import UserInputResponse

from flowlib.resources.decorators import model

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
STATE_DIR = "agent_states"

# ===================== Define LLM Model Configuration =====================
@model("default")
class LLMModelConfig:
    """Configuration for the conversation LLM."""
    path = "/path/to/your/model.gguf"
    model_type = "llama"
    n_ctx = 4096
    n_threads = 4
    n_batch = 512
    use_gpu = True
    n_gpu_layers = -1
    temperature = 0.7
    max_tokens = 1000

# ===================== Create User Input Callback =====================
async def user_input_callback(data: Dict[str, Any]) -> UserInputResponse:
    """Handle user input requests from the agent.
    
    Args:
        data: Formatted request data
        
    Returns:
        User response
    """
    # Extract data from the request
    message = data.get('formatted_message', 'Please provide input:')
    prompt = data.get('prompt', '> ')
    
    # Display the message to the user
    print(f"\n{message}")
    
    # Get input from the user
    user_input = input(f"{prompt} ")
    
    # Create and return the response
    return UserInputResponse(
        input=user_input,
        timestamp=datetime.datetime.now(),
        additional_context={}
    )

# ===================== State Management =====================
async def list_available_conversations() -> Dict[str, Dict[str, str]]:
    """List available saved conversations.
    
    Returns:
        Dictionary of available conversations with task_id as key
    """
    # Create state persister for listing
    persister = create_state_persister(
        persister_type="file", 
        base_path=STATE_DIR
    )
    
    await persister.initialize()
    try:
        states = await persister.list_states()
        return {s["task_id"]: s for s in states}
    finally:
        await persister.shutdown()

async def prompt_for_conversation() -> Optional[str]:
    """Prompt user to select an existing conversation or start a new one.
    
    Returns:
        Selected task_id or None for a new conversation
    """
    # List available conversations
    conversations = await list_available_conversations()
    
    if not conversations:
        print("\nNo saved conversations found. Starting a new conversation.")
        return None
        
    # Display available conversations
    print("\n=== Available Conversations ===")
    print("0. Start a new conversation")
    
    sorted_conversations = sorted(
        conversations.items(), 
        key=lambda x: x[1].get("timestamp", ""), 
        reverse=True
    )
    
    for i, (task_id, metadata) in enumerate(sorted_conversations, 1):
        timestamp = metadata.get("timestamp", "Unknown date")
        description = metadata.get("task_description", "No description")
        print(f"{i}. {description} (Last updated: {timestamp})")
    
    # Get user selection
    while True:
        try:
            choice = input("\nSelect a conversation (0-{}): ".format(len(sorted_conversations)))
            choice_num = int(choice)
            
            if choice_num == 0:
                return None
            elif 1 <= choice_num <= len(sorted_conversations):
                return sorted_conversations[choice_num-1][0]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")

# ===================== Main Agent Setup =====================
async def setup_agent(task_id: Optional[str] = None) -> AgentCore:
    """Set up and configure the agent.
    
    Args:
        task_id: Optional task ID to resume a conversation
        
    Returns:
        Configured agent ready for conversation
    """
    # Ensure state directory exists
    os.makedirs(STATE_DIR, exist_ok=True)
    
    state_config = StatePersistenceConfig(
        persistence_type="file",
        path=STATE_DIR,
        auto_save=True,
        auto_load=bool(task_id)
    )
    
    # Create properly structured configurations for components
    planner_config = PlannerConfig(
        model_name="default",             # Use the registered default model
        provider_name="llamacpp",         # Specify the provider explicitly
        planning_temperature=0.7,
        planning_max_tokens=1000,
        input_generation_temperature=0.7,
        input_generation_max_tokens=1000
    )
    
    reflection_config = ReflectionConfig(
        model_name="default",             # Use the registered default model
        provider_name="llamacpp",         # Specify the provider explicitly
        temperature=0.7,
        max_tokens=1000
    )
    
    engine_config = EngineConfig(
        max_iterations=10,
        stop_on_error=False,
        log_level="DEBUG"
    )
    
    memory_config = MemoryConfig(
        use_vector_memory=True,
        use_working_memory=True,
        store_execution_history=True
    )
    
    # Create a properly structured agent configuration
    config = AgentConfig(
        name="ConversationalAssistant",
        task_description="Provide helpful responses to user questions and engage in conversation",
        planner_config=planner_config,
        reflection_config=reflection_config,
        engine_config=engine_config,
        memory_config=memory_config,
        state_config=state_config,
# Providers are configured via ~/.flowlib/configs/ and role assignments
        # No hardcoded provider_config needed - agents use registry system
    )
    
    # Create the agent
    task_description = "Provide helpful responses to user questions and engage in conversation"
    agent = AgentCore(
        config=config,
        task_description=task_description
    )
    
    # Set task ID in the state if provided
    if task_id:
        agent.state.data["task_id"] = task_id
    else:
        # Generate a new task ID if none provided
        agent.state.data["task_id"] = str(uuid.uuid4())
    
    # Register flows
    conversation_flow = ConversationFlow()
    user_input_flow = UserInputFlow()
    user_input_flow.set_input_callback(user_input_callback)
    
    return agent

# ===================== Main Conversation Loop =====================
async def run_conversation(agent: AgentCore):
    """Run the conversation loop with the agent.
    
    Args:
        agent: The configured agent
    """
    task_id = agent.state.task_id
    
    print("\n=== Conversational Agent ===")
    print(f"Session ID: {task_id}")
    print("Type 'exit' or 'quit' to end the conversation.")
    
    # Initialize the agent
    await agent.initialize()
    
    # Check if we've loaded an existing conversation
    if agent.state.system_messages or agent.state.user_messages:
        print("\n=== Conversation History ===")
        history_length = min(len(agent.state.user_messages), len(agent.state.system_messages))
        
        # Show last few messages for context
        start_idx = max(0, history_length - 3)  # Show last 3 exchanges
        for i in range(start_idx, history_length):
            if i < len(agent.state.user_messages):
                print(f"\nYou: {agent.state.user_messages[i]}")
            if i < len(agent.state.system_messages):
                print(f"Assistant: {agent.state.system_messages[i]}")
        
        print("\n=== Continuing Conversation ===")
    
    # Main conversation loop
    while True:
        # Get user input
        user_message = input("\nYou: ")
        
        # Check for exit commands
        if user_message.lower() in ['exit', 'quit', 'bye']:
            print("\nGoodbye! Have a great day!")
            print(f"Your conversation has been saved. To resume, use session ID: {task_id}")
            break
        
        # Update the agent state with the user message
        agent.state.add_user_message(user_message)
        
        # Execute a single agent cycle to generate a response
        await agent.execute_cycle()
        
        # Get the last system message (agent's response)
        if agent.state.system_messages:
            last_message = agent.state.system_messages[-1]
            print(f"\nAssistant: {last_message}")
        else:
            print("\nAssistant: I'm not sure how to respond to that.")
    
    # Ensure state is saved before shutdown
    await agent.save_state()
    
    # Shutdown the agent
    await agent.shutdown()

# ===================== Main Entry Point =====================
async def main():
    """Main entry point for the conversational agent."""
    try:
        # Trigger auto-discovery to load configurations from ~/.flowlib/
        from flowlib.resources.auto_discovery import discover_configurations
        discover_configurations()
        
        # Ask if user wants to resume a conversation
        task_id = await prompt_for_conversation()
        
        # Set up the agent with or without a task_id
        agent = await setup_agent(task_id)
        
        # Run the conversation loop
        await run_conversation(agent)
        
    except Exception as e:
        logger.error(f"Error running conversational agent: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 