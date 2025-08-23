#!/usr/bin/env python3
"""Run the Flowlib Agent REPL - an interactive development environment."""

import asyncio
import sys
import os

# Add the project to the Python path (accounting for apps/ directory)
apps_dir = os.path.dirname(os.path.abspath(__file__))
flowlib_root = os.path.dirname(apps_dir)  # Go up from apps/ to flowlib/
project_root = os.path.dirname(flowlib_root)  # Go up to AI-Flowlib/
sys.path.insert(0, project_root)

from flowlib.agent.runners.repl import start_agent_repl
from flowlib.agent.core import Agent
from flowlib.agent.models.config import AgentConfig
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.decorators.decorators import model_config, embedding_config
from flowlib.resources.models.config_resource import EmbeddingConfigResource
from flowlib.resources.registry.registry import resource_registry
# ResourceType no longer needed - using single-argument registry access
from flowlib.providers import provider_registry
# Providers are now loaded via auto-discovery from ~/.flowlib/configs/
import logging

# Configure flowlib logging to reduce noise in REPL
def configure_flowlib_logging(quiet_mode: bool = True):
    """Configure logging levels for clean REPL output.
    
    Args:
        quiet_mode: If True, suppress most flowlib logging output
    """
    if quiet_mode:
        # Set the main flowlib logger to WARNING to suppress most output
        flowlib_logger = logging.getLogger('flowlib')
        flowlib_logger.setLevel(logging.WARNING)
        
        # Set specific noisy loggers to ERROR for even less noise
        very_noisy_loggers = [
            'flowlib.providers.llm.llama_cpp.provider',
            'flowlib.providers.registry', 
            'flowlib.providers.llm.base',
            'flowlib.resources.registry',
            'flowlib.flows.registry',
            'flowlib.flows.base',
            'flowlib.providers.vector.chroma.provider',
            'flowlib.providers.embedding',
            'flowlib.agent.utils.model_config',
            'chromadb',  # Suppress chromadb library noise
            'chromadb.telemetry',
            'chromadb.config'
        ]
        
        for logger_name in very_noisy_loggers:
            noisy_logger = logging.getLogger(logger_name)
            noisy_logger.setLevel(logging.ERROR)
        
        # Set moderately noisy loggers to WARNING  
        somewhat_noisy_loggers = [
            'flowlib.agent.shell_command.flow',
            'flowlib.agent.core.dual_path_agent',
            'flowlib.agent.engine.base',
            'flowlib.agent.memory',
            'flowlib.agent.planning',
            'flowlib.agent.reflection',
            'flowlib.agent.conversation',
            'flowlib.agent.classification'
        ]
        
        for logger_name in somewhat_noisy_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
    else:
        # Normal logging mode - set flowlib to INFO
        flowlib_logger = logging.getLogger('flowlib')
        flowlib_logger.setLevel(logging.INFO)

# Configure logging early - this will be reconfigured based on args
configure_flowlib_logging()

logger = logging.getLogger(__name__)


# Configuration should be loaded from ~/.flowlib
# No hardcoded configurations in the code!


async def setup_providers():
    """Trigger auto-discovery to load providers from ~/.flowlib/configs/
    
    This replaces manual provider setup with the proper configuration system.
    Providers are configured via:
    1. Configuration files in ~/.flowlib/configs/
    2. Role assignments in ~/.flowlib/roles/assignments.py
    """
    from flowlib.resources.auto_discovery import discover_configurations
    
    # Trigger auto-discovery to load all configurations and role assignments
    discover_configurations()
    
    logger.info("Provider setup complete - configurations loaded from ~/.flowlib/")


async def setup_model_resources():
    """Set up model resources for the agent."""
    # The @model decorators automatically register the resources
    # We just need to ensure they're available
    
    try:
        # Check if LLM models are registered
        small_model = resource_registry.get("agent-model-small")
        large_model = resource_registry.get("agent-model-large")
        logger.info(f"LLM models registered - Small: {bool(small_model)}, Large: {bool(large_model)}")
    except Exception as e:
        logger.warning(f"LLM model not found: {e}")
    
    try:
        # Check if embedding model is registered  
        embedding_model = resource_registry.get("agent-embedding-model")
        logger.info(f"Embedding model registered: {embedding_model}")
    except Exception as e:
        logger.warning(f"Embedding model not found: {e}")


async def main(persona=None):
    """Run the Flowlib Agent REPL."""
    print("🤖 Flowlib Agent - Interactive REPL")
    print("=" * 60)
    print()
    print("Welcome to the Flowlib Agent interactive development environment!")
    print()
    print("🎯 Available Commands:")
    print("  /help           - Show all available commands")
    print("  /tool list      - List all available tools")
    print("  /todo list      - Show current TODO list")
    print("  /memory         - Show conversation memory")
    print("  /flows          - List available flows")
    print()
    print("🔧 Tool Examples:")
    print("  @read file_path=README.md")
    print("  @grep pattern='class.*:' path='.' include='*.py'")
    print("  @bash command='ls -la'")
    print("  #todo Analyze the codebase structure")
    print()
    print("💡 Flow Execution:")
    print("  !flow-name {\"param\": \"value\"}")
    print()
    print("Type 'exit' or 'quit' to leave the REPL.")
    print("=" * 60)
    
    # Set up providers manually and model resources
    await setup_providers()
    await setup_model_resources()
    
    # Auto-discovery will load role assignments that map model names to providers
    # No need to manually ensure models exist - the system will fail cleanly if they don't
    
    # Determine persona to use
    default_persona = """You are an expert AI assistant for the Flowlib framework.

You understand:
- The flow-based architecture with @flow, @stage, and @pipeline decorators
- The provider system for LLM, database, cache, and other services
- The resource registry for prompts, models, and configurations
- The agent system with memory, planning, and reflection capabilities

You can help users:
- Understand and navigate the Flowlib codebase
- Create new flows and integrate providers
- Debug issues and improve existing code
- Work with the agent's memory and planning systems

When users ask for help:
1. Use tools to explore the codebase
2. Break down complex tasks into TODOs
3. Provide clear, actionable guidance
4. Help debug any issues that arise

You communicate clearly and provide structured, helpful responses."""

    # Use custom persona if provided, otherwise use default
    agent_persona = persona if persona else default_persona
    
    # Configure state persistence for REPL
    from flowlib.agent.models.config import StatePersistenceConfig, PlannerConfig, ReflectionConfig, AgentMemoryConfig, VectorMemoryConfig, KnowledgeMemoryConfig, WorkingMemoryConfig
    
    state_config = StatePersistenceConfig(
        persistence_type="file",
        base_path="./repl_states",
        auto_save=True,
        auto_load=False,  # Don't auto-load on startup for REPL
        save_frequency="cycle",  # Save after each planning-execution cycle
        max_states=10  # Keep last 10 states per task
    )
    
    planner_config = PlannerConfig(
        model_name="agent-model-large",
        provider_name="llamacpp"
    )
    
    reflection_config = ReflectionConfig(
        model_name="agent-model-large",
        provider_name="llamacpp"
    )
    
    memory_config = AgentMemoryConfig(
        working_memory=WorkingMemoryConfig(default_ttl_seconds=3600),
        vector_memory=VectorMemoryConfig(
            vector_provider_name="chroma",
            embedding_provider_name="llamacpp_embedding"
        ),
        knowledge_memory=KnowledgeMemoryConfig(
            graph_provider_name="neo4j",
            provider_settings={
                "uri": "bolt://localhost:7687",
                "username": "neo4j", 
                "password": "pleaseChangeThisPassword"
            }
        ),
        fusion_provider_name="llamacpp",
        fusion_model_name="agent-model-small",
        store_execution_history=True
    )
    
    # Create agent configuration
    config = AgentConfig(
        name="FlowlibAgent",
        provider_name="llamacpp",
        persona=agent_persona,
        task_description="Interactive REPL development assistant",
        planner_config=planner_config,
        reflection_config=reflection_config,
        memory_config=memory_config,
        state_config=state_config
    )
    
    agent = Agent(config=config, task_description="Interactive REPL development assistant")
    
    # Initialize agent
    await agent.initialize()
    
    # Example TODOs disabled - users can create their own with '/todo add' or '#todo'
    
    # Start REPL
    await start_agent_repl(
        agent=agent,
        history_file=".flowlib_repl_history.txt"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run the Flowlib Agent interactive REPL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_repl.py                    # Start the interactive REPL (quiet mode)
  python run_repl.py --verbose          # Start with verbose logging
  python run_repl.py --debug            # Start with debug logging (most verbose)
  python run_repl.py --simple           # Start with minimal configuration
  python run_repl.py --history my.txt   # Use custom history file

The REPL provides an interactive environment for:
  - Exploring and understanding the Flowlib framework
  - Managing development tasks with TODOs
  - Using tools to analyze and modify code
  - Testing flows and providers

Logging modes:
  - Default: Clean output, only REPL messages and errors
  - --verbose: Shows flowlib INFO messages and above
  - --debug: Shows all flowlib DEBUG messages (very verbose)

Note: Configure your models and providers in ~/.flowlib directory.
"""
    )
    
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Start with minimal configuration (no pre-loaded TODOs or custom persona)"
    )
    
    parser.add_argument(
        "--history",
        type=str,
        default=".flowlib_repl_history.txt",
        help="Path to history file (default: .flowlib_repl_history.txt)"
    )
    
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        help="Custom persona for the agent (default: uses built-in expert persona)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging from flowlib components"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true", 
        help="Enable debug logging (most verbose)"
    )
    
    args = parser.parse_args()
    
    # Configure logging based on arguments
    if args.debug:
        # Debug mode - show everything
        logging.getLogger('flowlib').setLevel(logging.DEBUG)
        configure_flowlib_logging(quiet_mode=False)
    elif args.verbose:
        # Verbose mode - show INFO and above
        configure_flowlib_logging(quiet_mode=False)
    else:
        # Default quiet mode
        configure_flowlib_logging(quiet_mode=True)
    
    if args.simple:
        # Simple mode - just start the REPL with defaults
        asyncio.run(start_agent_repl(history_file=args.history))
    else:
        # Full mode with custom configuration
        asyncio.run(main(persona=args.persona))