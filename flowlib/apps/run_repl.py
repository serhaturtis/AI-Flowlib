#!/usr/bin/env python3
"""Run the Flowlib Agent REPL - an interactive development environment."""

import asyncio
import sys
import os
import subprocess
import time
import signal
import atexit

# Add the project to the Python path (accounting for apps/ directory)
apps_dir = os.path.dirname(os.path.abspath(__file__))
flowlib_root = os.path.dirname(apps_dir)  # Go up from apps/ to flowlib/
project_root = os.path.dirname(flowlib_root)  # Go up to AI-Flowlib/
sys.path.insert(0, project_root)

from flowlib.agent.runners.repl import start_agent_repl
from flowlib.agent.core import BaseAgent
from flowlib.agent.models.config import AgentConfig
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.decorators.decorators import model_config, embedding_config
from flowlib.resources.models.config_resource import EmbeddingConfigResource
from flowlib.resources.registry.registry import resource_registry
# ResourceType no longer needed - using single-argument registry access
from flowlib.providers import provider_registry
# Providers are now loaded via auto-discovery from ~/.flowlib/configs/
import logging
import urllib.request

# Import flows to ensure they get registered with the flow system
from flowlib.agent.components.conversation import ConversationFlow
from flowlib.agent.components.shell_command import ShellCommandFlow  
from flowlib.agent.components.classification import MessageClassifierFlow

logger = logging.getLogger(__name__)


def check_port(port: int) -> bool:
    """Check if a port is available."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result != 0  # True if port is free


def manage_docker_services(action: str = "start", reset: bool = False) -> bool:
    """Manage Docker services required for REPL.
    
    Args:
        action: Either 'start', 'stop', or 'reset'
        reset: If True with 'start', removes volumes first (fresh start)
        
    Returns:
        True if action succeeded, False otherwise
    """
    docker_compose_file = os.path.join(project_root, "docker-compose.repl.yml")
    
    if not os.path.exists(docker_compose_file):
        logger.warning(f"Docker compose file not found: {docker_compose_file}")
        return False
        
    try:
        if action == "start":
            # If reset requested, clean up first
            if reset:
                print("Resetting services (removing old data)...")
                subprocess.run(
                    ["docker", "compose", "-f", docker_compose_file, "down", "-v"],
                    capture_output=True,
                    text=True
                )
            
            # Check if ports are already in use (services might be running already)
            neo4j_running = not check_port(7687)
            chroma_running = not check_port(8000)
            
            if neo4j_running and chroma_running:
                print("Services appear to be already running.")
                return True
            elif neo4j_running or chroma_running:
                print("Warning: Some services are already running on required ports:")
                if neo4j_running:
                    print("  - Neo4j is using port 7687")
                if chroma_running:
                    print("  - ChromaDB is using port 8000")
                print("Please stop these services manually or use different ports.")
                return False
            
            print("Starting required services (Neo4j, ChromaDB)...")
            # Try docker compose (new style) first, then docker-compose (old style)
            start_success = False
            try:
                result = subprocess.run(
                    ["docker", "compose", "-f", docker_compose_file, "up", "-d"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    start_success = True
                else:
                    logger.debug(f"docker compose failed: {result.stderr}")
            except FileNotFoundError:
                logger.debug("docker compose command not found, trying docker-compose")
            
            if not start_success:
                try:
                    result = subprocess.run(
                        ["docker-compose", "-f", docker_compose_file, "up", "-d"],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        start_success = True
                    else:
                        logger.error(f"docker-compose failed: {result.stderr}")
                        return False
                except FileNotFoundError:
                    logger.error("Neither 'docker compose' nor 'docker-compose' commands found")
                    return False
            
            if not start_success:
                return False
            
            # Wait for services to be ready
            print("Waiting for services to be ready...")
            max_retries = 60  # Increased to 2 minutes
            
            # Check Neo4j (needs more time to initialize)
            print("  Waiting for Neo4j...")
            for i in range(max_retries):
                try:
                    # Check both HTTP and Bolt ports
                    urllib.request.urlopen("http://localhost:7474", timeout=1)
                    # Also verify Bolt protocol is ready
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(('localhost', 7687))
                    sock.close()
                    if result == 0:
                        print("  ✓ Neo4j is ready")
                        # Give Neo4j a bit more time to fully initialize
                        time.sleep(3)
                        break
                except:
                    pass
                
                if i == max_retries - 1:
                    print("  ⚠ Neo4j failed to start")
                    return False
                time.sleep(2)
            
            # Check ChromaDB (may take longer to initialize)
            print("  Waiting for ChromaDB...")
            for i in range(60):  # Give ChromaDB more time
                try:
                    # Use v2 API for health check
                    response = urllib.request.urlopen("http://localhost:8000/api/v2/version", timeout=2)
                    if response.getcode() == 200:
                        print("  ✓ ChromaDB is ready")
                        break
                except:
                    pass
                
                if i == 59:
                    # Check if container is at least running
                    result = subprocess.run(
                        ["docker", "ps", "-q", "-f", "name=flowlib-chroma"],
                        capture_output=True,
                        text=True
                    )
                    if result.stdout.strip():
                        print("  ⚠ ChromaDB container is running but API not responding")
                        print("    Continuing anyway - it may still be initializing")
                        break
                    else:
                        print("  ⚠ ChromaDB container failed to start")
                        return False
                time.sleep(2)
                    
            print("All services are ready!\n")
            return True
            
        elif action == "stop":
            print("\nStopping services...")
            # Try docker compose (new style) first, then docker-compose (old style)
            stop_success = False
            try:
                result = subprocess.run(
                    ["docker", "compose", "-f", docker_compose_file, "down"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    stop_success = True
            except FileNotFoundError:
                pass
            
            if not stop_success:
                try:
                    result = subprocess.run(
                        ["docker-compose", "-f", docker_compose_file, "down"],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        stop_success = True
                except FileNotFoundError:
                    pass
            
            if stop_success:
                print("Services stopped.")
            else:
                logger.warning("Failed to stop services")
                
            return stop_success
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker command failed: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Failed to {action} services: {e}")
        return False


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
        fusion_provider_name="default-llm",
        fusion_model_name="agent-model-small",
        store_execution_history=True
    )
    
    # Create agent configuration
    config = AgentConfig(
        name="FlowlibAgent",
        provider_name="default-llm",
        persona=agent_persona,
        task_description="Interactive REPL development assistant",
        planner_config=planner_config,
        reflection_config=reflection_config,
        memory_config=memory_config,
        state_config=state_config
    )
    
    # Start REPL with queue-based agent
    await start_agent_repl(
        agent_id="repl_agent",
        config=config,
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
    
    # Start Docker services (with automatic reset on first run if needed)
    if not manage_docker_services("start"):
        print("Initial start failed, attempting fresh start with clean volumes...")
        if not manage_docker_services("start", reset=True):
            print("Warning: Failed to start services even after reset.")
            print("You may need to check Docker installation or ports.")
    
    # Register cleanup handler
    def cleanup():
        manage_docker_services("stop")
    
    atexit.register(cleanup)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal, shutting down...")
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if args.simple:
            # Simple mode - just start the REPL with defaults
            asyncio.run(start_agent_repl(history_file=args.history))
        else:
            # Full mode with custom configuration
            asyncio.run(main(persona=args.persona))
    finally:
        # Ensure cleanup happens even on exceptions
        cleanup()