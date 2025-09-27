#!/usr/bin/env python3
"""Run the Flowlib Agent REPL - an interactive development environment."""

import argparse
import asyncio
import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from typing import Any, Optional, cast

# Add the project to the Python path (accounting for apps/ directory)
apps_dir = os.path.dirname(os.path.abspath(__file__))
flowlib_root = os.path.dirname(apps_dir)  # Go up from apps/ to flowlib/
project_root = os.path.dirname(flowlib_root)  # Go up to AI-Flowlib/
sys.path.insert(0, project_root)

from flowlib.agent.components.memory.component import AgentMemoryConfig  # noqa: E402
from flowlib.agent.components.memory.knowledge import KnowledgeMemoryConfig  # noqa: E402
from flowlib.agent.components.memory.vector import VectorMemoryConfig  # noqa: E402
from flowlib.agent.components.memory.working import WorkingMemoryConfig  # noqa: E402
from flowlib.agent.models.config import AgentConfig, StatePersistenceConfig  # noqa: E402
from flowlib.agent.runners.repl import start_agent_repl  # noqa: E402
from flowlib.core.project import Project  # noqa: E402
from flowlib.resources.registry.registry import resource_registry  # noqa: E402
from flowlib.resources.models.agent_config_resource import AgentConfigResource  # noqa: E402

# Import flows to ensure they get registered with the flow system
# Note: Tool calling flow moved to tools package - removed unused import

logger = logging.getLogger(__name__)


def check_port(port: int) -> bool:
    """Check if a port is available."""
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
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    connection_result = sock.connect_ex(('localhost', 7687))
                    sock.close()
                    if connection_result == 0:
                        print("  ✓ Neo4j is ready")
                        # Give Neo4j a bit more time to fully initialize
                        time.sleep(3)
                        break
                except (urllib.error.URLError, socket.error, OSError, ConnectionRefusedError):
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
                except (urllib.error.URLError, socket.error, OSError, ConnectionRefusedError):
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

        else:
            logger.error(f"Invalid action: {action}. Must be 'start' or 'stop'")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"Docker command failed: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Failed to {action} services: {e}")
        return False


# Configure flowlib logging to reduce noise in REPL
def configure_flowlib_logging(quiet_mode: bool = True) -> None:
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
            'flowlib.agent.core.dual_path_agent',
            'flowlib.agent.engine.base',
            'flowlib.agent.memory',
            'flowlib.agent.planning',
            'flowlib.agent.reflection',
            'flowlib.agent.conversation'
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


async def setup_providers(project_path: Optional[str] = None) -> None:
    """Load project configurations and providers.

    This replaces manual provider setup with the proper configuration system.
    Providers are configured via:
    1. Configuration files in project's .flowlib/configs/
    2. Role assignments in project's .flowlib/roles/assignments.py

    Args:
        project_path: Optional project path. If None, uses ~/.flowlib/
    """
    # Initialize and load project
    project = Project(project_path)
    project.initialize()  # Create dirs, copy templates if needed
    project.load_configurations()  # Load all configs into registries

    if project_path:
        logger.info(f"Provider setup complete - configurations loaded from {project_path}/.flowlib/")
    else:
        logger.info("Provider setup complete - configurations loaded from ~/.flowlib/")


async def setup_model_resources() -> None:
    """Set up model resources for the agent."""
    # The @model decorators automatically register the resources
    # We just need to ensure they're available
    
    try:
        # Check if LLM models are registered
        default_model = resource_registry.get("default-model")
        logger.info(f"LLM model registered - Default: {bool(default_model)}")
    except Exception as e:
        logger.warning(f"LLM model not found: {e}")
    
    try:
        # Check if embedding model is registered  
        embedding_model = resource_registry.get("agent-embedding-model")
        logger.info(f"Embedding model registered: {embedding_model}")
    except Exception as e:
        logger.warning(f"Embedding model not found: {e}")


async def main(agent_config_name: str = "default-agent-config",
               custom_persona: Optional[str] = None,
               project_path: Optional[str] = None) -> int:
    """Run the Flowlib Agent REPL.

    Args:
        agent_config_name: Name of the agent configuration to use (from resource registry)
        custom_persona: Optional custom persona to override the config's persona
        project_path: Optional project path. If None, uses ~/.flowlib/
    """
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

    # Set up providers and model resources
    await setup_providers(project_path)
    await setup_model_resources()

    # Get agent configuration from resource registry
    try:
        from flowlib.config.role_manager import role_manager
        # Try to resolve through role assignment first
        try:
            actual_config_name = role_manager.get_role_assignment(agent_config_name)
            if actual_config_name:
                agent_config_resource = resource_registry.get(actual_config_name)
                logger.info(f"Loaded agent config '{agent_config_name}' -> '{actual_config_name}'")
            else:
                # No role assignment, try direct lookup
                agent_config_resource = resource_registry.get(agent_config_name)
                logger.info(f"Loaded agent config '{agent_config_name}' directly")
        except Exception:
            # Direct lookup failed, try again
            agent_config_resource = resource_registry.get(agent_config_name)
            logger.info(f"Loaded agent config '{agent_config_name}' directly")
    except Exception as e:
        logger.error(f"Could not load agent configuration '{agent_config_name}': {e}")

        # Show available agent configs for debugging
        try:
            available_configs = [name for name in resource_registry.list() if 'agent-config' in name.lower()]
            logger.info(f"Available agent configs: {available_configs}")
        except Exception:
            logger.debug("Could not list available agent configs")

        logger.info("Using minimal default configuration instead")
        # Create a minimal default configuration
        agent_config_resource = None

    # Extract configuration values from the resource
    if agent_config_resource:
        # Cast to AgentConfigResource for proper type checking
        config_resource = cast(AgentConfigResource, agent_config_resource)
        agent_persona = custom_persona or config_resource.persona
        profile_name = config_resource.profile_name
        model_name = config_resource.model_name
        llm_name = config_resource.llm_name
        temperature = config_resource.temperature
        max_iterations = config_resource.max_iterations
        enable_memory = config_resource.enable_memory
        enable_learning = config_resource.enable_learning
    else:
        # Fallback defaults
        agent_persona = custom_persona or "A helpful assistant"
        profile_name = "default-agent-profile"
        model_name = "default-model"
        llm_name = "default-llm"
        temperature = 0.7
        max_iterations = 10
        enable_memory = True
        enable_learning = True

    logger.info(f"Using agent configuration: {agent_config_name}")
    logger.info(f"Agent profile: {profile_name}")

    logger.info(f"Persona: {agent_persona[:100]}..." if len(agent_persona) > 100 else f"Persona: {agent_persona}")
    
    # Configure state persistence for REPL
    state_config = StatePersistenceConfig(
        persistence_type="file",
        base_path="./repl_states",
        auto_save=True,
        auto_load=False,  # Don't auto-load on startup for REPL
        save_frequency="cycle",  # Save after each planning-execution cycle
        max_states=10  # Keep last 10 states per task
    )

    # Configure memory based on agent config
    if enable_memory:
        memory_config = AgentMemoryConfig(
            working_memory=WorkingMemoryConfig(default_ttl_seconds=3600),
            vector_memory=VectorMemoryConfig(
                vector_provider_config="default-vector-db",
                embedding_provider_config="default-embedding"
            ),
            knowledge_memory=KnowledgeMemoryConfig(
                graph_provider_config="default-graph-db"
            ),
            fusion_llm_config=llm_name,
            store_execution_history=True
        )
    else:
        memory_config = AgentMemoryConfig(
            working_memory=WorkingMemoryConfig(default_ttl_seconds=3600),
            vector_memory=VectorMemoryConfig(),
            knowledge_memory=KnowledgeMemoryConfig(),
            fusion_llm_config="default-llm",
            store_execution_history=False
        )

    # Create agent configuration with values from agent config resource
    config = AgentConfig(
        name="FlowlibAgent",
        provider_name=llm_name,
        persona=agent_persona,
        profile_name=profile_name,  # Agent profile for tool access
        task_description="Interactive REPL development assistant",
        model_name=model_name,
        task_decomposer_max_tokens=1024,
        task_decomposer_temperature=0.2,
        temperature=temperature,
        max_iterations=max_iterations,
        enable_memory=enable_memory,
        enable_learning=enable_learning,
        # Memory configuration
        memory=memory_config,
        # State configuration
        state_config=state_config
    )
    
    # Start REPL with queue-based agent
    await start_agent_repl(
        agent_id="repl_agent",
        config=config,
        history_file=".flowlib_repl_history.txt"
    )

    return 0


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run the Flowlib Agent interactive REPL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_repl.py                                        # Use default agent config
  python run_repl.py --agent-config creative-agent         # Use creative agent
  python run_repl.py --project /path/to/project             # Use project-specific configs
  python run_repl.py --project . --agent-config musician   # Use current dir project
  python run_repl.py --verbose                              # Start with verbose logging
  python run_repl.py --debug                                # Start with debug logging
  python run_repl.py --simple                               # Start with minimal configuration
  python run_repl.py --history my.txt                       # Use custom history file
  python run_repl.py --persona "Custom persona override"   # Override agent's persona

The REPL provides an interactive environment for:
  - Exploring and understanding the Flowlib framework
  - Managing development tasks with TODOs
  - Using tools to analyze and modify code
  - Testing flows and providers

Logging modes:
  - Default: Clean output, only REPL messages and errors
  - --verbose: Shows flowlib INFO messages and above
  - --debug: Shows all flowlib DEBUG messages (very verbose)

Note: Configure your models, providers, and agents in ~/.flowlib/ or project's .flowlib/ directory.
Available agent configs depend on your project configuration.
"""
    )
    
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Start with minimal configuration (no pre-loaded TODOs)"
    )
    
    parser.add_argument(
        "--history",
        type=str,
        default=".flowlib_repl_history.txt",
        help="Path to history file (default: .flowlib_repl_history.txt)"
    )
    
    parser.add_argument(
        "--agent-config",
        type=str,
        default="default-agent-config",
        help="Agent configuration to use (default: default-agent-config)"
    )

    parser.add_argument(
        "--persona",
        type=str,
        help="Override the agent config's persona (optional - only use if you need to customize)"
    )

    parser.add_argument(
        "--project",
        type=str,
        help="Project path containing .flowlib/ directory (default: uses ~/.flowlib/)"
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
    def cleanup() -> None:
        manage_docker_services("stop")
    
    atexit.register(cleanup)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig: int, frame: Optional[Any]) -> None:
        print("\nReceived interrupt signal, shutting down...")
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if args.simple:
            # Simple mode - just start the REPL with defaults
            from flowlib.agent.models.config import AgentConfig
            simple_config = AgentConfig(
                name="SimpleAgent",
                persona="a helpful assistant",
                provider_name="default-llm",
                memory=AgentMemoryConfig()
            )
            asyncio.run(start_agent_repl(
                agent_id="simple_agent",
                config=simple_config,
                history_file=args.history
            ))
        else:
            # Full mode with custom configuration
            asyncio.run(main(
                agent_config_name=args.agent_config,
                custom_persona=args.persona,
                project_path=args.project
            ))
    finally:
        # Ensure cleanup happens even on exceptions
        cleanup()