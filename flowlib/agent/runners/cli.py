#!/usr/bin/env python3
"""
AI-Flowlib Agent CLI
===================
Command-line interface for running individual agents.
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

# Add the current directory to Python path for development
current_dir = Path(__file__).parent.parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from flowlib.agent.core.orchestrator import AgentOrchestrator
    from flowlib.agent.models.config import AgentConfig
    from flowlib.agent.runners.interactive import run_interactive_session
except ImportError as e:
    print(f"Error importing flowlib modules: {e}")
    print("Make sure flowlib is properly installed.")
    sys.exit(1)


class AgentCLI:
    """Command-line interface for running agents."""
    
    def __init__(self):
        self.home_dir = Path.home()
        self.config_dir = self.home_dir / ".flowlib"
        self.agents_dir = self.config_dir / "agents"
    
    def load_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Load agent configuration from file."""
        config_file = self.agents_dir / f"{agent_name}.json"
        
        if not config_file.exists():
            print(f"Error: Agent '{agent_name}' not found.")
            print(f"Run 'flowlib-config' to create agents.")
            return None
        
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading agent config: {e}")
            return None
    
    def create_agent_from_config(self, config: Dict[str, Any]) -> AgentOrchestrator:
        """Create an AgentOrchestrator instance from configuration."""
        # Build AgentConfig from the saved configuration
        # Validate required configuration fields
        if "persona" not in config or "name" not in config["persona"]:
            raise ValueError("Configuration missing required 'persona.name' field")
        if "mode" not in config or "settings" not in config["mode"] or "max_turns" not in config["mode"]["settings"]:
            raise ValueError("Configuration missing required 'mode.settings.max_turns' field")
            
        agent_config = AgentConfig(
            name=config["name"],
            persona=config["persona"]["name"],
            provider_name=config["provider"]["provider_type"],
            system_prompt=config["persona"]["system_prompt"],
            max_iterations=config["mode"]["settings"]["max_turns"]
        )
        
        return AgentOrchestrator(agent_config)
    
    def print_welcome(self, config: Dict[str, Any]):
        """Print welcome message for the agent."""
        print(f"\n🤖 {config['name']} Agent")
        print(f"Persona: {config['persona']['name']}")
        print(f"Mode: {config['mode']['name']}")
        interface = config["interface"] if "interface" in config else "repl"
        print(f"Interface: {interface.upper()}")
        print(f"Provider: {config['provider']['provider_type']}")
        
        if config['persona']['personality']:
            print(f"Personality: {config['persona']['personality']}")
        
        print("\nType 'exit', 'quit', or press Ctrl+C to end the conversation.")
        print("-" * 60)
    
    async def run_agent(self, agent_name: str, args: argparse.Namespace):
        """Run the specified agent."""
        # Load configuration
        config = self.load_agent_config(agent_name)
        if not config:
            return 1
        
        try:
            # Create agent
            agent = self.create_agent_from_config(config)
            
            # Initialize agent
            await agent.initialize()
            
            # Print welcome message
            if not args.quiet:
                self.print_welcome(config)
            
            # Handle different modes of operation
            if args.message:
                # Single message mode
                response = await agent.process_message(args.message)
                print(response)
                
            elif args.file:
                # File input mode
                with open(args.file, 'r') as f:
                    content = f.read()
                response = await agent.process_message(content)
                print(response)
                
            else:
                # Interactive mode - choose interface based on config
                interface_type = config["interface"] if "interface" in config else "repl"
                
                if interface_type == 'repl':
                    # Use REPL interface
                    try:
                        from flowlib.agent.runners.repl import start_agent_repl
                        await start_agent_repl(agent)
                    except ImportError:
                        print("REPL interface not available, falling back to CLI")
                        await run_interactive_session(agent)
                else:
                    # Use simple CLI interface
                    await run_interactive_session(agent)
            
            # Shutdown agent
            await agent.shutdown()
            return 0
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            return 0
        except Exception as e:
            print(f"Error running agent: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
    
    def list_agents(self):
        """List all available agents."""
        if not self.agents_dir.exists():
            print("No agents configured yet.")
            print("Run 'flowlib-config' to create agents.")
            return
        
        agents = []
        for config_file in self.agents_dir.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                agents.append({
                    'name': config['name'],
                    'persona': config['persona']['name'],
                    'mode': config['mode']['name'],
                    'interface': (config["interface"] if "interface" in config else "repl").upper(),
                    'provider': config['provider']['provider_type']
                })
            except Exception:
                continue
        
        if not agents:
            print("No valid agents found.")
            return
        
        print("Available agents:")
        print("-" * 90)
        print(f"{'Name':<15} {'Persona':<15} {'Mode':<15} {'Interface':<10} {'Provider':<15}")
        print("-" * 90)
        
        for agent in agents:
            print(f"{agent['name']:<15} {agent['persona']:<15} {agent['mode']:<15} {agent['interface']:<10} {agent['provider']:<15}")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="AI-Flowlib Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  {agent_name}                    # Start interactive session
  {agent_name} -m "Hello!"        # Send single message
  {agent_name} -f input.txt       # Process file content
  {agent_name} --list            # List all agents
        """.format(agent_name="<agent_name>")
    )
    
    parser.add_argument(
        "agent",
        nargs="?",
        help="Agent name to run"
    )
    
    parser.add_argument(
        "-m", "--message",
        help="Send a single message and exit"
    )
    
    parser.add_argument(
        "-f", "--file",
        help="Process content from file"
    )
    
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all available agents"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress welcome message"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug information"
    )
    
    return parser


async def async_main(agent_name: Optional[str] = None):
    """Async main function."""
    cli = AgentCLI()
    
    # If called with specific agent name (from executable)
    if agent_name:
        parser = create_parser()
        args = parser.parse_args()
        args.agent = agent_name
        
        if args.list:
            cli.list_agents()
            return 0
        
        return await cli.run_agent(agent_name, args)
    
    # If called directly
    parser = create_parser()
    args = parser.parse_args()
    
    if args.list:
        cli.list_agents()
        return 0
    
    if not args.agent:
        parser.print_help()
        print("\nUse 'flowlib-config' to create and manage agents.")
        return 1
    
    return await cli.run_agent(args.agent, args)


def main(agent_name: Optional[str] = None):
    """Main entry point."""
    try:
        return asyncio.run(async_main(agent_name))
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())