#!/usr/bin/env python3
"""Unified Flowlib Agent CLI - single entry point for all execution modes."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Add project to Python path
apps_dir = os.path.dirname(os.path.abspath(__file__))
flowlib_root = os.path.dirname(apps_dir)
project_root = os.path.dirname(flowlib_root)
sys.path.insert(0, project_root)

from flowlib.agent.execution.strategy import ExecutionMode  # noqa: E402
from flowlib.agent.launcher import AgentLauncher  # noqa: E402
from flowlib.config.required_alias_validator import RequiredAliasValidator  # noqa: E402
from flowlib.core.project.project import Project  # noqa: E402
from flowlib.core.project.scaffold import (  # noqa: E402
    AgentScaffold,
    ProjectScaffold,
    ToolScaffold,
)
from flowlib.core.project.validator import ProjectValidator  # noqa: E402

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure logging based on verbosity."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Quiet noisy libraries
    if level != "DEBUG":
        logging.getLogger("flowlib.providers").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.ERROR)


async def main() -> int:
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(
        prog="flowlib-agent",
        description="Unified Flowlib Agent Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch REPL for book-writer agent
  flowlib-agent run --project ./projects/book-writer --agent book-writer --mode repl

  # Run autonomous execution
  flowlib-agent run --agent my-agent --mode autonomous --max-cycles 20

  # Start daemon with timer trigger
  flowlib-agent run --agent email-agent --mode daemon --trigger timer:3600

  # Start remote worker
  flowlib-agent run --agent worker --mode remote --queue tasks

Execution Modes:
  repl        - Interactive REPL with Rich UI
  autonomous  - Run N cycles then exit
  daemon      - Continuous execution with triggers
  remote      - Message queue consumer
        """,
    )

    # Global options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Run agent")
    run_parser.add_argument("--project", type=str, help="Project path (default: ~/.flowlib/)")
    run_parser.add_argument("--agent", required=True, help="Agent configuration name")
    run_parser.add_argument(
        "--mode",
        required=True,
        choices=["repl", "autonomous", "daemon", "remote"],
        help="Execution mode",
    )

    # Mode-specific options
    run_parser.add_argument("--max-cycles", type=int, help="[autonomous] Maximum execution cycles")
    run_parser.add_argument(
        "--trigger",
        type=str,
        help="[daemon] Trigger config (e.g., 'timer:3600' or 'email:gmail-default:300')",
    )
    run_parser.add_argument("--queue", type=str, help="[remote] Task queue name")
    run_parser.add_argument("--config", type=str, help="Path to mode-specific config file (YAML)")

    # scaffold command
    scaffold_parser = subparsers.add_parser("scaffold", help="Generate project/tool/agent scaffolding")
    scaffold_sub = scaffold_parser.add_subparsers(dest="scaffold_target", required=True)

    scaffold_project = scaffold_sub.add_parser("project", help="Create a new project skeleton")
    scaffold_project.add_argument("--name", required=True, help="Project name (used for directory)")
    scaffold_project.add_argument("--root", type=str, default="projects", help="Projects root directory")
    scaffold_project.add_argument("--description", type=str, default="Project scaffold.", help="Project README description")
    scaffold_project.add_argument("--agent", action="append", dest="agents", help="Agent names to create (repeatable)")
    scaffold_project.add_argument("--tool-category", action="append", dest="tool_categories", help="Tool categories to pre-create")
    scaffold_project.add_argument(
        "--with-example-tools",
        action="store_true",
        help="Generate example tools for each specified category",
    )

    scaffold_agent = scaffold_sub.add_parser("agent", help="Create an agent config stub")
    scaffold_agent.add_argument("--project", required=True, help="Path to project directory")
    scaffold_agent.add_argument("--name", required=True, help="Agent name (used as config id)")
    scaffold_agent.add_argument("--persona", default="I am a helpful Flowlib agent.", help="Persona string")
    scaffold_agent.add_argument("--category", action="append", dest="categories", help="Allowed tool categories (repeatable)")
    scaffold_agent.add_argument("--description", default="Generated agent config.", help="Class docstring/description")

    scaffold_tool = scaffold_sub.add_parser("tool", help="Create a tool skeleton")
    scaffold_tool.add_argument("--project", required=True, help="Path to project directory")
    scaffold_tool.add_argument("--category", required=True, help="Tool category (e.g., generic, software)")
    scaffold_tool.add_argument("--name", required=True, help="Tool name identifier")
    scaffold_tool.add_argument("--description", default="Generated tool scaffold.", help="Tool description")
    scaffold_tool.add_argument("--with-prompts", action="store_true", help="Include prompts.py stub")
    scaffold_tool.add_argument("--with-flow", action="store_true", help="Include flow.py stub")

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate project structure")
    validate_parser.add_argument("--project", type=str, help="Project path (default: ~/.flowlib/)")

    args = parser.parse_args()

    # Configure logging
    if args.debug:
        configure_logging("DEBUG")
    elif args.verbose:
        configure_logging("INFO")
    else:
        configure_logging("WARNING")

    if args.command == "run":
        return await run_agent(args)
    if args.command == "scaffold":
        run_scaffold(args)
        return 0
    if args.command == "validate":
        run_validation(args)
        return 0

    return 0


async def run_agent(args: argparse.Namespace) -> int:
    """Run agent in specified mode."""

    try:
        # Parse execution mode
        mode = ExecutionMode(args.mode)

        # Build execution config
        execution_config = await build_execution_config(mode, args)

        # Create launcher
        launcher = AgentLauncher(project_path=args.project)
        await launcher.initialize()

        # Launch agent
        await launcher.launch(
            agent_config_name=args.agent, mode=mode, execution_config=execution_config
        )

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Failed to run agent: {e}", exc_info=True)
        return 1


async def build_execution_config(mode: ExecutionMode, args: argparse.Namespace) -> dict[str, Any]:
    """Build mode-specific execution config from CLI args."""

    # If config file provided, load it
    if args.config:
        import yaml

        config_path = Path(args.config)
        if not config_path.exists():
            raise ValueError(f"Config file not found: {args.config}")

        with open(config_path) as f:
            return yaml.safe_load(f)

    # Otherwise build from CLI args
    config: dict[str, Any] = {}

    if mode == ExecutionMode.AUTONOMOUS:
        if args.max_cycles:
            config["max_cycles"] = args.max_cycles

    elif mode == ExecutionMode.DAEMON:
        if args.trigger:
            config["message_sources"] = parse_trigger_arg(args.trigger)

    elif mode == ExecutionMode.REMOTE:
        if args.queue:
            config["task_queue_name"] = args.queue

    return config


def run_scaffold(args: argparse.Namespace) -> None:
    """Handle scaffold CLI commands."""

    if args.scaffold_target == "project":
        scaffold = ProjectScaffold(root_path=Path(args.root))
        project_path = scaffold.create_project(
            name=args.name,
            description=args.description,
            agent_names=args.agents,
            tool_categories=args.tool_categories,
            create_example_tools=args.with_example_tools,
        )
        logger.info(f"Created project scaffold at {project_path}")
        return

    if args.scaffold_target == "agent":
        project_path = Path(args.project).resolve()
        scaffold = AgentScaffold(project_path)
        file_path = scaffold.create_agent(
            name=args.name,
            persona=args.persona,
            allowed_categories=args.categories,
            description=args.description,
        )
        logger.info(f"Created agent scaffold at {file_path}")
        return

    if args.scaffold_target == "tool":
        project_path = Path(args.project).resolve()
        scaffold = ToolScaffold(project_path)
        tool_path = scaffold.create_tool(
            name=args.name,
            category=args.category,
            description=args.description,
            include_prompts=args.with_prompts,
            include_flow=args.with_flow,
        )
        logger.info(f"Created tool scaffold at {tool_path}")
        return

    raise ValueError(f"Unknown scaffold target: {args.scaffold_target}")


def run_validation(args: argparse.Namespace) -> None:
    """Validate that a project adheres to Flowlib structure requirements."""

    project_path_str = args.project if args.project else None
    project_path = Path(project_path_str).resolve() if project_path_str else Path.home() / ".flowlib"

    # Validate project structure
    validator = ProjectValidator()
    structure_result = validator.validate(project_path)

    structure_valid = structure_result.is_valid
    if not structure_valid:
        logger.error("Project structure validation failed:")
        for issue in structure_result.issues:
            logger.error(" - %s: %s", issue.path, issue.message)

    # Validate required aliases
    try:
        # Load project to trigger alias loading
        project = Project(project_path=project_path_str)
        project.load_configurations()

        # Validate required aliases
        alias_result = RequiredAliasValidator.validate_project()

        if alias_result.valid and not alias_result.warnings:
            logger.info("✓ Required alias validation passed")
        elif alias_result.valid and alias_result.warnings:
            logger.warning("Required alias validation passed with warnings:")
            logger.warning(alias_result.get_error_message())
        else:
            logger.error("✗ Required alias validation failed:")
            logger.error(alias_result.get_error_message())
            structure_valid = False

    except Exception as e:
        logger.error(f"Failed to validate required aliases: {e}")
        structure_valid = False

    # Final result
    if structure_valid:
        logger.info("✓ Project validation completed successfully")
        return

    raise SystemExit(1)


def parse_trigger_arg(trigger: str) -> list:
    """Parse trigger CLI argument to message source config.

    Examples:
        timer:3600 -> TimerMessageSourceConfig(interval_seconds=3600)
        email:gmail:300 -> EmailMessageSourceConfig(provider=gmail, interval=300)
    """
    from flowlib.agent.core.message_sources import (
        EmailMessageSourceConfig,
        TimerMessageSourceConfig,
    )

    parts = trigger.split(":")
    trigger_type = parts[0]

    if trigger_type == "timer":
        interval = int(parts[1]) if len(parts) > 1 else 3600
        return [
            TimerMessageSourceConfig(name="cli_timer", interval_seconds=interval, run_on_start=True)
        ]

    elif trigger_type == "email":
        provider = parts[1] if len(parts) > 1 else "default-email"
        interval = int(parts[2]) if len(parts) > 2 else 300
        return [
            EmailMessageSourceConfig(
                name="cli_email",
                email_provider_name=provider,
                check_interval_seconds=interval,
            )
        ]

    else:
        raise ValueError(f"Unknown trigger type: {trigger_type}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
