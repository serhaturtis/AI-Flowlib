"""Shell command execution flow following Flowlib conventions."""

import asyncio
import logging
import os
import shutil
import subprocess
import time
from typing import List

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry
# Removed ProviderType import - using config-driven provider access
from flowlib.resources.registry.registry import resource_registry
from flowlib.resources.models.constants import ResourceType

from .models import ShellCommandIntentInput, ShellCommandOutput, GeneratedCommand

logger = logging.getLogger(__name__)


@flow(
    name="shell-command",
    description="Execute shell commands on the local system based on high-level intent",
    is_infrastructure=False
)
class ShellCommandFlow:
    """Flow for executing shell commands on the local system.
    
    This flow takes a high-level intent description and generates the appropriate
    shell command to achieve that goal, then executes it safely with proper
    error handling and output capture.
    """
    
    # List of common commands to check for availability
    _COMMON_COMMANDS = [
        "curl", "wget", "jq", "python", "pip", "grep", 
        "sed", "awk", "ls", "cd", "mkdir", "rm", "cat", "echo", "df",
        "git", "docker", "kubectl", "tar", "unzip", "zip", "touch"
    ]

    async def _is_command_available(self, command_name: str) -> bool:
        """Check if a command is available in the system PATH."""
        return shutil.which(command_name) is not None

    def _parse_primary_command(self, command_string: str) -> str:
        """Extract the primary command from a shell command string."""
        if not command_string or not command_string.strip():
            return ""
        
        # Split on common shell operators and take the first command
        parts = command_string.strip().split()
        if parts:
            return parts[0]
        return ""

    async def _get_available_commands(self) -> List[str]:
        """Get list of available commands on the system."""
        available_commands = []
        for cmd in self._COMMON_COMMANDS:
            if await self._is_command_available(cmd):
                available_commands.append(cmd)
        return available_commands

    async def _generate_command(self, input_data: ShellCommandIntentInput) -> GeneratedCommand:
        """Generate shell command using LLM based on intent."""
        # Get LLM provider using model-driven approach
        llm = await provider_registry.get_by_config("default-llm")
        if not llm:
            raise RuntimeError("Could not get LLM provider for 'agent-model-small'")
        
        # Get available commands
        available_commands = await self._get_available_commands()
        available_commands_text = "\n".join([f"- {cmd}" for cmd in available_commands])
        
        # Get prompt template
        prompt = resource_registry.get("shell_command_generation")
        if not prompt:
            raise RuntimeError("Could not find shell_command_generation prompt")
        
        # Prepare prompt variables following conventions
        prompt_vars = {
            "intent": input_data.intent,
            "target_resource": input_data.target_resource or "None",
            "parameters": str(input_data.parameters) if input_data.parameters else "None",
            "output_description": input_data.output_description,
            "available_commands_list": available_commands_text
        }
        
        logger.info(f"Generating shell command for intent: {input_data.intent}")
        logger.debug(f"Available commands: {available_commands}")
        
        # Use generate_structured with default model
        result = await llm.generate_structured(
            prompt=prompt,
            output_type=GeneratedCommand,
            model_name="default-model",
            prompt_variables=prompt_vars
        )
        
        return result

    async def _execute_command(self, command: str, working_dir: str, timeout: int) -> ShellCommandOutput:
        """Execute the shell command safely and capture output."""
        start_time = time.time()
        
        # Validate the command contains only allowed commands
        primary_command = self._parse_primary_command(command)
        available_commands = await self._get_available_commands()
        
        if primary_command and primary_command not in available_commands:
            logger.warning(f"Command '{primary_command}' not in allowed commands list")
            return ShellCommandOutput(
                command=command,
                exit_code=-1,
                stderr=f"Command '{primary_command}' is not available or not allowed",
                execution_time=time.time() - start_time,
                success=False,
                working_dir=working_dir
            )
        
        try:
            logger.info(f"Executing command: {command}")
            
            # Execute command with subprocess
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ShellCommandOutput(
                    command=command,
                    exit_code=-1,
                    stderr=f"Command timed out after {timeout} seconds",
                    execution_time=time.time() - start_time,
                    success=False,
                    working_dir=working_dir
                )
            
            execution_time = time.time() - start_time
            success = process.returncode == 0
            
            result = ShellCommandOutput(
                command=command,
                exit_code=process.returncode,
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                execution_time=execution_time,
                success=success,
                working_dir=working_dir
            )
            
            if success:
                logger.info(f"Command executed successfully in {execution_time:.2f}s")
            else:
                logger.warning(f"Command failed with exit code {process.returncode}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}")
            return ShellCommandOutput(
                command=command,
                exit_code=-1,
                stderr=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time,
                success=False,
                working_dir=working_dir
            )

    @pipeline(input_model=ShellCommandIntentInput, output_model=ShellCommandOutput)
    async def run_pipeline(self, input_data: ShellCommandIntentInput) -> ShellCommandOutput:
        """Main pipeline that generates and executes shell commands based on intent.
        
        Args:
            input_data: Intent-based input describing what should be accomplished
            
        Returns:
            Execution results including command, output, and success status
            
        Raises:
            RuntimeError: If required providers are not available
        """
        working_dir = input_data.working_dir or os.getcwd()
        
        try:
            # Generate command using LLM
            generated = await self._generate_command(input_data)
            
            # Execute the generated command
            result = await self._execute_command(
                command=generated.command,
                working_dir=working_dir,
                timeout=input_data.timeout
            )
            
            logger.info(f"Shell command flow completed: {result.success}")
            return result
            
        except Exception as e:
            logger.error(f"Shell command flow failed: {e}")
            return ShellCommandOutput(
                command="",
                exit_code=-1,
                stderr=f"Flow error: {str(e)}",
                execution_time=0.0,
                success=False,
                working_dir=working_dir
            )