"""Bash tool implementation."""

import asyncio
import os
import subprocess
import time
from pathlib import Path
from typing import cast
from ...models import ToolResult, ToolExecutionContext, ToolStatus, TodoItem
from ...decorators import tool
from .models import BashParameters, BashResult


@tool(name="bash", tool_category="systems", description="Execute shell commands with timeout and output capture")
class BashTool:
    """Tool for executing shell commands."""
    
    def get_name(self) -> str:
        """Get tool name."""
        return "bash"
    
    def get_description(self) -> str:
        """Get tool description."""
        return "Execute shell commands with timeout and output capture"
    
    async def execute(
        self, 
        todo: TodoItem,  # TodoItem with task description
        context: ToolExecutionContext  # Execution context
    ) -> ToolResult:
        """Execute shell command."""
        
        # Generate parameters from todo content
        try:
            params = await self._generate_parameters(todo, context)
        except Exception as e:
            return BashResult(
                status=ToolStatus.ERROR,
                message=f"Failed to generate parameters: {str(e)}"
            )
        
        # Determine working directory
        working_dir = params.working_directory
        if not working_dir:
            working_dir = context.working_directory if context else os.getcwd()
        
        # Resolve working directory path
        if working_dir:
            working_dir = str(Path(working_dir).resolve())
            if not Path(working_dir).exists():
                return BashResult(
                    status=ToolStatus.ERROR,
                    message=f"Working directory does not exist: {working_dir}",
                    command=params.command,
                    working_directory=working_dir
                )
        
        # Set up environment
        env = os.environ.copy()
        if params.env_vars:
            env.update(params.env_vars)
        
        start_time = time.time()
        
        try:
            # Execute command
            process = await asyncio.create_subprocess_shell(
                params.command,
                stdout=subprocess.PIPE if params.capture_output else None,
                stderr=subprocess.PIPE if params.capture_output else None,
                cwd=working_dir,
                env=env
            )
            
            try:
                # Handle timeout=0 as "no timeout"
                timeout_value = None if params.timeout == 0 else params.timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout_value
                )
            except asyncio.TimeoutError:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                
                return BashResult(
                    status=ToolStatus.ERROR,
                    message=f"Command timed out after {params.timeout} seconds",
                    command=params.command,
                    exit_code=process.returncode,
                    execution_time=time.time() - start_time,
                    timed_out=True,
                    working_directory=working_dir
                )
            
            execution_time = time.time() - start_time
            
            # Decode output
            stdout_str = stdout.decode('utf-8', errors='replace') if stdout else ""
            stderr_str = stderr.decode('utf-8', errors='replace') if stderr else ""
            
            # Determine status based on exit code
            if process.returncode == 0:
                status = ToolStatus.SUCCESS
                message = "Command executed successfully"
            else:
                status = ToolStatus.ERROR
                message = f"Command failed with exit code {process.returncode}"
            
            return BashResult(
                status=status,
                message=message,
                command=params.command,
                exit_code=process.returncode,
                stdout=stdout_str,
                stderr=stderr_str,
                execution_time=execution_time,
                timed_out=False,
                working_directory=working_dir
            )
            
        except FileNotFoundError:
            return BashResult(
                status=ToolStatus.ERROR,
                message="Shell not found or command not executable",
                command=params.command,
                execution_time=time.time() - start_time,
                working_directory=working_dir
            )
        
        except PermissionError:
            return BashResult(
                status=ToolStatus.ERROR,
                message="Permission denied executing command",
                command=params.command,
                execution_time=time.time() - start_time,
                working_directory=working_dir
            )
        
        except OSError as e:
            return BashResult(
                status=ToolStatus.ERROR,
                message=f"OS error executing command: {str(e)}",
                command=params.command,
                execution_time=time.time() - start_time,
                working_directory=working_dir
            )
    
    async def _generate_parameters(self, todo: TodoItem, context: ToolExecutionContext) -> BashParameters:
        """Generate BashParameters from todo content using flow."""
        from flowlib.flows.registry.registry import flow_registry
        from .flow import BashParameterGenerationInput, BashParameterGenerationFlow

        # Get the parameter generation flow class
        flow_obj = flow_registry.get("bash-parameter-generation")
        if flow_obj is None:
            raise RuntimeError("Bash parameter generation flow not found in registry")
        flow_instance = cast(BashParameterGenerationFlow, flow_obj)
        
        
        # Extract task content from todo
        task_content = todo.content if hasattr(todo, 'content') else str(todo)
        
        # Create flow input
        flow_input = BashParameterGenerationInput(
            task_content=task_content,
            working_directory=context.working_directory or os.getcwd()
        )
        
        # Execute flow to generate parameters
        result = await flow_instance.run_pipeline(flow_input)
        
        return cast(BashParameters, result.parameters)