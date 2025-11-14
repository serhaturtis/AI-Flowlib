# Creating Custom Agents: Complete Guide

This guide covers everything you need to know about creating custom agents in flowlib.

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Step-by-Step Creation](#step-by-step-creation)
4. [Configuration Files](#configuration-files)
5. [Custom Tools](#custom-tools)
6. [Custom Flows](#custom-flows)
7. [Running Agents](#running-agents)
8. [Examples](#examples)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### What is a Flowlib Agent?

A flowlib agent is an autonomous AI system that:
- Has a defined **persona** and capabilities
- Uses **tools** to interact with the environment
- Can **plan**, **execute**, and **reflect** on tasks
- Operates in different **modes** (REPL, autonomous, daemon)

### Agent Types

**Simple Agents** (e.g., book-writer, systems-engineer, english-teacher):
- Standard REPL or autonomous mode
- Uses generic or domain-specific tools
- No custom runtime behavior

**Complex Agents** (e.g., field-engineer):
- Custom daemon runners
- Background loops (monitoring, email checking)
- State management
- Health checks and rate limiting

---

## Project Structure

### Standard Project Layout

```
projects/
└── your-agent/
    ├── agents/                      # Agent configurations
    │   ├── __init__.py
    │   └── your_agent.py           # @agent_config decorator
    │
    ├── configs/                     # Provider and model configs
    │   ├── __init__.py
    │   ├── llm_provider.py         # @llm_config (if needed)
    │   ├── model_config.py         # @llm_config (if needed)
    │   ├── embedding.py            # @embedding_config (if needed)
    │   ├── vector_db.py            # @vector_db_config (if needed)
    │   └── graph_db.py             # @graph_db_config (if needed)
    │
    ├── roles/                       # Role assignments
    │   ├── __init__.py
    │   └── assignments.py          # role_manager.assign_role() calls
    │
    ├── tools/                       # Custom tools (optional)
    │   ├── __init__.py
    │   └── your_tool/
    │       ├── __init__.py
    │       ├── tool.py             # @tool decorator
    │       ├── models.py           # Parameters and Result models
    │       ├── flow.py             # Business logic (optional)
    │       └── prompts.py          # LLM prompts (optional)
    │
    ├── flows/                       # Custom flows (optional)
    │   ├── __init__.py
    │   └── your_flow.py            # @flow decorator
    │
    ├── inputs/                      # Input files (optional)
    ├── outputs/                     # Output files (optional)
    ├── __init__.py
    └── README.md
```

### Required Directories

At minimum, you need:
- `agents/` - Agent configuration
- `configs/` - Provider/model configs
- `roles/` - Provider/model role assignments (aliases)

### Optional Directories

- `tools/` - Custom tools specific to your agent
- `flows/` - Custom flows for complex operations
- `inputs/`, `outputs/` - For file-based operations

---

## Step-by-Step Creation

### Step 1: Create Project Directory

```bash
cd projects/
mkdir -p my-agent/{agents,configs,roles,tools,flows}

# Create __init__.py files
touch my-agent/__init__.py
touch my-agent/agents/__init__.py
touch my-agent/configs/__init__.py
touch my-agent/roles/__init__.py
touch my-agent/tools/__init__.py
touch my-agent/flows/__init__.py
```

### Step 2: Define Agent Configuration

Create `agents/my_agent.py`:

```python
"""My Agent configuration."""

from flowlib.resources.decorators.decorators import agent_config


@agent_config("my-agent")
class MyAgentConfig:
    """My custom agent for [purpose]."""

    persona = (
        "I am [role description]. "
        "I specialize in [capabilities]. "
        "I [behavioral traits]. "
    )

    allowed_tool_categories = ["generic", "software"]
    model_name = "default-model"
    llm_name = "default-llm"

    temperature = 0.7          # Creativity vs consistency
    max_iterations = 10        # Max task execution cycles
    enable_learning = False    # Memory/learning capabilities
    verbose = False            # Debug logging
```

**Key Fields:**
- `persona`: Defines agent's identity, capabilities, and behavior
- `allowed_tool_categories`: Tool categories the agent is allowed to use
- `model_name`: LLM model configuration name
- `llm_name`: LLM provider configuration name
- `temperature`: 0.0-1.0 (lower = deterministic, higher = creative)
- `max_iterations`: Safety limit for task execution loops

**Common Tool Categories:**
- `generic`: read, write, edit, bash, and conversation tools (almost always required)
- `software`: coding tools (planning, editing, analysis)
- `systems`: system administration and infrastructure tools
- `data`: analytics, visualization, and processing tools
- Any custom categories defined by your project (e.g., `music`, `x_api`)

### Step 3: Configure Providers and Models

#### Option A: Use Existing Configs

Create `roles/assignments.py`:

```python
"""Role assignments mapping semantic names to configs."""

import logging

from flowlib.config.role_manager import role_manager

logger = logging.getLogger(__name__)

try:
    # Use existing global configs
    role_manager.assign_role("default-llm", "example-llamacpp-provider")
    role_manager.assign_role("default-model", "example-model")
    role_manager.assign_role("default-embedding", "example-llamacpp-embedding-provider")
    role_manager.assign_role("default-embedding-model", "example-embedding-model")
    role_manager.assign_role("default-vector-db", "example-vector-db-provider")
    role_manager.assign_role("default-graph-db", "example-graph-db-provider")

    logger.info("Role assignments completed successfully")

except Exception as e:
    logger.warning(f"Failed to create some role assignments: {e}")
```

#### Option B: Create Custom Configs

**LLM Provider** (`configs/llm_provider.py`):

```python
"""Custom LLM provider configuration."""

from flowlib.resources.decorators.decorators import llm_config


@llm_config("my-agent-llm-provider")
class MyAgentLLMProviderConfig:
    """LlamaCpp provider for my agent."""

    provider_type = "llamacpp"
    settings = {
        "max_concurrent_models": 2,
        "timeout": 120,
        "max_retries": 3,
        "verbose": False,
    }
```

**Model Config** (`configs/model_config.py`):

```python
"""Custom model configuration."""

from flowlib.resources.decorators.decorators import llm_config


@llm_config(
    "my-agent-model",
    provider_type="llamacpp",
    config={
        "path": "/path/to/models/model.gguf",
        "n_ctx": 8192,
        "use_gpu": True,
        "n_gpu_layers": 33,
        "temperature": 0.7,
        "max_tokens": 4096,
    },
)
class MyAgentModelConfig:
    """Model configuration for my agent."""
    pass
```

**Then update role assignments:**

```python
"""Role assignments for my agent."""

import logging

from flowlib.config.role_manager import role_manager

logger = logging.getLogger(__name__)

try:
    role_manager.assign_role("default-llm", "my-agent-llm-provider")
    role_manager.assign_role("default-model", "my-agent-model")
    role_manager.assign_role("default-embedding", "example-llamacpp-embedding-provider")
    role_manager.assign_role("default-embedding-model", "example-embedding-model")
    role_manager.assign_role("default-vector-db", "example-vector-db-provider")
    role_manager.assign_role("default-graph-db", "example-graph-db-provider")

    logger.info("Role assignments completed successfully")

except Exception as e:
    logger.warning(f"Failed to create some role assignments: {e}")
```

### Step 5: Create Run Script

Create `../../run_my_agent.sh` in project root:

```bash
#!/bin/bash
# Script to run my agent REPL

./run_agent run --project ./projects/my-agent --agent my-agent --mode repl
```

Make it executable:

```bash
chmod +x run_my_agent.sh
```

### Step 6: Test Your Agent

```bash
./run_my_agent.sh
```

If successful, you'll see the REPL prompt. Try a simple task:

```
Hello! Please introduce yourself and list your capabilities.
```

---

## Configuration Files

### Agent Config (`@agent_config`)

**Purpose**: Defines the agent's identity, persona, and core settings.

**Location**: `agents/your_agent.py`

**Structure**:
```python
@agent_config("config-name")
class YourAgentConfig:
    persona = "..."              # Identity and behavior
    allowed_tool_categories = ["generic"]  # Tool categories this agent can use
    model_name = "..."           # LLM model config name
    llm_name = "..."             # LLM provider config name
    temperature = 0.7            # Generation randomness
    max_iterations = 10          # Safety limit
    enable_learning = False      # Memory features
    verbose = False              # Debug logging
```

**Persona Guidelines**:
- Start with role/identity: "I am a [role]..."
- List capabilities: "I can [do X, Y, Z]..."
- Define behavior: "I always [behavior]..."
- Set boundaries: "I cannot/will not [limitations]..."
- Be specific and actionable

**Example Personas**:

```python
# Technical agent
persona = (
    "I am an expert systems engineer. "
    "I can analyze system architecture, diagnose problems, and implement solutions. "
    "I always provide detailed technical explanations and consider security implications. "
    "I will not make destructive changes without explicit confirmation."
)

# Creative agent
persona = (
    "I am a creative writing assistant. "
    "I help generate stories, characters, and plot outlines. "
    "I provide constructive feedback and suggestions. "
    "I respect author's vision and never override their creative choices."
)

# Data agent
persona = (
    "I am a data analyst. "
    "I can process datasets, generate visualizations, and provide insights. "
    "I explain statistical methods clearly and validate data quality. "
    "I flag potential biases and limitations in analysis."
)
```

### Tool Categories

Agents now declare tool access directly inside their `@agent_config` class using `allowed_tool_categories`. Common categories:
  - `"generic"`: read/write/edit/bash/conversation (most agents need this)
  - `"software"`: coding and code-analysis tools
  - `"systems"`: infrastructure/operations tools
  - `"data"`: analytics/visualization tools
  - Any domain-specific categories (e.g., `"music"`, `"x_api"`)

There's no separate profile file—your agent config is the single source of truth.

### Role Assignments

**Purpose**: Maps semantic role names to actual configuration names using role_manager.

**Location**: `roles/assignments.py`

**Structure**:
```python
"""Role assignments for your agent."""

import logging

from flowlib.config.role_manager import role_manager

logger = logging.getLogger(__name__)

try:
    role_manager.assign_role("default-llm", "your-llm-provider-config")
    role_manager.assign_role("default-model", "your-model-config")
    role_manager.assign_role("default-embedding", "your-embedding-config")
    role_manager.assign_role("default-embedding-model", "your-embedding-model-config")
    role_manager.assign_role("default-vector-db", "your-vector-db-config")
    role_manager.assign_role("default-graph-db", "your-graph-db-config")

    logger.info("Role assignments completed successfully")

except Exception as e:
    logger.warning(f"Failed to create some role assignments: {e}")
```

**Required Roles**:
- `default-llm`: LLM provider (e.g., llamacpp, googleai)
- `default-model`: Specific model configuration
- `default-embedding`: Embedding provider (for memory)
- `default-embedding-model`: Embedding model configuration
- `default-vector-db`: Vector database (for memory)
- `default-graph-db`: Graph database (for knowledge)

**Optional Roles** (for multimodal agents):
- `default-multimodal-llm`: Multimodal LLM provider
- `default-multimodal-model`: Multimodal model config

### Provider Configs

#### LLM Provider (`@llm_config`)

```python
@llm_config("provider-name")
class ProviderConfig:
    provider_type = "llamacpp"  # or "googleai"
    settings = {
        "max_concurrent_models": 2,
        "timeout": 120,
        "max_retries": 3,
        "verbose": False,
    }
```

**Provider Types**:
- `llamacpp`: Local models via llama.cpp
- `googleai`: Google AI (Gemini)

#### Model Config (`@llm_config`)

```python
@llm_config(
    "model-name",
    provider_type="llamacpp",
    config={
        "path": "/path/to/model.gguf",
        "n_ctx": 8192,
        "use_gpu": True,
        "n_gpu_layers": 33,
        "temperature": 0.7,
        "max_tokens": 4096,
    },
)
class ModelConfig:
    pass
```

**Key Parameters**:
- `path`: Model file path (.gguf for llamacpp)
- `n_ctx`: Context window size (tokens)
- `use_gpu`: Enable GPU acceleration
- `n_gpu_layers`: Layers to offload to GPU
- `temperature`: Generation randomness (0.0-1.0)
- `max_tokens`: Maximum response length

#### Multimodal Provider (`@multimodal_llm_provider_config`)

```python
@multimodal_llm_provider_config("provider-name")
class MultimodalProviderConfig:
    provider_type = "llamacpp_multimodal"
    settings = {
        "max_concurrent_models": 1,
        "timeout": 300,  # Longer for image processing
        "verbose": True,
    }
```

#### Multimodal Model (`@multimodal_llm_config`)

```python
@multimodal_llm_config(
    "model-name",
    provider_type="llamacpp_multimodal",
    config={
        "path": "/path/to/model.gguf",
        "clip_model_path": "/path/to/mmproj.gguf",
        "n_ctx": 8192,
        "use_gpu": True,
        "n_gpu_layers": 48,
        "temperature": 0.3,
        "chat_format": "gemma-3",  # or "llava-1-5", "moondream"
    },
)
class MultimodalModelConfig:
    pass
```

**Chat Formats**:
- `llava-1-5`: LLaVA 1.5 models
- `llava-1-6`: LLaVA 1.6 models
- `gemma-3`: Gemma 3 vision models
- `moondream`: Moondream models
- `llama-3-vision`: Llama 3 vision models

#### Memory Providers

**Embedding Provider**:
```python
@embedding_config(
    "embedding-name",
    provider_type="llamacpp_embedding",
    config={
        "path": "/path/to/embedding-model.gguf",
        "dimensions": 1024,
        "normalize": True,
    },
)
class EmbeddingConfig:
    pass
```

**Vector Database**:
```python
@vector_db_config("vector-db-name")
class VectorDBConfig:
    provider_type = "chroma"  # or "qdrant", "pinecone"
    settings = {
        "host": "localhost",
        "port": 8000,
        "collection_name": "agent_vectors",
        "persist_directory": "~/.flowlib/chroma_data",
    }
```

**Graph Database**:
```python
@graph_db_config("graph-db-name")
class GraphDBConfig:
    provider_type = "neo4j"  # or "memory", "arango"
    settings = {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "password",
    }
```

---

## Custom Tools

### When to Create Custom Tools

Create custom tools when:
- Your agent needs domain-specific functionality
- You need to integrate external APIs/services
- You want to enforce specific workflows
- Generic tools don't meet your requirements

### Tool Structure

```
tools/
└── your_tool/
    ├── __init__.py
    ├── tool.py           # Tool decorator and execute method
    ├── models.py         # Parameters and Result models
    ├── flow.py           # Business logic (optional)
    └── prompts.py        # LLM prompts (optional)
```

### Creating a Simple Tool

**Step 1: Define Models** (`models.py`):

```python
"""Tool parameter and result models."""

from pydantic import Field

from flowlib.agent.components.task.execution.models import (
    ToolParameters,
    ToolResult,
    ToolStatus,
)


class MyToolParameters(ToolParameters):
    """Parameters for my custom tool."""

    input_value: str = Field(..., description="Input value to process")
    option: str = Field(default="default", description="Processing option")


class MyToolResult(ToolResult):
    """Result from my custom tool."""

    output_value: str = Field(..., description="Processed output")
    metadata: dict = Field(default_factory=dict, description="Additional info")
```

**Step 2: Implement Tool** (`tool.py`):

```python
"""My custom tool implementation."""

from flowlib.agent.components.task.execution.decorators import tool
from flowlib.agent.components.task.execution.interfaces import ToolExecutionContext

from .models import MyToolParameters, MyToolResult, ToolStatus


@tool(
    name="my_tool",
    description="Does something useful with input",
    category="custom",
)
async def my_tool(
    params: MyToolParameters,
    context: ToolExecutionContext,
) -> MyToolResult:
    """Execute my custom tool.

    Args:
        params: Tool parameters
        context: Execution context (agent, memory, providers)

    Returns:
        Tool result with output
    """
    try:
        # Your tool logic here
        output = process_input(params.input_value, params.option)

        return MyToolResult(
            status=ToolStatus.SUCCESS,
            message=f"Successfully processed: {params.input_value}",
            output_value=output,
            metadata={"option_used": params.option},
        )

    except Exception as e:
        return MyToolResult(
            status=ToolStatus.ERROR,
            message=f"Tool failed: {str(e)}",
            output_value="",
        )


def process_input(value: str, option: str) -> str:
    """Business logic."""
    # Your implementation
    return f"Processed: {value} with {option}"
```

**Key Points**:
- Use `@tool` decorator with name, description, category
- Function must be `async def`
- Takes `ToolParameters` and `ToolExecutionContext`
- Returns `ToolResult` with status
- Always handle exceptions

### Creating a Tool with LLM Integration

**Step 1: Define Prompts** (`prompts.py`):

```python
"""Prompts for LLM-powered tool."""

from pydantic import Field

from flowlib.flows.base.base import PromptTemplate


class AnalysisPrompt(PromptTemplate):
    """Prompt for analyzing input."""

    template: str = """
Analyze the following input and extract key information:

Input: {input_text}

Provide:
1. Summary (2-3 sentences)
2. Key entities (people, places, organizations)
3. Main topics
4. Sentiment (positive/negative/neutral)

Format your response as JSON with these fields.
"""

    config: dict = Field(
        default={
            "temperature": 0.3,  # Low for consistency
            "max_tokens": 1000,
        }
    )
```

**Step 2: Create Flow** (`flow.py`):

```python
"""Business logic flow for analysis tool."""

from flowlib.flows.base.base import BaseFlow, FlowResult
from flowlib.providers.core.registry import provider_registry

from .models import AnalysisParameters, AnalysisData
from .prompts import AnalysisPrompt


class AnalysisFlow(BaseFlow[AnalysisParameters, AnalysisData]):
    """Flow for analyzing text input."""

    async def run(self, params: AnalysisParameters) -> FlowResult[AnalysisData]:
        """Execute analysis flow."""

        # Get LLM provider
        llm_provider = await provider_registry.get_by_config("default-llm")

        # Format prompt
        prompt = AnalysisPrompt()

        # Generate analysis
        response = await llm_provider.generate_structured(
            prompt=prompt,
            output_type=AnalysisData,
            model_name="default-model",
            prompt_variables={"input_text": params.text},
        )

        return FlowResult(
            success=True,
            data=response,
            message="Analysis complete",
        )
```

**Step 3: Use Flow in Tool** (`tool.py`):

```python
@tool(
    name="analyze_text",
    description="Analyzes text using LLM",
    category="custom",
)
async def analyze_text(
    params: AnalysisParameters,
    context: ToolExecutionContext,
) -> AnalysisResult:
    """Analyze text with LLM."""

    flow = AnalysisFlow()
    result = await flow.run(params)

    if result.success:
        return AnalysisResult(
            status=ToolStatus.SUCCESS,
            message="Analysis complete",
            analysis_data=result.data,
        )
    else:
        return AnalysisResult(
            status=ToolStatus.ERROR,
            message=f"Analysis failed: {result.error}",
        )
```

### Tool Categories

Assign your tool to a category:

```python
@tool(
    name="tool_name",
    category="custom",  # or "software", "systems", "data", "generic"
)
```

Then add the category to your agent config:

```python
allowed_tool_categories = [
    "generic",  # Built-in tools
    "custom",   # Your custom tools
]
```

---

## Custom Flows

### When to Create Flows

Flows are for:
- Multi-step operations
- LLM-powered processing
- Reusable business logic
- Complex workflows

### Flow Structure

```python
"""Custom flow implementation."""

from flowlib.flows.base.base import BaseFlow, FlowResult
from flowlib.flows.decorators.decorators import flow


@flow(
    name="my_flow",
    description="Does complex processing",
)
class MyFlow(BaseFlow[InputType, OutputType]):
    """Custom flow for [purpose]."""

    async def run(self, params: InputType) -> FlowResult[OutputType]:
        """Execute flow logic."""

        # Step 1: Prepare
        prepared = await self._prepare(params)

        # Step 2: Process
        processed = await self._process(prepared)

        # Step 3: Finalize
        result = await self._finalize(processed)

        return FlowResult(
            success=True,
            data=result,
            message="Flow complete",
        )

    async def _prepare(self, params: InputType) -> dict:
        """Preparation step."""
        # Implementation
        pass

    async def _process(self, data: dict) -> dict:
        """Processing step."""
        # Implementation
        pass

    async def _finalize(self, data: dict) -> OutputType:
        """Finalization step."""
        # Implementation
        pass
```

---

## Running Agents

### REPL Mode (Interactive)

**Usage**:
```bash
./run_agent run --project ./projects/my-agent --agent my-agent --mode repl
```

**Characteristics**:
- Interactive command-line interface
- User provides tasks manually
- Agent executes and waits for next input
- Best for: development, testing, interactive work

**REPL Commands**:
- `/help` - Show available commands
- `/exit` - Exit REPL
- `/clear` - Clear conversation history
- `/status` - Show agent status

### Autonomous Mode

**Usage**:
```bash
./run_agent run --project ./projects/my-agent --agent my-agent --mode autonomous --task "Complete task X"
```

**Characteristics**:
- Runs until task complete or max_iterations reached
- No user interaction during execution
- Returns final result
- Best for: batch processing, automation

### Daemon Mode

**Usage**: Requires custom daemon runner (like field-engineer)

```python
# run_daemon.py
class MyDaemon:
    async def run(self):
        while not self.shutdown_requested:
            await self.check_and_process()
            await asyncio.sleep(interval)
```

**Characteristics**:
- Background process
- Monitoring loops
- Event-driven execution
- Best for: monitoring, email bots, scheduled tasks

---

## Examples

### Example 1: Simple Task Agent

**Purpose**: Helps with general tasks

```python
# agents/task_agent.py
@agent_config("task-agent")
class TaskAgentConfig:
    persona = (
        "I am a helpful task assistant. "
        "I can read files, write reports, and execute bash commands. "
        "I always confirm before making changes."
    )
    allowed_tool_categories = ["generic"]
    model_name = "default-model"
    llm_name = "default-llm"
    temperature = 0.7
    max_iterations = 10
    enable_learning = False

# roles/assignments.py
"""Role assignments."""

import logging
from flowlib.config.role_manager import role_manager

logger = logging.getLogger(__name__)

try:
    role_manager.assign_role("default-llm", "example-llamacpp-provider")
    role_manager.assign_role("default-model", "example-model")
    role_manager.assign_role("default-embedding", "example-llamacpp-embedding-provider")
    role_manager.assign_role("default-embedding-model", "example-embedding-model")
    role_manager.assign_role("default-vector-db", "example-vector-db-provider")
    role_manager.assign_role("default-graph-db", "example-graph-db-provider")
    
    logger.info("Role assignments completed successfully")
except Exception as e:
    logger.warning(f"Failed to create some role assignments: {e}")
```

**Run**:
```bash
./run_agent run --project ./projects/task-agent --agent task-agent --mode repl
```

### Example 2: Code Analysis Agent

**Purpose**: Analyzes codebases

```python
# agents/code_agent.py
@agent_config("code-analyzer")
class CodeAnalyzerConfig:
    persona = (
        "I am a code analysis expert. "
        "I can review code quality, identify bugs, and suggest improvements. "
        "I provide detailed explanations and follow best practices."
    )
    allowed_tool_categories = ["generic", "software"]
    model_name = "default-model"
    llm_name = "default-llm"
    temperature = 0.3  # Lower for consistency
    max_iterations = 15
    enable_learning = True  # Remember patterns
```

### Example 3: Multimodal Vision Agent

**Purpose**: Processes images (like english-teacher)

```python
# configs/multimodal_provider.py
@multimodal_llm_provider_config("vision-agent-provider")
class VisionAgentProviderConfig:
    provider_type = "llamacpp_multimodal"
    settings = {
        "max_concurrent_models": 1,
        "timeout": 300,
    }

# configs/multimodal_model.py
@multimodal_llm_config(
    "vision-agent-model",
    provider_type="llamacpp_multimodal",
    config={
        "path": "/path/to/gemma-3-12b.gguf",
        "clip_model_path": "/path/to/mmproj.gguf",
        "chat_format": "gemma-3",
        "n_ctx": 8192,
        "use_gpu": True,
        "n_gpu_layers": 48,
    },
)
class VisionAgentModelConfig:
    pass

# roles/assignments.py
"""Role assignments."""

import logging
from flowlib.config.role_manager import role_manager

logger = logging.getLogger(__name__)

try:
    role_manager.assign_role("default-llm", "example-llamacpp-provider")
    role_manager.assign_role("default-model", "example-model")
    role_manager.assign_role("default-embedding", "example-llamacpp-embedding-provider")
    role_manager.assign_role("default-embedding-model", "example-embedding-model")
    role_manager.assign_role("default-vector-db", "example-vector-db-provider")
    role_manager.assign_role("default-graph-db", "example-graph-db-provider")
    
    logger.info("Role assignments completed successfully")
except Exception as e:
    logger.warning(f"Failed to create some role assignments: {e}")
```

---

## Best Practices

### Code Organization

**✅ DO**:
- All imports at top of file (NO function-level imports)
- One configuration per file
- Clear, descriptive names
- Use decorators for registration
- Follow project structure

**❌ DON'T**:
- Function-level imports (violates flowlib conventions)
- Hardcoded values (use configs)
- Custom launchers for simple agents
- Duplicate configurations
- Mix concerns (one file, one purpose)

### Configuration

**✅ DO**:
- Use role assignments for flexibility
- Config-driven architecture
- Single source of truth
- Fail-fast error handling
- Document required setup

**❌ DON'T**:
- Hardcode paths, values
- Use fallback logic
- Mask errors silently
- Create duplicate configs
- Mix provider and model configs

### Tool Development

**✅ DO**:
- Clear, descriptive tool names
- Comprehensive parameter descriptions
- Always return ToolResult
- Handle all exceptions
- Use type hints
- Document expected behavior

**❌ DON'T**:
- Let exceptions propagate
- Return None or undefined types
- Omit parameter descriptions
- Create tools without clear purpose
- Mix multiple concerns in one tool

### Testing

**✅ DO**:
- Test agent in REPL mode first
- Verify tool access (check tool_categories)
- Test with simple tasks before complex ones
- Check log output for errors
- Validate configurations load

**❌ DON'T**:
- Skip testing simple cases
- Ignore warnings/errors
- Test only complex scenarios
- Assume configs load correctly

---

## Troubleshooting

### "Resource 'X' not found"

**Cause**: Configuration not loaded or role assignment missing

**Fix**:
1. Check `roles/assignments.py` has the role defined
2. Verify config file has correct decorator
3. Ensure config file is in correct directory
4. Check for syntax errors in config file
5. Restart agent (configs load at startup)

```python
# roles/assignments.py must have:
"""Role assignments."""

import logging
from flowlib.config.role_manager import role_manager

logger = logging.getLogger(__name__)

try:
    role_manager.assign_role("default-llm", "example-llamacpp-provider")
    role_manager.assign_role("default-model", "example-model")
    role_manager.assign_role("default-embedding", "example-llamacpp-embedding-provider")
    role_manager.assign_role("default-embedding-model", "example-embedding-model")
    role_manager.assign_role("default-vector-db", "example-vector-db-provider")
    role_manager.assign_role("default-graph-db", "example-graph-db-provider")
    
    logger.info("Role assignments completed successfully")
except Exception as e:
    logger.warning(f"Failed to create some role assignments: {e}")
```

### "Tool not found" or "Tool not accessible"

**Cause**: Tool category missing from `allowed_tool_categories`

**Fix**: Update your agent config:

```python
allowed_tool_categories = [
    "generic",  # For read, write, bash, etc.
    "custom",   # For your custom tools
]
```

### Agent doesn't respond or hangs

**Possible Causes**:
- Model loading (first run takes time)
- GPU memory issues
- Infinite loop in tool/flow
- max_iterations reached

**Fix**:
1. Check logs for errors
2. Reduce n_gpu_layers if OOM
3. Add timeout to long operations
4. Increase max_iterations if needed

### Import errors

**Cause**: Usually function-level imports

**Fix**: Move all imports to top of file:

```python
# ❌ WRONG
def my_function():
    import something  # NO!

# ✅ CORRECT
import something  # At top of file

def my_function():
    # Use it here
```

### "Failed to initialize agent"

**Causes**:
- Missing required configs
- Invalid provider settings
- Model file not found
- Dependency not installed

**Fix**:
1. Check all required roles are assigned
2. Verify model file paths exist
3. Check provider settings are valid
4. Install required packages (llama-cpp-python, etc.)

### Multimodal model not working

**Causes**:
- Missing mmproj file
- Wrong chat_format
- Handler not available (e.g., Gemma3ChatHandler)

**Fix**:
1. Verify both model and mmproj files exist
2. Check chat_format matches model type
3. Update llama-cpp-python if handler missing
4. Check model and mmproj are compatible

---

## Cheat Sheet

### Minimum Viable Agent

```bash
# 1. Create structure
mkdir -p projects/my-agent/{agents,configs,roles}
touch projects/my-agent/{agents,configs,roles}/__init__.py

# 2. Create agent config (agents/my_agent.py)
@agent_config("my-agent")
class MyAgentConfig:
    persona = "I am..."
    allowed_tool_categories = ["generic"]
    model_name = "default-model"
    llm_name = "default-llm"

# 3. Create role assignments (roles/assignments.py)
"""Role assignments."""

import logging
from flowlib.config.role_manager import role_manager

logger = logging.getLogger(__name__)

try:
    role_manager.assign_role("default-llm", "example-llamacpp-provider")
    role_manager.assign_role("default-model", "example-model")
    role_manager.assign_role("default-embedding", "example-llamacpp-embedding-provider")
    role_manager.assign_role("default-embedding-model", "example-embedding-model")
    role_manager.assign_role("default-vector-db", "example-vector-db-provider")
    role_manager.assign_role("default-graph-db", "example-graph-db-provider")
    
    logger.info("Role assignments completed successfully")
except Exception as e:
    logger.warning(f"Failed to create some role assignments: {e}")

# 4. Create run script (run_my_agent.sh in project root)
#!/bin/bash
./run_agent run --project ./projects/my-agent --agent my-agent --mode repl

# 5. Run!
chmod +x run_my_agent.sh
./run_my_agent.sh
```

### Quick Reference

**Decorators**:
- `@agent_config("name")` - Agent configuration
- `@llm_config("name")` - LLM provider
- `@llm_config("name", provider_type, config={...})` - LLM model
- `@multimodal_llm_provider_config("name")` - Multimodal provider
- `@multimodal_llm_config("name", provider_type, config={...})` - Multimodal model
- `@embedding_config("name", provider_type, config={...})` - Embedding
- `@vector_db_config("name")` - Vector database
- `@graph_db_config("name")` - Graph database
- `@tool(name, description, category)` - Custom tool
- `@flow(name, description)` - Custom flow

**Tool Categories**:
- `generic` - read, write, edit, bash, conversation
- `software` - Code tools
- `systems` - System admin tools
- `data` - Data processing tools
- `custom` - Your domain tools

**Execution Modes**:
- `repl` - Interactive mode
- `autonomous` - Run until complete
- `daemon` - Background service (custom runner required)

---

## Additional Resources

- **Flowlib Core Docs**: `flowlib/docs/`
- **Example Projects**:
  - `projects/book-writer/` - Simple REPL agent
  - `projects/systems-engineer/` - Technical agent
  - `projects/english-teacher/` - Multimodal vision agent
  - `projects/field-engineer/` - Complex daemon agent
- **Provider Configs**: `flowlib/resources/example_configs/`
- **Built-in Tools**: `flowlib/agent/components/task/execution/tool_implementations/`

---

**Last Updated**: 2025-11-04
**Version**: 1.0.0
