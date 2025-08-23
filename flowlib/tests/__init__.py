"""
Comprehensive test suite for Flowlib framework.

This test suite covers all major components of the Flowlib framework:
- Core system (context, errors, registry, validation)
- Flow system (decorators, execution, results)
- Provider system (registration, initialization, contracts)
- Resource system (models, prompts, registration)
- Agent system (core, memory, planning, reflection)
- Utilities (formatting, validation, serialization)
"""

import pytest
import asyncio
import logging
from typing import Any, Dict, Optional
from unittest.mock import Mock, AsyncMock

# Configure test logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests

# Common test fixtures and utilities
class MockProvider:
    """Mock provider for testing."""
    
    def __init__(self, name: str = "test_provider"):
        self.name = name
        self.initialized = False
        
    async def initialize(self):
        """Mock initialization."""
        self.initialized = True
        
    async def shutdown(self):
        """Mock shutdown."""
        self.initialized = False


class MockLLMProvider(MockProvider):
    """Mock LLM provider for testing."""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Mock text generation."""
        return f"Mock response to: {prompt[:50]}"
    
    async def generate_structured(self, prompt: str, schema: dict, **kwargs) -> dict:
        """Mock structured generation."""
        return {"response": "mock_structured_response"}


class MockEmbeddingProvider(MockProvider):
    """Mock embedding provider for testing."""
    
    async def embed(self, text: str) -> list[float]:
        """Mock embedding generation."""
        return [0.1] * 384  # Standard embedding size
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Mock batch embedding."""
        return [[0.1] * 384 for _ in texts]


def create_test_event_loop():
    """Create a new event loop for testing."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


# Test configuration
TEST_CONFIG = {
    "timeout": 30,
    "log_level": "WARNING",
    "mock_external_services": True
}