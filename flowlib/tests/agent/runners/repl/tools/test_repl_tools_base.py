"""Tests for base REPL tool classes."""

import json
import pytest
from typing import List
from unittest.mock import Mock, patch, MagicMock

from flowlib.agent.runners.repl.tools.base import (
    ToolResultStatus,
    ToolResult,
    ToolParameter,
    REPLTool,
    ToolRegistry
)


class TestToolResultStatus:
    """Test suite for ToolResultStatus enum."""
    
    def test_enum_values(self):
        """Test that enum has expected values."""
        assert ToolResultStatus.SUCCESS.value == "success"
        assert ToolResultStatus.ERROR.value == "error"
        assert ToolResultStatus.WARNING.value == "warning"


class TestToolResult:
    """Test suite for ToolResult dataclass."""
    
    def test_basic_creation(self):
        """Test creating a basic ToolResult."""
        result = ToolResult(
            status=ToolResultStatus.SUCCESS,
            content="Test content"
        )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert result.content == "Test content"
        assert result.error is None
        assert result.metadata is None
    
    def test_full_creation(self):
        """Test creating a ToolResult with all fields."""
        metadata = {"key": "value", "count": 42}
        result = ToolResult(
            status=ToolResultStatus.ERROR,
            content=None,
            error="Test error",
            metadata=metadata
        )
        
        assert result.status == ToolResultStatus.ERROR
        assert result.content is None
        assert result.error == "Test error"
        assert result.metadata == metadata
    
    def test_to_dict(self):
        """Test converting ToolResult to dictionary."""
        result = ToolResult(
            status=ToolResultStatus.SUCCESS,
            content="Test content",
            error=None,
            metadata={"key": "value"}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["status"] == "success"
        assert result_dict["content"] == "Test content"
        assert result_dict["error"] is None
        assert result_dict["metadata"] == {"key": "value"}
    
    def test_to_dict_empty_metadata(self):
        """Test to_dict with None metadata returns empty dict."""
        result = ToolResult(
            status=ToolResultStatus.SUCCESS,
            content="Test"
        )
        
        result_dict = result.to_dict()
        assert result_dict["metadata"] == {}


class TestToolParameter:
    """Test suite for ToolParameter model."""
    
    def test_required_fields(self):
        """Test creating ToolParameter with required fields."""
        param = ToolParameter(
            name="test_param",
            type="str",
            description="A test parameter"
        )
        
        assert param.name == "test_param"
        assert param.type == "str"
        assert param.description == "A test parameter"
        assert param.required is True  # Default
        assert param.default is None
    
    def test_all_fields(self):
        """Test creating ToolParameter with all fields."""
        param = ToolParameter(
            name="optional_param",
            type="int",
            description="An optional parameter",
            required=False,
            default=42
        )
        
        assert param.name == "optional_param"
        assert param.type == "int"
        assert param.required is False
        assert param.default == 42


# Create a concrete implementation for testing
class MockTestTool(REPLTool):
    """Test implementation of REPLTool."""
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="required_str",
                type="str",
                description="A required string"
            ),
            ToolParameter(
                name="optional_int",
                type="int",
                description="An optional integer",
                required=False,
                default=10
            ),
            ToolParameter(
                name="bool_param",
                type="bool",
                description="A boolean parameter",
                required=False
            ),
            ToolParameter(
                name="list_param",
                type="list",
                description="A list parameter",
                required=False
            )
        ]
    
    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            content=f"Executed with: {kwargs}",
            metadata=kwargs
        )


class TestREPLTool:
    """Test suite for REPLTool abstract base class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = MockTestTool()
    
    def test_initialization(self):
        """Test tool initialization."""
        assert self.tool.name == "mocktest"  # MockTestTool -> mocktest
        assert self.tool.description == "Test implementation of REPLTool."
        assert len(self.tool.parameters) == 4
    
    def test_custom_docstring(self):
        """Test that docstring becomes description."""
        class CustomDocTool(REPLTool):
            """Custom tool description"""
            def _define_parameters(self): return []
            async def execute(self, **kwargs): return ToolResult(ToolResultStatus.SUCCESS, "")
        
        tool = CustomDocTool()
        assert tool.description == "Custom tool description"
    
    def test_no_docstring(self):
        """Test default description when no docstring."""
        class NoDocTool(REPLTool):
            def _define_parameters(self): return []
            async def execute(self, **kwargs): return ToolResult(ToolResultStatus.SUCCESS, "")
        
        tool = NoDocTool()
        assert tool.description == "nodoc tool"  # NoDocTool -> nodoc
    
    def test_validate_parameters_success(self):
        """Test successful parameter validation."""
        validated = self.tool.validate_parameters(
            required_str="test",
            optional_int=20,
            bool_param=True,
            list_param=["a", "b", "c"]
        )
        
        assert validated["required_str"] == "test"
        assert validated["optional_int"] == 20
        assert validated["bool_param"] is True
        assert validated["list_param"] == ["a", "b", "c"]
    
    def test_validate_parameters_missing_required(self):
        """Test validation fails for missing required parameter."""
        with pytest.raises(ValueError, match="Required parameter 'required_str' missing"):
            self.tool.validate_parameters()
    
    def test_validate_parameters_defaults(self):
        """Test default values are used."""
        validated = self.tool.validate_parameters(required_str="test")
        
        assert validated["required_str"] == "test"
        assert validated["optional_int"] == 10  # Default value
        assert validated["bool_param"] is None
        assert validated["list_param"] is None
    
    def test_validate_parameters_type_conversion(self):
        """Test parameter type conversion."""
        # String conversion
        validated = self.tool.validate_parameters(
            required_str=123  # Will be converted to string
        )
        assert validated["required_str"] == "123"
        
        # Integer conversion
        validated = self.tool.validate_parameters(
            required_str="test",
            optional_int="42"  # String to int
        )
        assert validated["optional_int"] == 42
        
        # Boolean conversion
        validated = self.tool.validate_parameters(
            required_str="test",
            bool_param="true"  # String to bool
        )
        assert validated["bool_param"] is True
    
    def test_validate_parameters_bool_conversions(self):
        """Test various boolean conversions."""
        test_cases = [
            (True, True),
            ("true", True),
            ("True", True),
            ("1", True),
            (1, True),
            (False, False),
            ("false", False),
            ("False", False),
            ("0", False),
            (0, False),
            ("anything else", False)
        ]
        
        for input_val, expected in test_cases:
            validated = self.tool.validate_parameters(
                required_str="test",
                bool_param=input_val
            )
            assert validated["bool_param"] == expected
    
    def test_validate_parameters_list_json(self):
        """Test list parameter parsing from JSON."""
        validated = self.tool.validate_parameters(
            required_str="test",
            list_param='["a", "b", "c"]'  # JSON string
        )
        assert validated["list_param"] == ["a", "b", "c"]
    
    def test_validate_parameters_list_comma_separated(self):
        """Test list parameter parsing from comma-separated string."""
        validated = self.tool.validate_parameters(
            required_str="test",
            list_param="a, b, c"  # Comma-separated
        )
        assert validated["list_param"] == ["a", "b", "c"]
    
    def test_validate_parameters_invalid_int(self):
        """Test error for invalid integer conversion."""
        with pytest.raises(ValueError, match="must be an integer"):
            self.tool.validate_parameters(
                required_str="test",
                optional_int="not_a_number"
            )
    
    def test_get_schema(self):
        """Test schema generation."""
        schema = self.tool.get_schema()
        
        assert schema["name"] == "mocktest"  # MockTestTool -> mocktest
        assert schema["description"] == "Test implementation of REPLTool."
        assert "parameters" in schema
        
        params = schema["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params
        
        # Check properties
        props = params["properties"]
        assert "required_str" in props
        assert props["required_str"]["type"] == "str"
        assert props["required_str"]["description"] == "A required string"
        
        assert "optional_int" in props
        assert props["optional_int"]["type"] == "int"
        assert props["optional_int"]["default"] == 10
        
        # Check required list
        assert params["required"] == ["required_str"]
    
    @pytest.mark.asyncio
    async def test_execute_implementation(self):
        """Test execute method is called correctly."""
        result = await self.tool.execute(required_str="test", extra_param="value")
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "required_str" in result.metadata
        assert result.metadata["required_str"] == "test"


class TestToolRegistry:
    """Test suite for ToolRegistry."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the _register_default_tools to avoid loading real tools
        with patch.object(ToolRegistry, '_register_default_tools'):
            self.registry = ToolRegistry()
    
    def test_initialization(self):
        """Test registry initialization."""
        assert isinstance(self.registry.tools, dict)
        assert len(self.registry.tools) == 0  # No tools because we mocked default registration
    
    def test_register_tool(self):
        """Test registering a tool."""
        tool = MockTestTool()
        self.registry.register(tool)
        
        assert "mocktest" in self.registry.tools
        assert self.registry.tools["mocktest"] is tool
    
    def test_get_tool(self):
        """Test getting a tool by name."""
        tool = MockTestTool()
        self.registry.register(tool)
        
        retrieved = self.registry.get("mocktest")
        assert retrieved is tool
        
        # Test non-existent tool
        # Removed redundant context.get() test - strict validation
    
    def test_list_tools(self):
        """Test listing all tool names."""
        tool1 = MockTestTool()
        
        class AnotherTool(MockTestTool):
            pass
        
        tool2 = AnotherTool()
        
        self.registry.register(tool1)
        self.registry.register(tool2)
        
        tools = self.registry.list_tools()
        assert len(tools) == 2
        assert "mocktest" in tools
        assert "another" in tools  # AnotherTool -> another
    
    def test_get_schemas(self):
        """Test getting schemas for all tools."""
        tool = MockTestTool()
        self.registry.register(tool)
        
        schemas = self.registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "mocktest"
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test executing a tool through registry."""
        tool = MockTestTool()
        self.registry.register(tool)
        
        result = await self.registry.execute_tool(
            "mocktest",
            required_str="test value"
        )
        
        assert result.status == ToolResultStatus.SUCCESS
        assert "required_str" in result.metadata
    
    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test executing non-existent tool."""
        result = await self.registry.execute_tool("nonexistent")
        
        assert result.status == ToolResultStatus.ERROR
        assert "Tool 'nonexistent' not found" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_tool_validation_error(self):
        """Test executing tool with validation error."""
        tool = MockTestTool()
        self.registry.register(tool)
        
        # Missing required parameter
        result = await self.registry.execute_tool("mocktest")
        
        assert result.status == ToolResultStatus.ERROR
        assert "Required parameter" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_tool_execution_error(self):
        """Test handling execution errors."""
        class ErrorTool(REPLTool):
            def _define_parameters(self): return []
            async def execute(self, **kwargs):
                raise RuntimeError("Execution failed")
        
        tool = ErrorTool()
        self.registry.register(tool)
        
        result = await self.registry.execute_tool("error")  # ErrorTool -> error
        
        assert result.status == ToolResultStatus.ERROR
        assert "Execution failed" in result.error
    
    def test_register_tools_from_module(self):
        """Test registering tools from a module."""
        # Create a module-like object with explicit attributes
        class MockModule:
            pass
        
        mock_module = MockModule()
        mock_module.TestTool1 = type('TestTool1', (MockTestTool,), {})
        mock_module.TestTool2 = type('TestTool2', (MockTestTool,), {})
        # NonExistentTool is not added, so hasattr will return False
        
        # Capture any print calls
        import builtins
        with patch('importlib.import_module', return_value=mock_module):
            # Suppress print output during test
            with patch.object(builtins, 'print'):
                self.registry._register_tools_from_module(
                    'test_module',
                    ['TestTool1', 'TestTool2', 'NonExistentTool']
                )
        
        # Should register 2 tools (NonExistentTool should be skipped)
        assert len(self.registry.tools) == 2
        assert "test1" in self.registry.tools  # TestTool1 -> test1
        assert "test2" in self.registry.tools  # TestTool2 -> test2
    
    def test_register_tools_import_error(self):
        """Test handling import errors gracefully."""
        with patch('importlib.import_module', side_effect=ImportError("Module not found")):
            # Should not raise, just print warning
            self.registry._register_tools_from_module('nonexistent', ['Tool'])
        
        # Registry should remain empty
        assert len(self.registry.tools) == 0
    
    def test_register_default_tools(self):
        """Test default tools registration."""
        # Create a new registry without mocking
        registry = ToolRegistry()
        
        # Should have registered some tools
        # The exact number may vary, but there should be some
        assert len(registry.tools) > 0
        
        # Check some expected tools exist
        expected_tools = ['read', 'write', 'edit', 'bash', 'glob', 'grep']
        registered_tools = registry.list_tools()
        
        for tool_name in expected_tools:
            assert any(tool_name in registered for registered in registered_tools)