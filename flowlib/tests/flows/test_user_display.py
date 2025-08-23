"""Tests for flows user display functionality."""

import pytest
from unittest.mock import Mock
from typing import Protocol, Optional

from flowlib.flows.user_display.user_display import (
    UserDisplayable,
    extract_user_display,
    format_flow_output_for_user,
    _format_shell_command_output,
    _format_conversation_output
)


class TestUserDisplayableProtocol:
    """Test UserDisplayable protocol functionality."""
    
    def test_protocol_detection(self):
        """Test that UserDisplayable protocol can be detected."""
        class DisplayableClass:
            def get_user_display(self) -> str:
                return "Test display text"
        
        instance = DisplayableClass()
        
        # Should be detected as implementing UserDisplayable
        assert isinstance(instance, UserDisplayable)
    
    def test_protocol_not_implemented(self):
        """Test classes that don't implement UserDisplayable."""
        class NonDisplayableClass:
            pass
        
        instance = NonDisplayableClass()
        
        # Should not be detected as implementing UserDisplayable
        assert not isinstance(instance, UserDisplayable)
    
    def test_protocol_method_signature(self):
        """Test that get_user_display method has correct signature."""
        class DisplayableClass:
            def get_user_display(self) -> str:
                return "Display text"
        
        instance = DisplayableClass()
        result = instance.get_user_display()
        
        assert isinstance(result, str)
        assert result == "Display text"


class TestExtractUserDisplay:
    """Test extract_user_display function."""
    
    def test_extract_from_displayable_object(self):
        """Test extracting display text from UserDisplayable object."""
        class DisplayableResult:
            def get_user_display(self) -> str:
                return "Custom display text"
        
        result = DisplayableResult()
        display_text = extract_user_display(result)
        
        assert display_text == "Custom display text"
    
    def test_extract_from_object_with_method(self):
        """Test extracting from object with get_user_display method."""
        class ResultWithMethod:
            def get_user_display(self):
                return "Method display text"
        
        result = ResultWithMethod()
        display_text = extract_user_display(result)
        
        assert display_text == "Method display text"
    
    def test_extract_method_exception_fallback(self):
        """Test fallback when get_user_display method raises exception."""
        class BrokenDisplayMethod:
            def get_user_display(self):
                raise RuntimeError("Display method failed")
            
            def __init__(self):
                self.result = "Fallback text"
        
        result = BrokenDisplayMethod()
        display_text = extract_user_display(result)
        
        assert display_text == "Fallback text"
    
    def test_extract_from_dict_user_display(self):
        """Test extracting from dictionary with user_display field."""
        result = {"user_display": "Dict display text", "other_field": "ignored"}
        display_text = extract_user_display(result)
        
        assert display_text == "Dict display text"
    
    def test_extract_from_dict_display_text(self):
        """Test extracting from dictionary with display_text field."""
        result = {"display_text": "Display text field", "other_field": "ignored"}
        display_text = extract_user_display(result)
        
        assert display_text == "Display text field"
    
    def test_extract_from_dict_summary(self):
        """Test extracting from dictionary with summary field."""
        result = {"summary": "Summary text", "other_field": "ignored"}
        display_text = extract_user_display(result)
        
        assert display_text == "Summary text"
    
    def test_extract_from_dict_message(self):
        """Test extracting from dictionary with message field."""
        result = {"message": "Message text", "other_field": "ignored"}
        display_text = extract_user_display(result)
        
        assert display_text == "Message text"
    
    def test_extract_from_dict_response(self):
        """Test extracting from dictionary with response field."""
        result = {"response": "Response text", "other_field": "ignored"}
        display_text = extract_user_display(result)
        
        assert display_text == "Response text"
    
    def test_extract_from_dict_output(self):
        """Test extracting from dictionary with output field."""
        result = {"output": "Output text", "other_field": "ignored"}
        display_text = extract_user_display(result)
        
        assert display_text == "Output text"
    
    def test_extract_from_dict_result(self):
        """Test extracting from dictionary with result field."""
        result = {"result": "Result text", "other_field": "ignored"}
        display_text = extract_user_display(result)
        
        assert display_text == "Result text"
    
    def test_extract_field_precedence(self):
        """Test that display fields are checked in correct precedence order."""
        result = {
            "result": "Result text",
            "user_display": "User display text",
            "summary": "Summary text"
        }
        display_text = extract_user_display(result)
        
        # user_display should take precedence
        assert display_text == "User display text"
    
    def test_extract_from_object_attributes(self):
        """Test extracting from object attributes."""
        class ResultObject:
            def __init__(self):
                self.summary = "Object summary"
                self.other_attr = "ignored"
        
        result = ResultObject()
        display_text = extract_user_display(result)
        
        assert display_text == "Object summary"
    
    def test_extract_empty_field_ignored(self):
        """Test that empty fields are ignored."""
        result = {
            "user_display": "",  # Empty, should be ignored
            "summary": "Summary text"
        }
        display_text = extract_user_display(result)
        
        assert display_text == "Summary text"
    
    def test_extract_non_string_field_ignored(self):
        """Test that non-string fields are ignored."""
        result = {
            "user_display": 123,  # Not a string, should be ignored
            "summary": "Summary text"
        }
        display_text = extract_user_display(result)
        
        assert display_text == "Summary text"
    
    def test_extract_no_displayable_content(self):
        """Test extraction when no displayable content is found."""
        result = {"irrelevant_field": "irrelevant value"}
        display_text = extract_user_display(result)
        
        assert display_text is None
    
    def test_extract_from_non_dict_non_object(self):
        """Test extraction from primitive types."""
        assert extract_user_display("string") is None
        assert extract_user_display(123) is None
        assert extract_user_display([1, 2, 3]) is None


class TestFormatFlowOutputForUser:
    """Test format_flow_output_for_user function."""
    
    def test_format_with_user_display(self):
        """Test formatting when result has user display text."""
        class DisplayableResult:
            def get_user_display(self) -> str:
                return "Custom formatted output"
        
        result = DisplayableResult()
        formatted = format_flow_output_for_user("test-flow", result)
        
        assert formatted == "Custom formatted output"
    
    def test_format_shell_command_flow(self):
        """Test formatting for shell-command flow."""
        result = {
            "command": "ls -la",
            "stdout": "file1.txt\nfile2.txt",
            "stderr": ""
        }
        formatted = format_flow_output_for_user("shell-command", result, success=True)
        
        assert "Command executed successfully:" in formatted
        assert "ls -la" in formatted
        assert "file1.txt" in formatted
    
    def test_format_conversation_flow(self):
        """Test formatting for conversation flow."""
        result = {"response": "Hello, how can I help you?"}
        formatted = format_flow_output_for_user("conversation", result, success=True)
        
        assert formatted == "Hello, how can I help you?"
    
    def test_format_generic_success(self):
        """Test generic formatting for successful execution."""
        result = {"some_data": "value"}
        formatted = format_flow_output_for_user("custom-flow", result, success=True)
        
        assert formatted == "✅ custom-flow completed successfully"
    
    def test_format_generic_failure(self):
        """Test generic formatting for failed execution."""
        result = {"error": "Something went wrong"}
        formatted = format_flow_output_for_user("custom-flow", result, success=False)
        
        assert formatted == "❌ custom-flow failed: Something went wrong"
    
    def test_format_generic_failure_no_error_field(self):
        """Test generic formatting for failure without error field."""
        result = {"some_data": "value"}
        formatted = format_flow_output_for_user("custom-flow", result, success=False)
        
        assert formatted == "❌ custom-flow failed: Unknown error"
    
    def test_format_failure_with_error_attribute(self):
        """Test formatting for failure with error attribute."""
        class ResultWithError:
            def __init__(self):
                self.error = "Specific error message"
        
        result = ResultWithError()
        formatted = format_flow_output_for_user("custom-flow", result, success=False)
        
        assert formatted == "❌ custom-flow failed: Specific error message"


class TestFormatShellCommandOutput:
    """Test _format_shell_command_output function."""
    
    def test_format_successful_with_stdout(self):
        """Test formatting successful command with stdout."""
        result = {
            "command": "echo 'hello'",
            "stdout": "hello\n",
            "stderr": ""
        }
        formatted = _format_shell_command_output(result, success=True)
        
        assert "Command executed successfully:" in formatted
        assert "echo 'hello'" in formatted
        assert "hello" in formatted
        assert "```bash" in formatted
    
    def test_format_failed_with_stderr(self):
        """Test formatting failed command with stderr."""
        result = {
            "command": "invalid_command",
            "stdout": "",
            "stderr": "command not found\n"
        }
        formatted = _format_shell_command_output(result, success=False)
        
        assert "Command failed:" in formatted
        assert "invalid_command" in formatted
        assert "command not found" in formatted
        assert "Error:" in formatted
    
    def test_format_command_no_output(self):
        """Test formatting command with no output."""
        result = {
            "command": "touch file.txt",
            "stdout": "",
            "stderr": ""
        }
        formatted = _format_shell_command_output(result, success=True)
        
        assert "Command executed:" in formatted
        assert "touch file.txt" in formatted
        assert "(No output)" in formatted
    
    def test_format_no_command(self):
        """Test formatting when no command is available."""
        result = {"stdout": "some output", "stderr": ""}
        formatted = _format_shell_command_output(result, success=True)
        
        assert formatted == "Shell command executed"
    
    def test_format_object_attributes(self):
        """Test formatting with object attributes."""
        class CommandResult:
            def __init__(self):
                self.command = "pwd"
                self.stdout = "/home/user\n"
                self.stderr = ""
        
        result = CommandResult()
        formatted = _format_shell_command_output(result, success=True)
        
        assert "pwd" in formatted
        assert "/home/user" in formatted
    
    def test_format_non_dict_non_object(self):
        """Test formatting with primitive type."""
        formatted = _format_shell_command_output("string", success=True)
        
        assert formatted == "Shell command executed"


class TestFormatConversationOutput:
    """Test _format_conversation_output function."""
    
    def test_format_with_response(self):
        """Test formatting conversation with response field."""
        result = {"response": "Hello, I'm here to help!"}
        formatted = _format_conversation_output(result, success=True)
        
        assert formatted == "Hello, I'm here to help!"
    
    def test_format_with_message(self):
        """Test formatting conversation with message field."""
        result = {"message": "This is a message"}
        formatted = _format_conversation_output(result, success=True)
        
        assert formatted == "This is a message"
    
    def test_format_response_precedence(self):
        """Test that response field takes precedence over message."""
        result = {
            "response": "Response text",
            "message": "Message text"
        }
        formatted = _format_conversation_output(result, success=True)
        
        assert formatted == "Response text"
    
    def test_format_no_response_or_message(self):
        """Test formatting when no response or message is available."""
        result = {"other_field": "other value"}
        formatted = _format_conversation_output(result, success=True)
        
        assert formatted == "Conversation completed"
    
    def test_format_object_attributes(self):
        """Test formatting with object attributes."""
        class ConversationResult:
            def __init__(self):
                self.response = "Object response"
        
        result = ConversationResult()
        formatted = _format_conversation_output(result, success=True)
        
        assert formatted == "Object response"
    
    def test_format_non_dict_non_object(self):
        """Test formatting with primitive type."""
        formatted = _format_conversation_output("string", success=True)
        
        assert formatted == "Conversation completed"