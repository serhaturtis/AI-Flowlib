"""Tests for agent classification models."""

import pytest
from typing import Dict, List
from pydantic import ValidationError

from flowlib.agent.components.classification.models import (
    MessageClassification,
    MessageClassifierInput
)


class TestMessageClassification:
    """Test MessageClassification model."""
    
    def test_message_classification_creation_minimal(self):
        """Test creating MessageClassification with minimal required fields."""
        classification = MessageClassification(
            execute_task=True,
            confidence=0.85,
            category="instruction"
        )
        
        assert classification.execute_task is True
        assert classification.confidence == 0.85
        assert classification.category == "instruction"
        assert classification.task_description is None
    
    def test_message_classification_creation_with_task(self):
        """Test creating MessageClassification with task description."""
        classification = MessageClassification(
            execute_task=True,
            confidence=0.92,
            category="complex_task",
            task_description="Execute the specified command"
        )
        
        assert classification.execute_task is True
        assert classification.confidence == 0.92
        assert classification.category == "complex_task"
        assert classification.task_description == "Execute the specified command"
    
    def test_message_classification_conversation(self):
        """Test creating MessageClassification for conversation."""
        classification = MessageClassification(
            execute_task=False,
            confidence=0.95,
            category="greeting"
        )
        
        assert classification.execute_task is False
        assert classification.confidence == 0.95
        assert classification.category == "greeting"
        assert classification.task_description is None
    
    def test_message_classification_confidence_boundaries(self):
        """Test MessageClassification with confidence boundary values."""
        # Test minimum confidence
        classification_min = MessageClassification(
            execute_task=False,
            confidence=0.0,
            category="uncertain"
        )
        assert classification_min.confidence == 0.0
        
        # Test maximum confidence
        classification_max = MessageClassification(
            execute_task=True,
            confidence=1.0,
            category="clear_instruction"
        )
        assert classification_max.confidence == 1.0
    
    def test_message_classification_validation_missing_fields(self):
        """Test MessageClassification validation with missing required fields."""
        # Missing execute_task
        with pytest.raises(ValidationError):
            MessageClassification(
                confidence=0.8,
                category="test"
            )
        
        # Missing confidence
        with pytest.raises(ValidationError):
            MessageClassification(
                execute_task=True,
                category="test"
            )
        
        # Missing category
        with pytest.raises(ValidationError):
            MessageClassification(
                execute_task=True,
                confidence=0.8
            )
    
    def test_message_classification_invalid_confidence(self):
        """Test MessageClassification with invalid confidence values."""
        # Note: Pydantic doesn't enforce range validation by default
        # These tests verify the model accepts values outside 0-1 range
        # (range validation should be handled by the flow logic)
        
        classification_negative = MessageClassification(
            execute_task=True,
            confidence=-0.5,
            category="test"
        )
        assert classification_negative.confidence == -0.5
        
        classification_over_one = MessageClassification(
            execute_task=True,
            confidence=1.5,
            category="test"
        )
        assert classification_over_one.confidence == 1.5
    
    def test_message_classification_serialization(self):
        """Test MessageClassification serialization."""
        classification = MessageClassification(
            execute_task=True,
            confidence=0.78,
            category="file_operation",
            task_description="Create a new file with specified content"
        )
        
        data = classification.model_dump()
        
        assert data["execute_task"] is True
        assert data["confidence"] == 0.78
        assert data["category"] == "file_operation"
        assert data["task_description"] == "Create a new file with specified content"
    
    def test_message_classification_deserialization(self):
        """Test MessageClassification deserialization."""
        data = {
            "execute_task": False,
            "confidence": 0.91,
            "category": "acknowledgment",
            "task_description": None
        }
        
        classification = MessageClassification(**data)
        
        assert classification.execute_task is False
        assert classification.confidence == 0.91
        assert classification.category == "acknowledgment"
        assert classification.task_description is None
    
    def test_message_classification_json_serialization(self):
        """Test MessageClassification JSON serialization."""
        classification = MessageClassification(
            execute_task=True,
            confidence=0.83,
            category="data_analysis",
            task_description="Analyze the provided dataset"
        )
        
        json_str = classification.model_dump_json()
        
        # Verify it's valid JSON
        import json
        data = json.loads(json_str)
        
        assert data["execute_task"] is True
        assert data["confidence"] == 0.83
        assert data["category"] == "data_analysis"
        assert data["task_description"] == "Analyze the provided dataset"


class TestMessageClassifierInput:
    """Test MessageClassifierInput model."""
    
    def test_message_classifier_input_minimal(self):
        """Test creating MessageClassifierInput with minimal required fields."""
        input_data = MessageClassifierInput(message="Hello, how are you?")
        
        assert input_data.message == "Hello, how are you?"
        assert input_data.conversation_history == []
        assert input_data.memory_context_summary is None
    
    def test_message_classifier_input_with_history(self):
        """Test creating MessageClassifierInput with conversation history."""
        history = [
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "I'd need to check current weather data for you."}
        ]
        
        input_data = MessageClassifierInput(
            message="Can you check it now?",
            conversation_history=history
        )
        
        assert input_data.message == "Can you check it now?"
        assert len(input_data.conversation_history) == len(history)
        for i, msg in enumerate(input_data.conversation_history):
            assert msg.role == history[i]["role"]
            assert msg.content == history[i]["content"]
        assert input_data.memory_context_summary is None
    
    def test_message_classifier_input_with_memory_context(self):
        """Test creating MessageClassifierInput with memory context."""
        input_data = MessageClassifierInput(
            message="Continue with the analysis",
            memory_context_summary="Previous conversation about data analysis project"
        )
        
        assert input_data.message == "Continue with the analysis"
        assert input_data.conversation_history == []
        assert input_data.memory_context_summary == "Previous conversation about data analysis project"
    
    def test_message_classifier_input_complete(self):
        """Test creating MessageClassifierInput with all fields."""
        history = [
            {"role": "user", "content": "I need help with my project"},
            {"role": "assistant", "content": "I'd be happy to help. What kind of project?"},
            {"role": "user", "content": "It's a data analysis project"}
        ]
        
        input_data = MessageClassifierInput(
            message="Can you create a Python script for data visualization?",
            conversation_history=history,
            memory_context_summary="User is working on data analysis with Python and needs visualization"
        )
        
        assert input_data.message == "Can you create a Python script for data visualization?"
        assert len(input_data.conversation_history) == len(history)
        for i, msg in enumerate(input_data.conversation_history):
            assert msg.role == history[i]["role"]
            assert msg.content == history[i]["content"]
        assert input_data.memory_context_summary == "User is working on data analysis with Python and needs visualization"
    
    def test_message_classifier_input_validation_missing_message(self):
        """Test MessageClassifierInput validation with missing message."""
        with pytest.raises(ValidationError):
            MessageClassifierInput()
    
    def test_message_classifier_input_empty_message(self):
        """Test MessageClassifierInput with empty message."""
        input_data = MessageClassifierInput(message="")
        
        assert input_data.message == ""
        assert input_data.conversation_history == []
    
    def test_message_classifier_input_invalid_history_format(self):
        """Test MessageClassifierInput with invalid conversation history format."""
        # History should be list of dicts, but we test with invalid formats
        with pytest.raises(ValidationError):
            MessageClassifierInput(
                message="Test message",
                conversation_history="not a list"
            )
        
        with pytest.raises(ValidationError):
            MessageClassifierInput(
                message="Test message",
                conversation_history=[{"invalid": "format", "missing": "role_and_content"}]
            )
    
    def test_message_classifier_input_serialization(self):
        """Test MessageClassifierInput serialization."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        input_data = MessageClassifierInput(
            message="How can I help you today?",
            conversation_history=history,
            memory_context_summary="Greeting context"
        )
        
        data = input_data.model_dump()
        
        assert data["message"] == "How can I help you today?"
        assert data["conversation_history"] == history
        assert data["memory_context_summary"] == "Greeting context"
    
    def test_message_classifier_input_deserialization(self):
        """Test MessageClassifierInput deserialization."""
        data = {
            "message": "Process this file",
            "conversation_history": [
                {"role": "user", "content": "I have a file to process"},
                {"role": "assistant", "content": "What kind of processing do you need?"}
            ],
            "memory_context_summary": "File processing context"
        }
        
        input_data = MessageClassifierInput(**data)
        
        assert input_data.message == "Process this file"
        assert len(input_data.conversation_history) == 2
        assert input_data.conversation_history[0].role == "user"
        assert input_data.memory_context_summary == "File processing context"


class TestModelIntegration:
    """Test integration between classification models."""
    
    def test_typical_conversation_flow(self):
        """Test typical conversation classification flow."""
        # Input for greeting
        input_data = MessageClassifierInput(
            message="Hello! How are you doing today?"
        )
        
        # Expected classification result
        classification = MessageClassification(
            execute_task=False,
            confidence=0.95,
            category="greeting"
        )
        
        assert input_data.message == "Hello! How are you doing today?"
        assert classification.execute_task is False
        assert classification.category == "greeting"
    
    def test_typical_task_flow(self):
        """Test typical task classification flow."""
        # Input for task request
        input_data = MessageClassifierInput(
            message="Can you create a Python script that reads a CSV file and generates a plot?",
            conversation_history=[
                {"role": "user", "content": "I need help with data visualization"},
                {"role": "assistant", "content": "I can help you with that. What kind of data do you have?"}
            ],
            memory_context_summary="User needs help with Python data visualization"
        )
        
        # Expected classification result
        classification = MessageClassification(
            execute_task=True,
            confidence=0.92,
            category="code_generation",
            task_description="Create a Python script that reads a CSV file and generates a plot"
        )
        
        assert input_data.message.startswith("Can you create a Python script")
        assert len(input_data.conversation_history) == 2
        assert classification.execute_task is True
        assert classification.task_description is not None
    
    def test_classification_with_context_continuation(self):
        """Test classification that depends on conversation context."""
        # Input that references previous context
        input_data = MessageClassifierInput(
            message="Yes, please proceed with that approach",
            conversation_history=[
                {"role": "user", "content": "I need to analyze sales data"},
                {"role": "assistant", "content": "I can help with that. Should I create a Python script using pandas?"}
            ],
            memory_context_summary="User confirmed using Python pandas for sales data analysis"
        )
        
        # This should be classified as a task because it's confirming a task approach
        classification = MessageClassification(
            execute_task=True,
            confidence=0.85,
            category="confirmation_with_action",
            task_description="Proceed with creating Python script using pandas for sales data analysis"
        )
        
        assert input_data.message == "Yes, please proceed with that approach"
        assert classification.execute_task is True
        assert "pandas" in classification.task_description
    
    def test_edge_case_ambiguous_message(self):
        """Test classification of ambiguous messages."""
        # Ambiguous input
        input_data = MessageClassifierInput(
            message="What about the files?"
        )
        
        # Low confidence classification
        classification = MessageClassification(
            execute_task=False,  # Default to conversation for ambiguous cases
            confidence=0.3,
            category="ambiguous_question"
        )
        
        assert input_data.message == "What about the files?"
        assert classification.confidence < 0.5  # Low confidence for ambiguous cases
    
    def test_model_field_constraints(self):
        """Test that models enforce expected field constraints."""
        # Test that execute_task is boolean
        classification = MessageClassification(
            execute_task=True,
            confidence=0.8,
            category="test"
        )
        assert isinstance(classification.execute_task, bool)
        
        # Test that confidence is float
        assert isinstance(classification.confidence, float)
        
        # Test that category is string
        assert isinstance(classification.category, str)
        
        # Test that conversation_history is list
        input_data = MessageClassifierInput(message="test")
        assert isinstance(input_data.conversation_history, list)
    
    def test_serialization_roundtrip(self):
        """Test complete serialization/deserialization roundtrip."""
        # Create input data
        original_input = MessageClassifierInput(
            message="Create a backup of my database",
            conversation_history=[
                {"role": "user", "content": "I'm worried about data loss"},
                {"role": "assistant", "content": "I can help you create a backup"}
            ],
            memory_context_summary="User needs database backup for data protection"
        )
        
        # Serialize and deserialize
        input_data = original_input.model_dump()
        restored_input = MessageClassifierInput(**input_data)
        
        assert restored_input.message == original_input.message
        assert restored_input.conversation_history == original_input.conversation_history
        assert restored_input.memory_context_summary == original_input.memory_context_summary
        
        # Create classification
        original_classification = MessageClassification(
            execute_task=True,
            confidence=0.88,
            category="database_operation",
            task_description="Create a backup of the user's database"
        )
        
        # Serialize and deserialize
        classification_data = original_classification.model_dump()
        restored_classification = MessageClassification(**classification_data)
        
        assert restored_classification.execute_task == original_classification.execute_task
        assert restored_classification.confidence == original_classification.confidence
        assert restored_classification.category == original_classification.category
        assert restored_classification.task_description == original_classification.task_description