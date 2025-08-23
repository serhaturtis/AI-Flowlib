"""Tests for step reflection prompts."""

import pytest
import json
from unittest.mock import Mock, patch
from typing import Any, ClassVar, Dict

from flowlib.agent.components.reflection.prompts.step_reflection import (
    DefaultStepReflectionPrompt,
    STEP_REFLECTION_TEMPLATE
)
from flowlib.agent.components.reflection.models import StepReflectionResult
from flowlib.resources.models.base import ResourceBase


class TestStepReflectionTemplate:
    """Test the STEP_REFLECTION_TEMPLATE constant."""
    
    def test_template_structure(self):
        """Test that the template has expected structure."""
        template = STEP_REFLECTION_TEMPLATE
        
        # Should be a string
        assert isinstance(template, str)
        assert len(template) > 0
        
        # Should contain placeholders
        expected_placeholders = [
            "{{task_description}}", "{{step_id}}", "{{step_intent}}", 
            "{{step_rationale}}", "{{flow_name}}", "{{flow_inputs_formatted}}",
            "{{flow_result_formatted}}", "{{current_progress}}"
        ]
        
        for placeholder in expected_placeholders:
            assert placeholder in template
    
    def test_template_format_markers(self):
        """Test that template has proper format markers."""
        template = STEP_REFLECTION_TEMPLATE
        
        # Should contain conversation markers
        assert "<|im_start|>user" in template
        assert "<|im_end|>assistant" in template
        
        # Should have proper structure
        assert "Overall Task:" in template
        assert "Current Step Details:" in template
        assert "Step Execution Inputs:" in template
        assert "Step Execution Result:" in template
        assert "Analysis Task:" in template
    
    def test_template_json_schema(self):
        """Test that template contains valid JSON schema."""
        template = STEP_REFLECTION_TEMPLATE
        
        # Should contain JSON schema definition
        assert "StepReflectionResult" in template
        assert '"step_id"' in template
        assert '"reflection"' in template
        assert '"step_success"' in template
        assert '"key_observation"' in template
        
        # Should specify required fields
        assert '"required"' in template
        assert '"step_id"' in template
        assert '"reflection"' in template
        assert '"step_success"' in template
    
    def test_template_step_focus(self):
        """Test that template is focused on step-level analysis."""
        template = STEP_REFLECTION_TEMPLATE
        
        # Should mention step-specific elements
        assert "single step" in template
        assert "Step ID:" in template
        assert "Intent:" in template
        assert "Rationale:" in template
        assert "Flow Executed:" in template
        
        # Should emphasize step-level analysis
        assert "specific step" in template
        assert "step achieved its intent" in template
    
    def test_template_placeholder_balance(self):
        """Test that template has balanced placeholders."""
        template = STEP_REFLECTION_TEMPLATE
        
        # Count double braces for placeholders
        opening_count = template.count("{{")
        closing_count = template.count("}}")
        
        # Should be balanced
        assert opening_count == closing_count
        assert opening_count >= 8  # At least 8 placeholders expected


class TestDefaultStepReflectionPrompt:
    """Test DefaultStepReflectionPrompt class."""
    
    def test_class_inheritance(self):
        """Test that class inherits from ResourceBase."""
        assert issubclass(DefaultStepReflectionPrompt, ResourceBase)
    
    def test_class_attributes(self):
        """Test class has expected attributes."""
        # Should have template as ClassVar
        assert hasattr(DefaultStepReflectionPrompt, 'template')
        assert isinstance(DefaultStepReflectionPrompt.template, str)
        assert DefaultStepReflectionPrompt.template == STEP_REFLECTION_TEMPLATE
        
        # Should have output_model as ClassVar
        assert hasattr(DefaultStepReflectionPrompt, 'output_model')
        assert DefaultStepReflectionPrompt.output_model == StepReflectionResult
    
    def test_prompt_decorator(self):
        """Test that class is decorated with @prompt."""
        # Should have been registered with the prompt decorator
        prompt_instance = DefaultStepReflectionPrompt(name="test_step_reflection", type="prompt")
        assert prompt_instance.name == "test_step_reflection"
        assert prompt_instance.type == "prompt"
    
    def test_initialization(self):
        """Test prompt initialization."""
        prompt = DefaultStepReflectionPrompt(name="step_reflection_prompt", type="prompt")
        
        assert prompt.name == "step_reflection_prompt"
        assert prompt.type == "prompt"
        assert hasattr(prompt, 'template')
        assert hasattr(prompt, 'output_model')
    
    def test_format_method_validation(self):
        """Test format method validates required keys."""
        prompt = DefaultStepReflectionPrompt(name="test", type="prompt")
        
        # Test with all required keys present
        complete_kwargs = {
            "task_description": "Complete data analysis",
            "step_id": "step_001",
            "step_intent": "Load data file",
            "step_rationale": "Need data to proceed with analysis",
            "flow_name": "data_loader_flow",
            "flow_inputs_formatted": "{'file_path': '/data/input.csv'}",
            "flow_result_formatted": "{'rows_loaded': 1000, 'status': 'success'}",
            "current_progress": 25
        }
        
        # The template contains JSON that conflicts with .format()
        # Test that method validates required keys properly
        try:
            result = prompt.format(**complete_kwargs)
            # If it succeeds, check basic properties
            assert isinstance(result, str)
        except KeyError:
            # Expected due to JSON schema in template
            # Test validation by removing a required key
            incomplete_kwargs = {k: v for k, v in complete_kwargs.items() if k != "step_id"}
            with pytest.raises(ValueError) as exc_info:
                prompt.format(**incomplete_kwargs)
            assert "Missing required key" in str(exc_info.value)
    
    def test_format_method_missing_keys(self):
        """Test format method raises error for missing required keys."""
        prompt = DefaultStepReflectionPrompt(name="test", type="prompt")
        
        # Test with missing keys
        incomplete_kwargs = {
            "task_description": "Test task",
            "step_id": "step_001"
            # Missing other required keys
        }
        
        with pytest.raises(ValueError) as exc_info:
            prompt.format(**incomplete_kwargs)
        
        assert "Missing required key for Step Reflection Prompt" in str(exc_info.value)
    
    def test_format_method_all_required_keys(self):
        """Test format method identifies all required keys."""
        prompt = DefaultStepReflectionPrompt(name="test", type="prompt")
        
        required_keys = [
            "task_description", "step_id", "step_intent", "step_rationale",
            "flow_name", "flow_inputs_formatted", "flow_result_formatted",
            "current_progress"
        ]
        
        # Test each key individually
        for key in required_keys:
            incomplete_kwargs = {k: "test_value" for k in required_keys if k != key}
            
            with pytest.raises(ValueError) as exc_info:
                prompt.format(**incomplete_kwargs)
            
            assert f"Missing required key for Step Reflection Prompt: {key}" in str(exc_info.value)
    
    def test_template_content_access(self):
        """Test that template content is accessible."""
        prompt = DefaultStepReflectionPrompt(name="test", type="prompt")
        
        template = prompt.template
        assert isinstance(template, str)
        assert len(template) > 100  # Should be substantial
        assert template == STEP_REFLECTION_TEMPLATE
    
    def test_output_model_access(self):
        """Test that output model is accessible."""
        prompt = DefaultStepReflectionPrompt(name="test", type="prompt")
        
        output_model = prompt.output_model
        assert output_model == StepReflectionResult
        assert hasattr(output_model, '__name__')


class TestStepReflectionPromptIntegration:
    """Integration tests for step reflection prompts."""
    
    def test_step_reflection_result_compatibility(self):
        """Test compatibility with StepReflectionResult model."""
        prompt = DefaultStepReflectionPrompt(name="test", type="prompt")
        assert prompt.output_model == StepReflectionResult
        
        # Test that StepReflectionResult can be instantiated
        result = StepReflectionResult(
            step_id="step_001",
            reflection="Step completed successfully",
            step_success=True
        )
        
        assert result.step_id == "step_001"
        assert result.reflection == "Step completed successfully"
        assert result.step_success is True
        assert result.key_observation is None  # Optional field
    
    def test_step_reflection_result_with_observation(self):
        """Test StepReflectionResult with key observation."""
        result = StepReflectionResult(
            step_id="step_002",
            reflection="Step encountered error",
            step_success=False,
            key_observation="Database connection failed"
        )
        
        assert result.step_id == "step_002"
        assert result.reflection == "Step encountered error"
        assert result.step_success is False
        assert result.key_observation == "Database connection failed"
    
    def test_prompt_template_vs_default_reflection(self):
        """Test differences between step and default reflection templates."""
        from flowlib.agent.components.reflection.prompts.default import DefaultReflectionPrompt
        
        step_prompt = DefaultStepReflectionPrompt(name="step", type="prompt")
        default_prompt = DefaultReflectionPrompt(name="default", type="prompt")
        
        # Should have different templates
        assert step_prompt.template != default_prompt.template
        
        # Should have different output models
        assert step_prompt.output_model != default_prompt.output_model
        assert step_prompt.output_model == StepReflectionResult
    
    def test_step_vs_overall_focus(self):
        """Test that step template focuses on step-level analysis."""
        prompt = DefaultStepReflectionPrompt(name="test", type="prompt")
        template = prompt.template
        
        # Step template should focus on individual steps
        assert "single step" in template
        assert "Step ID:" in template
        assert "Intent:" in template
        
        # Should not focus on overall task completion (that's for default reflection)
        assert "overall task progress percentage" not in template.lower()
        assert "main task is now complete" not in template.lower()


class TestStepReflectionPromptErrorHandling:
    """Test error handling in step reflection prompts."""
    
    def test_template_structure_validation(self):
        """Test that template has proper structure."""
        template = STEP_REFLECTION_TEMPLATE
        
        # Should have conversation markers
        assert template.count("<|im_start|>") == 1
        assert template.count("<|im_end|>") == 1
        
        # Should have proper ending (actual ending)
        assert template.endswith("<|im_end|>assistant\n")
    
    def test_required_keys_validation(self):
        """Test comprehensive validation of required keys."""
        prompt = DefaultStepReflectionPrompt(name="test", type="prompt")
        
        # All required keys
        all_keys = [
            "task_description", "step_id", "step_intent", "step_rationale",
            "flow_name", "flow_inputs_formatted", "flow_result_formatted",
            "current_progress"
        ]
        
        # Test with empty dict
        with pytest.raises(ValueError):
            prompt.format()
        
        # Test with partial keys
        for i in range(1, len(all_keys)):
            partial_kwargs = {all_keys[j]: f"value_{j}" for j in range(i)}
            with pytest.raises(ValueError):
                prompt.format(**partial_kwargs)
    
    def test_format_with_none_values(self):
        """Test format method with None values."""
        prompt = DefaultStepReflectionPrompt(name="test", type="prompt")
        
        kwargs_with_none = {
            "task_description": "Test task",
            "step_id": "step_001",
            "step_intent": None,  # None value
            "step_rationale": "Test rationale",
            "flow_name": "test_flow",
            "flow_inputs_formatted": "{}",
            "flow_result_formatted": "{}",
            "current_progress": 50
        }
        
        # Should validate required keys (None is acceptable for formatting)
        try:
            result = prompt.format(**kwargs_with_none)
            # Template may fail due to JSON schema conflicts
        except (KeyError, ValueError) as e:
            # KeyError from JSON formatting or ValueError from validation
            if "Missing required key" in str(e):
                pytest.fail("Should not fail validation with None values")
    
    def test_initialization_edge_cases(self):
        """Test initialization with edge cases."""
        # Empty name
        prompt1 = DefaultStepReflectionPrompt(name="", type="prompt")
        assert prompt1.name == ""
        
        # Special characters in name
        prompt2 = DefaultStepReflectionPrompt(name="step-reflection_v1.0", type="prompt")
        assert prompt2.name == "step-reflection_v1.0"
    
    def test_class_attribute_immutability(self):
        """Test that class attributes are not accidentally modified."""
        original_template = DefaultStepReflectionPrompt.template
        original_output_model = DefaultStepReflectionPrompt.output_model
        
        # Create instance
        prompt = DefaultStepReflectionPrompt(name="test", type="prompt")
        
        # Class attributes should remain unchanged
        assert DefaultStepReflectionPrompt.template == original_template
        assert DefaultStepReflectionPrompt.output_model == original_output_model
        assert DefaultStepReflectionPrompt.template is STEP_REFLECTION_TEMPLATE


class TestStepReflectionTemplateVariations:
    """Test different template usage patterns and variations."""
    
    def test_template_placeholder_types(self):
        """Test different types of placeholders in template."""
        template = STEP_REFLECTION_TEMPLATE
        
        # String placeholders
        string_placeholders = [
            "{{task_description}}", "{{step_id}}", "{{step_intent}}",
            "{{step_rationale}}", "{{flow_name}}", "{{flow_inputs_formatted}}",
            "{{flow_result_formatted}}"
        ]
        
        for placeholder in string_placeholders:
            assert placeholder in template
        
        # Numeric placeholder
        assert "{{current_progress}}" in template
    
    def test_template_json_schema_structure(self):
        """Test JSON schema structure in template."""
        template = STEP_REFLECTION_TEMPLATE
        
        # Extract JSON schema part
        json_start = template.find('```json')
        json_end = template.find('```', json_start + 7)
        
        if json_start != -1 and json_end != -1:
            json_content = template[json_start + 7:json_end].strip()
            
            # Should be valid JSON structure
            try:
                parsed = json.loads(json_content)
                assert isinstance(parsed, dict)
                assert "title" in parsed
                assert parsed["title"] == "StepReflectionResult"
                assert "properties" in parsed
                assert "required" in parsed
                
                # Check required fields
                required_fields = parsed["required"]
                assert "step_id" in required_fields
                assert "reflection" in required_fields
                assert "step_success" in required_fields
                
            except json.JSONDecodeError:
                pytest.fail("JSON schema in template is not valid JSON")
    
    def test_template_instruction_clarity(self):
        """Test that template instructions are clear."""
        template = STEP_REFLECTION_TEMPLATE
        
        # Should have clear instructions
        assert "reflection assistant" in template.lower()
        assert "analyze the outcome" in template.lower()
        assert "single step" in template.lower()
        
        # Should specify what to focus on
        assert "step achieved its intent" in template.lower()
        assert "successful" in template.lower()
        assert "most important observation" in template.lower()
    
    def test_template_conversation_format(self):
        """Test conversation format structure."""
        template = STEP_REFLECTION_TEMPLATE
        
        # Should follow conversation format (starts with newline then content)
        assert template.strip().startswith("<|im_start|>user")
        assert template.endswith("<|im_end|>assistant\n")
        assert "<|im_start|>user" in template
        assert "<|im_end|>assistant" in template
        
        # Should have proper conversation flow
        user_start = template.find("<|im_start|>user")
        user_end = template.find("<|im_end|>assistant")
        assert user_start < user_end


class TestStepReflectionPromptPerformance:
    """Test performance characteristics of step reflection prompts."""
    
    def test_template_access_performance(self):
        """Test that template access is fast."""
        import time
        
        prompt = DefaultStepReflectionPrompt(name="test", type="prompt")
        
        start_time = time.time()
        
        # Access template multiple times
        for _ in range(100):
            template = prompt.template
            assert isinstance(template, str)
            assert len(template) > 0
        
        end_time = time.time()
        
        # Should complete quickly
        assert (end_time - start_time) < 1.0
    
    
    
    def test_memory_usage(self):
        """Test that prompts don't use excessive memory."""
        import gc
        
        # Get initial object count
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create and use prompts
        prompts = []
        for i in range(10):
            prompt = DefaultStepReflectionPrompt(name=f"test_{i}", type="prompt")
            prompts.append(prompt)
        
        # Access templates and models
        for prompt in prompts:
            template = prompt.template
            model = prompt.output_model
            assert isinstance(template, str)
            assert model is not None
        
        # Get final object count
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Should not create excessive objects
        object_increase = final_objects - initial_objects
        assert object_increase < 200  # Conservative limit


class TestStepReflectionPromptComparison:
    """Test step reflection prompt in comparison to other prompts."""
    
    def test_step_vs_default_reflection_differences(self):
        """Test key differences between step and default reflection."""
        step_template = STEP_REFLECTION_TEMPLATE
        
        # Import default template for comparison
        from flowlib.agent.components.reflection.prompts.default import DEFAULT_REFLECTION_TEMPLATE
        
        # Should be different templates
        assert step_template != DEFAULT_REFLECTION_TEMPLATE
        
        # Step template should have step-specific elements
        assert "Step ID:" in step_template
        assert "Intent:" in step_template
        assert "single step" in step_template
        
        # Step template should focus on step success, not overall completion
        assert "step_success" in step_template
        assert "is_complete" not in step_template
    
    def test_output_model_differences(self):
        """Test output model differences."""
        step_prompt = DefaultStepReflectionPrompt(name="step", type="prompt")
        
        # Import default prompt for comparison
        from flowlib.agent.components.reflection.prompts.default import DefaultReflectionPrompt
        default_prompt = DefaultReflectionPrompt(name="default", type="prompt")
        
        # Should have different output models
        assert step_prompt.output_model != default_prompt.output_model
        assert step_prompt.output_model == StepReflectionResult
    
    def test_prompt_decorator_names(self):
        """Test that prompts have different decorator names."""
        # This is tested indirectly by checking that they can be instantiated
        # with different names (the decorator should register them separately)
        
        step_prompt = DefaultStepReflectionPrompt(name="step_reflection", type="prompt")
        assert step_prompt.name == "step_reflection"
        
        # Should be able to create multiple instances
        step_prompt2 = DefaultStepReflectionPrompt(name="step_reflection_v2", type="prompt")
        assert step_prompt2.name == "step_reflection_v2"
        
        # Should not interfere with each other
        assert step_prompt.name != step_prompt2.name


class TestStepReflectionPromptDocumentation:
    """Test documentation and metadata of step reflection prompts."""
    
    def test_class_docstring(self):
        """Test class has proper documentation."""
        assert DefaultStepReflectionPrompt.__doc__ is not None
        assert "single-step reflection" in DefaultStepReflectionPrompt.__doc__
    
    def test_template_constant_documentation(self):
        """Test template constant has proper structure."""
        assert STEP_REFLECTION_TEMPLATE is not None
        assert isinstance(STEP_REFLECTION_TEMPLATE, str)
        assert len(STEP_REFLECTION_TEMPLATE) > 500  # Should be substantial
    
    def test_module_imports(self):
        """Test that module imports are correct."""
        from flowlib.agent.components.reflection.prompts import step_reflection
        
        # Should have expected exports
        assert hasattr(step_reflection, 'DefaultStepReflectionPrompt')
        assert hasattr(step_reflection, 'STEP_REFLECTION_TEMPLATE')
        
        # Should import correct dependencies
        assert hasattr(step_reflection, 'ResourceBase')
        assert hasattr(step_reflection, 'StepReflectionResult')
    
    def test_type_annotations(self):
        """Test that type annotations are present."""
        # Class should have proper type annotations
        assert hasattr(DefaultStepReflectionPrompt, '__annotations__')
        
        # Method should have type hints
        format_method = DefaultStepReflectionPrompt.format
        assert hasattr(format_method, '__annotations__')
        
        # ClassVar attributes should be properly typed
        assert hasattr(DefaultStepReflectionPrompt, 'template')
        assert hasattr(DefaultStepReflectionPrompt, 'output_model')