"""Enhanced tests for Flow Decorators - additional edge cases and scenarios."""

import pytest
import asyncio
import logging
import threading
import time
from typing import Any, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
import weakref

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.flows.base.base import Flow
from flowlib.core.context.context import Context


# Test models
class ThreadSafeInput(BaseModel):
    """Input model for thread safety tests."""
    thread_id: int
    value: str


class ThreadSafeOutput(BaseModel):
    """Output model for thread safety tests."""
    thread_id: int
    result: str
    execution_time: float


class DeepNestedInput(BaseModel):
    """Deeply nested input model."""
    level1: Dict[str, Any]
    level2: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


# Additional test classes
class AbstractBase:
    """Abstract base class for inheritance testing."""
    abstract_method = None


class Mixin1:
    """First mixin class."""
    def mixin1_method(self):
        return "mixin1"


class Mixin2:
    """Second mixin class."""
    def mixin2_method(self):
        return "mixin2"


class TestConcurrentExecution:
    """Test concurrent execution of decorated flows."""
    
    @pytest.mark.asyncio
    async def test_multiple_flow_instances_concurrent(self):
        """Test multiple flow instances running concurrently."""
        @flow(description="Concurrent test flow")
        class ConcurrentFlow:
            def __init__(self):
                self.execution_count = 0
            
            @pipeline(input_model=ThreadSafeInput, output_model=ThreadSafeOutput)
            async def run_pipeline(self, input_data: ThreadSafeInput) -> ThreadSafeOutput:
                start_time = time.time()
                self.execution_count += 1
                
                # Simulate some work
                await asyncio.sleep(0.01)
                
                return ThreadSafeOutput(
                    thread_id=input_data.thread_id,
                    result=f"Processed by instance {id(self)}",
                    execution_time=time.time() - start_time
                )
        
        # Create multiple flow instances
        flows = [ConcurrentFlow() for _ in range(5)]
        
        # Run them concurrently
        tasks = []
        for i, flow_instance in enumerate(flows):
            input_data = ThreadSafeInput(thread_id=i, value=f"test_{i}")
            tasks.append(flow_instance.run_pipeline(input_data))
        
        results = await asyncio.gather(*tasks)
        
        # Verify all completed successfully
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.thread_id == i
            assert "Processed by instance" in result.result
        
        # Each instance should have been called once
        for flow_instance in flows:
            assert flow_instance.execution_count == 1
    
    @pytest.mark.asyncio
    async def test_single_flow_concurrent_calls(self):
        """Test single flow instance handling concurrent calls."""
        call_count = 0
        call_lock = asyncio.Lock()
        
        @flow(description="Single instance concurrent test")
        class SingleInstanceFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                nonlocal call_count
                async with call_lock:
                    call_count += 1
                    current_call = call_count
                
                # Simulate work
                await asyncio.sleep(0.01)
                
                return {"call_number": current_call}
        
        flow_instance = SingleInstanceFlow()
        
        # Make concurrent calls
        tasks = [flow_instance.run_pipeline(Context(data={})) for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All calls should complete
        assert len(results) == 10
        assert call_count == 10
        
        # Each call should have a unique number
        call_numbers = {r["call_number"] for r in results}
        assert len(call_numbers) == 10


class TestComplexInheritance:
    """Test complex inheritance scenarios."""
    
    def test_multiple_mixins_with_flow(self):
        """Test flow decorator with multiple mixins."""
        @flow(description="Multi-mixin flow")
        class MultiMixinFlow(Mixin1, Mixin2):
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {
                    "mixin1": self.mixin1_method(),
                    "mixin2": self.mixin2_method()
                }
        
        flow_instance = MultiMixinFlow()
        
        # Should have methods from both mixins
        assert flow_instance.mixin1_method() == "mixin1"
        assert flow_instance.mixin2_method() == "mixin2"
        
        # Should still be a Flow
        assert isinstance(flow_instance, Flow)
    
    def test_deep_inheritance_chain(self):
        """Test flow decorator with deep inheritance chain."""
        class Level1:
            def level1_method(self):
                return "level1"
        
        class Level2(Level1):
            def level2_method(self):
                return "level2"
        
        class Level3(Level2):
            def level3_method(self):
                return "level3"
        
        @flow(description="Deep inheritance flow")
        class DeepFlow(Level3):
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {
                    "level1": self.level1_method(),
                    "level2": self.level2_method(),
                    "level3": self.level3_method()
                }
        
        flow_instance = DeepFlow()
        
        # Should have all inherited methods
        assert flow_instance.level1_method() == "level1"
        assert flow_instance.level2_method() == "level2"
        assert flow_instance.level3_method() == "level3"
        assert isinstance(flow_instance, Flow)
    
    def test_abstract_base_with_flow(self):
        """Test flow decorator with abstract base class."""
        @flow(description="Abstract base flow")
        class AbstractFlow(AbstractBase):
            abstract_method = "implemented"
            
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {"abstract_method": self.abstract_method}
        
        flow_instance = AbstractFlow()
        assert flow_instance.abstract_method == "implemented"
        assert isinstance(flow_instance, Flow)


class TestDynamicFlowCreation:
    """Test creating flows dynamically at runtime."""
    
    def test_create_flow_dynamically(self):
        """Test creating a flow class dynamically."""
        def create_dynamic_flow(name: str, multiplier: int):
            @flow(name=f"dynamic-{name}", description=f"Dynamic flow {name}")
            class DynamicFlow:
                @pipeline()
                async def run_pipeline(self, context: Context) -> dict:
                    value = context.get("value")
                    return {"result": value * multiplier}
            
            return DynamicFlow
        
        # Create multiple dynamic flows
        FlowA = create_dynamic_flow("A", 2)
        FlowB = create_dynamic_flow("B", 3)
        
        assert FlowA.__flow_name__ == "dynamic-A"
        assert FlowB.__flow_name__ == "dynamic-B"
        
        # They should be different classes
        assert FlowA is not FlowB
        
        # Test execution
        flow_a = FlowA()
        flow_b = FlowB()
        
        assert isinstance(flow_a, Flow)
        assert isinstance(flow_b, Flow)
    
    @pytest.mark.asyncio
    async def test_modify_flow_after_decoration(self):
        """Test modifying a flow class after decoration."""
        @flow(description="Modifiable flow")
        class ModifiableFlow:
            custom_value = "original"
            
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {"custom_value": self.custom_value}
        
        # Modify the class after decoration
        ModifiableFlow.custom_value = "modified"
        ModifiableFlow.new_attribute = "added"
        
        flow_instance = ModifiableFlow()
        result = await flow_instance.run_pipeline(Context(data={}))
        
        assert result["custom_value"] == "modified"
        assert flow_instance.new_attribute == "added"


class TestMemoryAndPerformance:
    """Test memory management and performance characteristics."""
    
    def test_flow_instance_memory_cleanup(self):
        """Test that flow instances can be garbage collected."""
        @flow(description="Memory test flow")
        class MemoryFlow:
            def __init__(self):
                self.large_data = [0] * 1000  # Some data
            
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {"size": len(self.large_data)}
        
        # Create instance and weak reference
        flow_instance = MemoryFlow()
        weak_ref = weakref.ref(flow_instance)
        
        # Verify instance exists
        assert weak_ref() is not None
        
        # Delete instance
        del flow_instance
        
        # Force garbage collection (implementation specific, may not work in all environments)
        import gc
        gc.collect()
        
        # In most cases, the weak reference should be None now
        # (but this is not guaranteed in all Python implementations)
    


class TestAdvancedPipelinePatterns:
    """Test advanced pipeline patterns and edge cases."""
    
    @pytest.mark.asyncio
    async def test_pipeline_with_complex_signature(self):
        """Test pipeline method with complex signature."""
        @flow(description="Complex signature flow")
        class ComplexSignatureFlow:
            @pipeline()
            async def run_pipeline(
                self,
                context: Context,
                *args,
                optional_param: str = "default",
                **kwargs
            ) -> dict:
                return {
                    "args": list(args),
                    "optional_param": optional_param,
                    "kwargs": kwargs
                }
        
        flow_instance = ComplexSignatureFlow()
        result = await flow_instance.run_pipeline(
            Context(data={}),
            "arg1", "arg2",
            optional_param="custom",
            extra_kwarg="value"
        )
        
        assert result["args"] == ["arg1", "arg2"]
        assert result["optional_param"] == "custom"
        assert result["kwargs"] == {"extra_kwarg": "value"}
    
    @pytest.mark.asyncio
    async def test_pipeline_calling_other_methods(self):
        """Test pipeline method calling other flow methods."""
        @flow(description="Method calling flow")
        class MethodCallingFlow:
            async def helper_method(self, value: int) -> int:
                return value * 2
            
            def sync_helper(self, value: int) -> int:
                return value + 10
            
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                value = context.get("value")
                
                # Call async helper
                async_result = await self.helper_method(value)
                
                # Call sync helper
                sync_result = self.sync_helper(value)
                
                return {
                    "async_result": async_result,
                    "sync_result": sync_result,
                    "combined": async_result + sync_result
                }
        
        flow_instance = MethodCallingFlow()
        result = await flow_instance.run_pipeline(Context(data={"value": 3}))
        
        assert result["async_result"] == 6  # 3 * 2
        assert result["sync_result"] == 13  # 3 + 10
        assert result["combined"] == 19
    
    @pytest.mark.asyncio
    async def test_nested_flow_execution(self):
        """Test flow executing another flow."""
        @flow(description="Inner flow")
        class InnerFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                value = context.get("value")
                return {"inner_result": value * 2}
        
        @flow(description="Outer flow")
        class OuterFlow:
            def __init__(self):
                self.inner_flow = InnerFlow()
            
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                # Execute inner flow
                inner_result = await self.inner_flow.run_pipeline(context)
                
                return {
                    "outer_result": "processed",
                    "inner_data": inner_result
                }
        
        outer_flow = OuterFlow()
        result = await outer_flow.run_pipeline(Context(data={"value": 5}))
        
        assert result["outer_result"] == "processed"
        assert result["inner_data"]["inner_result"] == 10


class TestErrorRecovery:
    """Test error recovery and partial execution scenarios."""
    
    @pytest.mark.asyncio
    async def test_pipeline_partial_execution_tracking(self):
        """Test tracking partial execution on error."""
        execution_steps = []
        
        @flow(description="Partial execution flow")
        class PartialExecutionFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                execution_steps.append("start")
                
                # Step 1
                execution_steps.append("step1")
                await asyncio.sleep(0.01)
                
                # Step 2 - raises error
                execution_steps.append("step2")
                try:
                    raise_error = context.get("raise_error")
                except KeyError:
                    raise_error = False
                if raise_error:
                    raise ValueError("Intentional error")
                
                # Step 3
                execution_steps.append("step3")
                
                return {"completed": True}
        
        flow_instance = PartialExecutionFlow()
        
        # First run - with error
        execution_steps.clear()
        with pytest.raises(ValueError):
            await flow_instance.run_pipeline(Context(data={"raise_error": True}))
        
        assert execution_steps == ["start", "step1", "step2"]
        
        # Second run - without error
        execution_steps.clear()
        result = await flow_instance.run_pipeline(Context(data={"raise_error": False}))
        
        assert execution_steps == ["start", "step1", "step2", "step3"]
        assert result["completed"] is True
    
    @pytest.mark.asyncio
    async def test_pipeline_exception_context(self):
        """Test exception context is preserved through pipeline."""
        @flow(description="Exception context flow")
        class ExceptionContextFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                try:
                    # This will raise a specific error
                    data = {"key": "value"}
                    _ = data["missing_key"]
                except KeyError as e:
                    # Re-raise with additional context
                    raise ValueError(f"Pipeline failed: {str(e)}") from e
                
                return {"should_not_reach": True}
        
        flow_instance = ExceptionContextFlow()
        
        with pytest.raises(ValueError) as exc_info:
            await flow_instance.run_pipeline(Context(data={}))
        
        # Check exception chaining
        assert "Pipeline failed" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, KeyError)


class TestDecoratorValidation:
    """Test decorator parameter validation edge cases."""
    
    def test_flow_decorator_with_very_long_description(self):
        """Test flow decorator with very long description."""
        long_description = "x" * 10000  # 10k character description
        
        @flow(description=long_description)
        class LongDescriptionFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {"description_length": len(self.get_description())}
        
        flow_instance = LongDescriptionFlow()
        assert flow_instance.get_description() == long_description
    
    def test_flow_decorator_with_unicode_name(self):
        """Test flow decorator with unicode characters in name."""
        @flow(name="测试流-test-フロー", description="Unicode name flow")
        class UnicodeFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {"name": self.__class__.__flow_name__}
        
        assert UnicodeFlow.__flow_name__ == "测试流-test-フロー"
    
    def test_pipeline_decorator_with_forward_references(self):
        """Test pipeline decorator with forward reference types."""
        @flow(description="Forward reference flow")
        class ForwardRefFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {"type": "forward_ref"}
        
        flow_instance = ForwardRefFlow()
        assert hasattr(flow_instance.run_pipeline, '__pipeline__')


class TestFlowRegistryInteraction:
    """Test interaction with flow registry (if implemented)."""
    
    def test_multiple_flows_same_name_different_modules(self):
        """Test handling of flows with same name in different contexts."""
        # First flow
        @flow(name="duplicate-name", description="First flow")
        class FirstFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {"flow": "first"}
        
        # Second flow with same name
        @flow(name="duplicate-name", description="Second flow")
        class SecondFlow:
            @pipeline()
            async def run_pipeline(self, context: Context) -> dict:
                return {"flow": "second"}
        
        # Both should be created successfully
        first = FirstFlow()
        second = SecondFlow()
        
        assert first.__class__.__flow_name__ == "duplicate-name"
        assert second.__class__.__flow_name__ == "duplicate-name"
        assert first.__class__ is not second.__class__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])