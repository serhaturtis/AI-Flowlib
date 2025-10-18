"""Examples demonstrating the Flow system with enforced rules.

This module shows how to create and use flows following the new rules:
1. Each flow must have exactly one pipeline method
2. Only execute() can be called from outside the flow
"""

import asyncio
from typing import Any, Dict

from pydantic import ConfigDict

from flowlib.core.context.context import Context
from flowlib.core.models import StrictBaseModel
from flowlib.flows.base.base import Flow
from flowlib.flows.decorators import flow, pipeline


class TextInput(StrictBaseModel):
    """Input model for text processing."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    options: Dict[str, Any] = {}

class TextOutput(StrictBaseModel):
    """Output model for text processing."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    result: str
    metadata: Dict[str, Any] = {}

@flow(description="A flow with input and output type validation")
class ATypedFlow:
    """A flow with input and output type validation."""

    async def process(self, data: TextInput) -> Dict[str, Any]:
        """Process the input data."""
        return {
            "processed": data.text.upper(),
            "options_used": data.options
        }

    async def format_output(self, data: Dict[str, Any]) -> TextOutput:
        """Format the output data."""
        return TextOutput(
            result=data["processed"],
            metadata={"options": data["options_used"]}
        )

    @pipeline(input_model=TextInput, output_model=TextOutput)
    async def execute_pipeline(self, data: TextInput) -> TextOutput:
        """Execute the pipeline with type checking."""
        processed = await self.process(data)
        return await self.format_output(processed)

@flow(description="Another flow with input and output type validation")
class AnotherTypedFlow:
    """Another flow with input and output type validation."""

    async def process(self, data: TextInput) -> Dict[str, Any]:
        """Process the input data."""
        return {
            "processed": data.text.lower(),
            "options_used": data.options
        }

    async def format_output(self, data: Dict[str, Any]) -> TextOutput:
        """Format the output data."""
        return TextOutput(
            result=data["processed"],
            metadata={"options": data["options_used"]}
        )

    @pipeline(input_model=TextInput, output_model=TextOutput)
    async def execute_pipeline(self, data: TextInput) -> TextOutput:
        """Execute the pipeline with type checking."""
        processed = await self.process(data)
        return await self.format_output(processed)


# Example 3: Combined Flow (using subflow composition)
@flow(description="A flow that combines multiple flows")  # type: ignore[arg-type]
class CombinedFlow(Flow):
    """A flow that combines multiple flows using subflow composition."""

    def __init__(self) -> None:
        self.simple_flow = ATypedFlow("simple_flow")
        self.typed_flow = AnotherTypedFlow("typed_flow")

    @pipeline
    async def run_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all flows in sequence."""
        # We can only call execute() on the flow instances
        simple_result = await self.simple_flow.execute(Context(data=data))

        # Prepare input for the typed flow
        typed_input = TextInput(
            text=simple_result.data["result"],
            options={"from_simple_flow": True}
        )

        # Execute the typed flow
        typed_result = await self.typed_flow.execute(Context(data=typed_input))

        # Combine results
        return {
            "simple_result": simple_result.data,
            "typed_result": typed_result.data
        }


# Example usage
async def run_examples() -> None:
    """Run the example flows."""
    # Example 1
    a_flow = ATypedFlow("example_flow_1")
    a_result = await a_flow.execute(Context(data={"text": "hello world"}))
    print(f"A Flow Result: {a_result.data}")

    # Example 2
    another_flow = AnotherTypedFlow("example_flow_2")
    an_input = TextInput(text="an input", options={"process": True})
    another_result = await another_flow.execute(Context(data=an_input))
    print(f"Another Flow Result: {another_result.data}")

    # Example 3
    combined_flow = CombinedFlow()
    combined_result = await combined_flow.execute(Context(data={"text": "combined example"}))
    print(f"Combined Flow Result: {combined_result.data}")

    # Try to access a private method directly (should fail)
    try:
        # This should raise an AttributeError
        await a_flow.process_text({"validated_text": "test"})  # type: ignore[attr-defined]
        print("ERROR: Private method was accessible!")
    except AttributeError:
        print("SUCCESS: Private method access prevented as expected")


if __name__ == "__main__":
    asyncio.run(run_examples())
