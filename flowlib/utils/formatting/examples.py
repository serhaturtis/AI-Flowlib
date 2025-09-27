"""Examples of formatting utility usage.

This module provides examples of how to use the formatting utilities
in various contexts within the flowlib ecosystem.
"""


# Import all the formatting utilities
from flowlib.utils.formatting import (
    # Text formatting
    process_escape_sequences,
    
    # Entity formatting
    format_conversation,
    format_state,
    format_history,
    extract_json
)


def example_text_formatting() -> None:
    """Example of text formatting usage."""
    # Process escape sequences in text
    raw_text = "This has escape sequences: \\n New line and \\t tab"
    processed = process_escape_sequences(raw_text)
    print("Original:", repr(raw_text))
    print("Processed:", repr(processed))
    

def example_conversation_formatting() -> None:
    """Example of conversation formatting usage."""
    # Create sample conversation history
    conversation = [
        {"speaker": "User", "content": "Hello, can you help me with a task?"},
        {"speaker": "Assistant", "content": "Of course! What do you need help with?"},
        {"speaker": "User", "content": "I need to create a report about climate change."}
    ]
    
    # Format conversation for prompt
    formatted = format_conversation(conversation)
    print("Formatted conversation:")
    print(formatted)
    
    # Format execution state
    state = {
        "task": "Create a report about climate change",
        "progress": 0.5,
        "last_action": "Research key statistics"
    }
    
    formatted_state = format_state(state)
    print("\nFormatted state:")
    print(formatted_state)
    
    # Format execution history
    history = [
        {
            "action": "execute_flow",
            "flow": "web-search",
            "reasoning": "Need to find the latest climate data",
            "reflection": "Found useful information about global temperature trends"
        },
        {
            "action": "execute_flow",
            "flow": "create-outline",
            "reasoning": "Need to organize the information into a report structure", 
            "reflection": "Created a coherent outline with key sections"
        }
    ]
    
    formatted_history = format_history(history)
    print("\nFormatted history:")
    print(formatted_history)


def example_json_extraction() -> None:
    """Example of JSON extraction usage."""
    # Example text with embedded JSON
    text_with_json = """
    I've analyzed the data and here are the results:
    
    {
        "temperature_increase": 1.5,
        "sea_level_rise": "0.3 meters",
        "main_causes": ["fossil fuels", "deforestation", "agriculture"]
    }
    
    Let me know if you need anything else.
    """
    
    # Extract JSON data
    json_data = extract_json(text_with_json)
    print("Extracted JSON data:")
    print(json_data)


def run_examples() -> None:
    """Run all examples."""
    print("=== TEXT FORMATTING EXAMPLES ===")
    example_text_formatting()
    
    print("\n=== CONVERSATION FORMATTING EXAMPLES ===")
    example_conversation_formatting()
    
    print("\n=== JSON EXTRACTION EXAMPLES ===")
    example_json_extraction()


if __name__ == "__main__":
    run_examples() 