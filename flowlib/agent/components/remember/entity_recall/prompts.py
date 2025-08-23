"""Prompts for entity recall flow."""

from typing import ClassVar
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.decorators.decorators import prompt


@prompt("entity_validation")
class EntityValidationPrompt(ResourceBase):
    """Prompt for validating entity existence and properties."""
    
    template: ClassVar[str] = """
Validate the following entity for recall operations:

Entity ID: {{entity_id}}
Entity Type: {{entity_type}}
Context: {{context}}

Determine if this entity is valid for recall and provide assessment:

Return your validation as a JSON object:
{
    "is_valid": true/false,
    "validation_message": "explanation of validation result",
    "suggested_alternatives": ["list", "of", "alternative", "entities"],
    "confidence": 0.95
}
"""
    
    config: ClassVar[dict] = {
        "max_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.95
    }