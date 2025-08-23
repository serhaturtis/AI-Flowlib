"""Prompt definitions for memory recall flows."""

from typing import ClassVar
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.decorators.decorators import prompt


@prompt("context_analysis")
class ContextAnalysisPrompt(ResourceBase):
    """Prompt for analyzing context to determine optimal recall strategy."""
    
    template: ClassVar[str] = """
Analyze the following context and query to determine the best memory recall strategy.

Context: {{context}}
Query: {{query}}

Based on this information, analyze:
1. What type of information is being sought
2. Which recall strategy would be most effective
3. Key concepts that should guide the search
4. Your confidence in this analysis

Available recall strategies:
- CONTEXTUAL: For queries related to current conversation or situation
- ENTITY: For queries about specific people, places, or things
- TEMPORAL: For queries about events, sequences, or time-based information  
- SEMANTIC: For queries about concepts, meanings, or related ideas

Return your analysis as a JSON object with this structure:
{
    "analysis": "detailed analysis of the context and query",
    "recommended_strategy": "one of: CONTEXTUAL, ENTITY, TEMPORAL, SEMANTIC",
    "key_concepts": ["list", "of", "key", "concepts"],
    "confidence": 0.85
}
"""
    
    config: ClassVar[dict] = {
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.95
    }