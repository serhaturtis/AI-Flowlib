from typing import List, ClassVar
from pydantic import Field
from flowlib.core.models import StrictBaseModel

# Import decorator
from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase

# --- Fusion Prompt and Model ---

@prompt("memory_fusion")
class MemoryFusionPrompt(ResourceBase):
    """Prompt for fusing search results from different memory types."""
    template: ClassVar[str] = """
You are a Memory Fusion Assistant. Synthesize search results from semantic, knowledge graph, working memories, and domain-specific knowledge plugins related to the user query.
Identify the most relevant pieces of information and combine them into a concise list and a brief summary. Focus on relevance to the query.
If a specific memory type returned no results (indicated by 'No relevant ... found.'), explicitly state that in your summary and do not invent information for that section.

User Query: "{{query}}"

### Semantic Search Results (Vector Memory):
{{vector_results}}

### Knowledge Graph Results:
{{knowledge_results}}

### Working Memory Results (Short-term):
{{working_results}}

### Knowledge Plugin Results (Domain-specific):
{{plugin_results}}

### Synthesized Output:
Provide:
- relevant_items: Key pieces of relevant information synthesized across all sources
- summary: Brief (1-2 sentence) summary acknowledging which sources contributed (or didn't)
"""


class FusedMemoryResult(StrictBaseModel):
    """Model for the fused result from LLM."""
    
    relevant_items: List[str] = Field(..., description="List of relevant items synthesized from different memory sources.")
    summary: str = Field(..., min_length=1, description="A brief summary of the combined findings, acknowledging sources.")


# --- KG Query Extraction Prompt and Model --- 

@prompt("kg_query_extraction")
class KGQueryExtractionPrompt(ResourceBase):
    """Prompt to extract relevant keywords/entities for KG search."""
    template: ClassVar[str] = """
You are a Knowledge Graph Query Assistant.
Analyze the user query and identify the main specific entities, concepts, or keywords that would be most relevant to search for in a knowledge graph containing structured information.
Focus on nouns or proper nouns that represent distinct items.
Avoid generic terms, questions, or conversational filler.
If no specific searchable terms are found, return an empty list.

User Query: "{{query}}"

Context (Optional): {{context}}

Relevant Keywords/Entities for KG Search:
"""

class ExtractedKGQueryTerms(StrictBaseModel):
    """Model for keywords/entities extracted for KG search."""
    
    terms: List[str] = Field(..., description="List of extracted keywords or entity names suitable for KG search.") 