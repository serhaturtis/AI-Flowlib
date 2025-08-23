"""Models for knowledge extraction and retrieval flows."""

from typing import List, Optional, Dict, Any
from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel
from enum import Enum


class KnowledgeType(str, Enum):
    """Types of knowledge that can be extracted."""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    PERSONAL = "personal"
    TECHNICAL = "technical"


class ExtractedKnowledge(StrictBaseModel):
    """Represents a piece of extracted knowledge."""
    # Inherits strict configuration from StrictBaseModel
    
    content: str = Field(..., description="The knowledge content")
    knowledge_type: KnowledgeType = Field(..., description="Type of knowledge")
    domain: str = Field(..., description="Knowledge domain (e.g., chemistry, personal)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    source_context: str = Field(..., description="Original context where knowledge was found")
    entities: List[str] = Field(default_factory=list, description="Key entities mentioned")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class KnowledgeExtractionInput(StrictBaseModel):
    """Input for knowledge extraction flow."""
    # Inherits strict configuration from StrictBaseModel
    
    text: str = Field(..., description="Text to extract knowledge from")
    context: str = Field(..., description="Context of the conversation or source")
    domain_hint: Optional[str] = Field(None, description="Suggested domain for the knowledge")
    extract_personal: bool = Field(True, description="Whether to extract personal information")


class KnowledgeExtractionOutput(StrictBaseModel):
    """Output from knowledge extraction flow."""
    # Inherits strict configuration from StrictBaseModel
    
    extracted_knowledge: List[ExtractedKnowledge] = Field(..., description="List of extracted knowledge items")
    processing_notes: str = Field(..., description="Notes about the extraction process")
    domains_detected: List[str] = Field(default_factory=list, description="Domains detected in the text")
    
    def get_user_display(self) -> str:
        """Get user-friendly display text for knowledge extraction results."""
        count = len(self.extracted_knowledge)
        domains = len(self.domains_detected)
        
        if count == 0:
            return "🔍 No knowledge extracted from the provided text."
        
        # Group by knowledge type for summary
        type_counts = {}
        for knowledge in self.extracted_knowledge:
            ktype = knowledge.knowledge_type.value
            type_counts[ktype] = type_counts[ktype] + 1 if ktype in type_counts else 1
        
        summary_parts = []
        summary_parts.append(f"🧠 Extracted {count} knowledge item{'s' if count != 1 else ''}")
        
        if domains > 0:
            domain_list = ", ".join(self.domains_detected)
            summary_parts.append(f"across {domains} domain{'s' if domains != 1 else ''}: {domain_list}")
        
        # Add type breakdown
        if type_counts:
            type_summary = ", ".join([f"{count} {ktype}" for ktype, count in type_counts.items()])
            summary_parts.append(f"\nTypes: {type_summary}")
        
        if self.processing_notes:
            summary_parts.append(f"\nNotes: {self.processing_notes}")
        
        return "".join(summary_parts)


class KnowledgeRetrievalInput(StrictBaseModel):
    """Input for knowledge retrieval flow."""
    # Inherits strict configuration from StrictBaseModel
    
    query: str = Field(..., description="Query to search for")
    domain: Optional[str] = Field(None, description="Specific domain to search in")
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results")
    include_plugins: bool = Field(True, description="Whether to search knowledge plugins")
    include_memory: bool = Field(True, description="Whether to search agent memory")


class RetrievedKnowledge(StrictBaseModel):
    """Represents retrieved knowledge from various sources."""
    # Inherits strict configuration from StrictBaseModel
    
    content: str = Field(..., description="The knowledge content")
    source: str = Field(..., description="Source of the knowledge (memory, plugin, etc.)")
    domain: str = Field(..., description="Knowledge domain")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Retrieval confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source-specific metadata")


class KnowledgeRetrievalOutput(StrictBaseModel):
    """Output from knowledge retrieval flow."""
    # Inherits strict configuration from StrictBaseModel
    
    retrieved_knowledge: List[RetrievedKnowledge] = Field(..., description="List of retrieved knowledge items")
    search_summary: str = Field(..., description="Summary of the search process and results")
    sources_searched: List[str] = Field(default_factory=list, description="Sources that were searched")
    total_results: int = Field(..., description="Total number of results found")
    
    def get_user_display(self) -> str:
        """Get user-friendly display text for knowledge retrieval results."""
        if self.total_results == 0:
            return "🔍 No knowledge found matching your query."
        
        # Group by source and domain for better summary
        source_counts = {}
        domains = set()
        
        for knowledge in self.retrieved_knowledge:
            source = knowledge.source
            source_counts[source] = source_counts[source] + 1 if source in source_counts else 1
            domains.add(knowledge.domain)
        
        summary_parts = []
        summary_parts.append(f"🎯 Found {len(self.retrieved_knowledge)} relevant knowledge items")
        
        if self.total_results > len(self.retrieved_knowledge):
            summary_parts.append(f" (showing top {len(self.retrieved_knowledge)} of {self.total_results} total)")
        
        # Add domain info
        if domains:
            domain_list = ", ".join(sorted(domains))
            summary_parts.append(f"\nDomains: {domain_list}")
        
        # Add source breakdown
        if source_counts:
            source_summary = ", ".join([f"{count} from {source}" for source, count in source_counts.items()])
            summary_parts.append(f"\nSources: {source_summary}")
        
        if self.search_summary:
            summary_parts.append(f"\n\n{self.search_summary}")
        
        return "".join(summary_parts)