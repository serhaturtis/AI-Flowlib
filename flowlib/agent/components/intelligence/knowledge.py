"""Clean Knowledge Models - Pure Domain Objects.

This module defines clean, focused domain models for knowledge representation
without any configuration pollution. All models are simple dataclasses that
focus solely on representing knowledge, not infrastructure concerns.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class ConfidenceLevel(str, Enum):
    """Standard confidence levels for knowledge items."""
    LOW = "low"          # 0.0 - 0.4
    MEDIUM = "medium"    # 0.4 - 0.7  
    HIGH = "high"        # 0.7 - 0.9
    VERY_HIGH = "very_high"  # 0.9 - 1.0


@dataclass
class Entity:
    """Pure entity representation - no configuration pollution.
    
    Represents a distinct object, person, place, or concept identified in content.
    """
    name: str
    type: str  # "person", "organization", "location", "concept", etc.
    description: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = ""
    aliases: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate entity data after creation."""
        if not self.name.strip():
            raise ValueError("Entity name cannot be empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get human-readable confidence level."""
        if self.confidence < 0.4:
            return ConfidenceLevel.LOW
        elif self.confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif self.confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH


@dataclass  
class Concept:
    """Pure concept representation.
    
    Represents an abstract idea, theme, or conceptual understanding extracted from content.
    """
    name: str
    description: str
    category: Optional[str] = None  # "technical", "business", "scientific", etc.
    examples: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    confidence: float = 1.0
    abstraction_level: int = 1  # 1=concrete, 5=abstract
    
    def __post_init__(self):
        """Validate concept data after creation."""
        if not self.name.strip():
            raise ValueError("Concept name cannot be empty")
        if not self.description.strip():
            raise ValueError("Concept description cannot be empty")
        if not (1 <= self.abstraction_level <= 5):
            raise ValueError("Abstraction level must be between 1 and 5")
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get human-readable confidence level."""
        if self.confidence < 0.4:
            return ConfidenceLevel.LOW
        elif self.confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif self.confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH


@dataclass
class Relationship:
    """Pure relationship representation.
    
    Represents a connection or association between entities or concepts.
    """
    source: str  # Source entity/concept name
    target: str  # Target entity/concept name
    type: str    # "is_a", "part_of", "related_to", "causes", "enables", etc.
    description: Optional[str] = None
    confidence: float = 1.0
    bidirectional: bool = False
    strength: str = "medium"  # "weak", "medium", "strong"
    
    def __post_init__(self):
        """Validate relationship data after creation."""
        if not self.source.strip():
            raise ValueError("Relationship source cannot be empty")
        if not self.target.strip():
            raise ValueError("Relationship target cannot be empty")
        if not self.type.strip():
            raise ValueError("Relationship type cannot be empty")
        if self.strength not in ["weak", "medium", "strong"]:
            raise ValueError("Strength must be 'weak', 'medium', or 'strong'")
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get human-readable confidence level."""
        if self.confidence < 0.4:
            return ConfidenceLevel.LOW
        elif self.confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif self.confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH


@dataclass
class Pattern:
    """Pure pattern representation.
    
    Represents a recurring structure, sequence, or template identified in content.
    """
    name: str
    description: str
    pattern_type: str = "general"  # "sequence", "structure", "template", "behavior", etc.
    examples: List[str] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)  # Variable parts of the pattern
    confidence: float = 1.0
    frequency: str = "unknown"  # "rare", "occasional", "common", "frequent"
    
    def __post_init__(self):
        """Validate pattern data after creation."""
        if not self.name.strip():
            raise ValueError("Pattern name cannot be empty")
        if not self.description.strip():
            raise ValueError("Pattern description cannot be empty")
        if self.frequency not in ["unknown", "rare", "occasional", "common", "frequent"]:
            raise ValueError("Frequency must be one of: unknown, rare, occasional, common, frequent")
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get human-readable confidence level."""
        if self.confidence < 0.4:
            return ConfidenceLevel.LOW
        elif self.confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif self.confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH


@dataclass
class KnowledgeSet:
    """Complete knowledge extracted from content.
    
    Represents all knowledge extracted from a piece of content in a single operation.
    This replaces the complex multi-flow orchestration with simple, unified extraction.
    """
    entities: List[Entity] = field(default_factory=list)
    concepts: List[Concept] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    patterns: List[Pattern] = field(default_factory=list)
    summary: str = ""
    confidence: float = 1.0
    extracted_at: datetime = field(default_factory=datetime.now)
    source_content: str = ""
    processing_notes: List[str] = field(default_factory=list)
    
    @property
    def total_items(self) -> int:
        """Get total number of knowledge items extracted."""
        return len(self.entities) + len(self.concepts) + len(self.relationships) + len(self.patterns)
    
    @property 
    def confidence_level(self) -> ConfidenceLevel:
        """Get overall confidence level for the knowledge set."""
        if self.confidence < 0.4:
            return ConfidenceLevel.LOW
        elif self.confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif self.confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the extracted knowledge."""
        return {
            "total_items": self.total_items,
            "entities_count": len(self.entities),
            "concepts_count": len(self.concepts),
            "relationships_count": len(self.relationships),
            "patterns_count": len(self.patterns),
            "confidence_level": self.confidence_level.value,
            "average_confidence": self._calculate_average_confidence(),
            "extraction_time": self.extracted_at.isoformat()
        }
    
    def _calculate_average_confidence(self) -> float:
        """Calculate average confidence across all knowledge items."""
        all_items = self.entities + self.concepts + self.relationships + self.patterns
        if not all_items:
            return self.confidence
        
        total_confidence = sum(item.confidence for item in all_items)
        return total_confidence / len(all_items)


@dataclass
class ContentAnalysis:
    """Analysis of content structure and characteristics.
    
    Provides intelligent analysis of content to guide knowledge extraction
    without complex strategy determination logic.
    """
    content_type: str  # "technical", "narrative", "structured", "conversational", etc.
    key_topics: List[str] = field(default_factory=list)
    complexity_level: str = "medium"  # "simple", "medium", "complex"
    language: str = "en"
    length_category: str = "medium"  # "short", "medium", "long"
    structure_type: str = "unstructured"  # "structured", "semi-structured", "unstructured"
    suggested_focus: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate analysis data after creation."""
        if self.complexity_level not in ["simple", "medium", "complex"]:
            raise ValueError("Complexity level must be 'simple', 'medium', or 'complex'")
        if self.length_category not in ["short", "medium", "long"]:
            raise ValueError("Length category must be 'short', 'medium', or 'long'")
        if self.structure_type not in ["structured", "semi-structured", "unstructured"]:
            raise ValueError("Structure type must be 'structured', 'semi-structured', or 'unstructured'")


@dataclass
class LearningResult:
    """Result of learning operation.
    
    Simple result object that provides clear feedback about learning operations
    without complex orchestration metadata.
    """
    success: bool
    knowledge: KnowledgeSet
    processing_time_seconds: float = 0.0
    message: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def has_errors(self) -> bool:
        """Check if there were any errors during learning."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there were any warnings during learning."""
        return len(self.warnings) > 0
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the learning result."""
        if not self.success:
            return f"Learning failed: {self.message}"
        
        stats = self.knowledge.get_stats()
        return (
            f"Successfully learned {stats['total_items']} items "
            f"({stats['entities_count']} entities, {stats['concepts_count']} concepts, "
            f"{stats['relationships_count']} relationships, {stats['patterns_count']} patterns) "
            f"with {stats['confidence_level']} confidence"
        )