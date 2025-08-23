"""
Models for agent intelligence component.

This module contains Pydantic models used by the agent intelligence system,
including knowledge, learning, and memory intelligence.
"""

from typing import Dict, Any, Optional, List, Set, Union
from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel
from enum import Enum
from datetime import datetime


class IntelligenceType(str, Enum):
    """Types of intelligence components."""
    KNOWLEDGE = "knowledge"
    LEARNING = "learning"
    MEMORY = "memory"


class LearningStrategy(str, Enum):
    """Learning strategies for intelligence."""
    PATTERN_RECOGNITION = "pattern_recognition"
    REINFORCEMENT = "reinforcement"
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    TRANSFER = "transfer"


class IntelligenceConfig(StrictBaseModel):
    """Configuration for intelligence components."""
    model_config = ConfigDict(extra="forbid")
    
    enabled_types: Set[IntelligenceType] = Field(default={IntelligenceType.KNOWLEDGE}, description="Enabled intelligence types")
    learning_strategy: LearningStrategy = Field(default=LearningStrategy.PATTERN_RECOGNITION, description="Default learning strategy")
    knowledge_threshold: float = Field(default=0.7, description="Threshold for knowledge confidence")
    learning_rate: float = Field(default=0.1, description="Learning rate for updates")


class KnowledgeItem(StrictBaseModel):
    """A piece of knowledge in the intelligence system."""
    model_config = ConfigDict(extra="forbid")
    
    knowledge_id: str = Field(description="Unique knowledge identifier")
    content: str = Field(description="Knowledge content")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this knowledge")
    source: str = Field(description="Source of the knowledge")
    tags: Set[str] = Field(default_factory=set, description="Knowledge tags")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")


class LearningEvent(StrictBaseModel):
    """An event in the learning process."""
    model_config = ConfigDict(extra="forbid")
    
    event_id: str = Field(description="Unique event identifier")
    event_type: str = Field(description="Type of learning event")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for learning")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="Output/result data")
    learning_outcome: str = Field(description="What was learned")
    confidence_change: float = Field(default=0.0, description="Change in confidence")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")


class PatternRecognitionResult(StrictBaseModel):
    """Result of pattern recognition analysis."""
    model_config = ConfigDict(extra="forbid")
    
    patterns_found: List[str] = Field(default_factory=list, description="Identified patterns")
    pattern_confidence: Dict[str, float] = Field(default_factory=dict, description="Confidence for each pattern")
    anomalies: List[str] = Field(default_factory=list, description="Detected anomalies")
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")


class IntelligenceMetrics(StrictBaseModel):
    """Metrics for intelligence system performance."""
    model_config = ConfigDict(extra="forbid")
    
    knowledge_items_count: int = Field(default=0, description="Total knowledge items")
    learning_events_count: int = Field(default=0, description="Total learning events")
    average_confidence: float = Field(default=0.0, description="Average knowledge confidence")
    patterns_recognized: int = Field(default=0, description="Total patterns recognized")
    last_learning_event: Optional[datetime] = Field(default=None, description="Last learning event timestamp")


class IntelligenceRequest(StrictBaseModel):
    """Request for intelligence operations."""
    model_config = ConfigDict(extra="forbid")
    
    operation_type: str = Field(description="Type of intelligence operation")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")
    context: Optional[str] = Field(default=None, description="Operation context")


class IntelligenceResult(StrictBaseModel):
    """Result of intelligence operations."""
    model_config = ConfigDict(extra="forbid")
    
    operation_type: str = Field(description="Type of operation performed")
    success: bool = Field(description="Whether operation was successful")
    result_data: Dict[str, Any] = Field(default_factory=dict, description="Operation result data")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in result")
    processing_time: float = Field(description="Time taken for operation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")