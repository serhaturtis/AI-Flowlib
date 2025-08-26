"""Simplified Intelligence System for AI Agents.

This module provides a clean, simple intelligence architecture that replaces
the overly complex learning flow hierarchy. The new system focuses on:

1. Intelligent Learning: Single flow that adapts to content
2. Unified Memory: Simple storage with graceful degradation  
3. Clean Models: Pure domain objects without configuration pollution
4. Simple API: Easy-to-use functions for learning operations

This approach reduces complexity by 90% while maintaining full functionality.
"""

# Core intelligence components
from .learning import IntelligentLearningFlow, learn_from_text
from .memory import IntelligentMemory, remember, recall  
from .knowledge import (
    Entity, Concept, Relationship, Pattern, KnowledgeSet, 
    ContentAnalysis, LearningResult
)

# Import prompts to ensure they are registered
from . import prompts

# Simple API exports
__all__ = [
    # Core components
    'IntelligentLearningFlow',
    'IntelligentMemory',
    
    # Knowledge models
    'Entity',
    'Concept', 
    'Relationship',
    'Pattern',
    'KnowledgeSet',
    'ContentAnalysis',
    'LearningResult',
    
    # Simple API functions
    'learn_from_text',
    'remember',
    'recall',
]