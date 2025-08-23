"""Tests for entity formatting utilities."""

import pytest
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from flowlib.utils.formatting.entities import format_entity_list, format_entity_for_display, format_entities_as_context
from flowlib.providers.graph.models import Entity, EntityAttribute, EntityRelationship


class TestFormatEntityList:
    """Test entity list formatting utilities."""
    
    def test_format_entity_list_basic(self):
        """Test basic entity list formatting."""
        entities = [
            Entity(id="1", type="person"),
            Entity(id="2", type="place"),
            Entity(id="3", type="organization")
        ]
        
        formatted = format_entity_list(entities)
        assert isinstance(formatted, str)
        assert "Person" in formatted  # Type is capitalized in output
        assert "Place" in formatted
        assert "Organization" in formatted
    
    def test_format_entity_list_empty(self):
        """Test formatting empty entity list."""
        formatted = format_entity_list([])
        assert formatted == "No entities"
    
    def test_format_entity_list_with_metadata(self):
        """Test formatting entity list with attributes."""
        entities = [
            Entity(
                id="1",
                type="person",
                attributes={
                    "name": EntityAttribute(name="name", value="Entity One", confidence=0.9),
                    "age": EntityAttribute(name="age", value="30", confidence=0.8)
                }
            ),
            Entity(
                id="2",
                type="place",
                attributes={
                    "name": EntityAttribute(name="name", value="Entity Two", confidence=0.9)
                }
            )
        ]
        
        formatted = format_entity_list(entities)
        assert isinstance(formatted, str)
        assert "Entity One" in formatted
        assert "Entity Two" in formatted
    
    def test_format_entity_list_single(self):
        """Test formatting single entity list."""
        entities = [
            Entity(id="1", type="test")
        ]
        
        formatted = format_entity_list(entities)
        assert isinstance(formatted, str)
        assert "Test" in formatted  # Type is capitalized in output
    
    def test_format_entity_list_compact(self):
        """Test formatting entity list in compact format."""
        entities = [
            Entity(id="1", type="person"),
            Entity(id="2", type="place")
        ]
        
        formatted = format_entity_list(entities, compact=True)
        assert isinstance(formatted, str)
        assert "person:1" in formatted
        assert "place:2" in formatted


class TestFormatEntityForDisplay:
    """Test individual entity formatting utilities."""
    
    def test_format_entity_for_display_basic(self):
        """Test basic entity display formatting."""
        entity = Entity(
            id="1",
            type="person",
            attributes={
                "name": EntityAttribute(name="name", value="Test Entity", confidence=0.9)
            }
        )
        
        formatted = format_entity_for_display(entity)
        assert isinstance(formatted, str)
        assert "Test Entity" in formatted
        assert "Person" in formatted  # Type is capitalized in output
        assert "1" in formatted
    
    def test_format_entity_for_display_with_metadata(self):
        """Test entity display formatting with attributes."""
        entity = Entity(
            id="1",
            type="person",
            attributes={
                "name": EntityAttribute(name="name", value="Test Entity", confidence=0.9),
                "age": EntityAttribute(name="age", value="25", confidence=0.8),
                "occupation": EntityAttribute(name="occupation", value="engineer", confidence=0.7)
            },
            last_updated="2023-01-01T12:00:00"
        )
        
        formatted = format_entity_for_display(entity)
        assert isinstance(formatted, str)
        assert "Test Entity" in formatted
        assert "Person" in formatted  # Type is capitalized in output
        assert "engineer" in formatted
    
    def test_format_entity_for_display_minimal(self):
        """Test entity display formatting with minimal data."""
        entity = Entity(id="1", type="test")
        
        formatted = format_entity_for_display(entity)
        assert isinstance(formatted, str)
        assert "Test" in formatted  # Type is capitalized in output
        assert "1" in formatted
    
    def test_format_entity_for_display_complex_metadata(self):
        """Test entity display formatting with relationships."""
        entity = Entity(
            id="1",
            type="organization",
            attributes={
                "name": EntityAttribute(name="name", value="Complex Entity", confidence=0.9),
                "location": EntityAttribute(name="location", value="New York", confidence=0.8)
            },
            relationships=[
                EntityRelationship(
                    target_entity="employee_1",
                    relation_type="employs",
                    confidence=0.9
                ),
                EntityRelationship(
                    target_entity="employee_2",
                    relation_type="employs",
                    confidence=0.8
                )
            ]
        )
        
        formatted = format_entity_for_display(entity)
        assert isinstance(formatted, str)
        assert "Complex Entity" in formatted
        assert "Organization" in formatted  # Type is capitalized in output
        assert "employs" in formatted
    
    def test_format_entity_for_display_detailed(self):
        """Test entity display formatting with detailed flag."""
        entity = Entity(
            id="1",
            type="person",
            attributes={
                "name": EntityAttribute(name="name", value="Test Entity", confidence=0.9, source="test")
            },
            importance=0.8,
            tags=["important", "test"]
        )
        
        formatted = format_entity_for_display(entity, detailed=True)
        assert isinstance(formatted, str)
        assert "Test Entity" in formatted
        assert "confidence" in formatted
        assert "Importance:" in formatted
        assert "Tags:" in formatted


class TestFormatEntitiesAsContext:
    """Test entities as context formatting utilities."""
    
    def test_format_entities_as_context_basic(self):
        """Test basic entities as context formatting."""
        entities = [
            Entity(
                id="1",
                type="person",
                attributes={
                    "name": EntityAttribute(name="name", value="John Doe", confidence=0.9)
                }
            ),
            Entity(
                id="2",
                type="place",
                attributes={
                    "name": EntityAttribute(name="name", value="New York", confidence=0.8)
                }
            )
        ]
        
        formatted = format_entities_as_context(entities)
        assert isinstance(formatted, str)
        assert "Relevant memory information" in formatted
        assert "John Doe" in formatted
        assert "New York" in formatted
    
    def test_format_entities_as_context_empty(self):
        """Test formatting empty entities as context."""
        formatted = format_entities_as_context([])
        assert formatted == ""