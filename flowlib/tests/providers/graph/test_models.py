"""Tests for graph models."""

import pytest
from datetime import datetime
from unittest.mock import patch
from pydantic import ValidationError
from flowlib.providers.graph.models import EntityAttribute, EntityRelationship, Entity, RelationshipUpdate


class TestEntityAttribute:
    """Test EntityAttribute model."""
    
    def test_create_entity_attribute_minimal(self):
        """Test creating EntityAttribute with minimal required fields."""
        attr = EntityAttribute(name="age", value="25")
        
        assert attr.name == "age"
        assert attr.value == "25"
        assert attr.confidence == 0.9  # Default value
        assert attr.source == "conversation"  # Default value
        assert attr.timestamp is not None  # Auto-generated
    
    def test_create_entity_attribute_full(self):
        """Test creating EntityAttribute with all fields."""
        timestamp = "2023-01-01T12:00:00"
        attr = EntityAttribute(
            name="full_name",
            value="John Doe", 
            confidence=0.8,
            source="user_profile",
            timestamp=timestamp
        )
        
        assert attr.name == "full_name"
        assert attr.value == "John Doe"
        assert attr.confidence == 0.8
        assert attr.source == "user_profile"
        assert attr.timestamp == timestamp
    
    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence values
        EntityAttribute(name="test", value="test", confidence=0.0)
        EntityAttribute(name="test", value="test", confidence=0.5)
        EntityAttribute(name="test", value="test", confidence=1.0)
        
        # Invalid confidence values
        with pytest.raises(ValidationError):
            EntityAttribute(name="test", value="test", confidence=-0.1)
        
        with pytest.raises(ValidationError):
            EntityAttribute(name="test", value="test", confidence=1.1)
    
    def test_required_fields(self):
        """Test that required fields are enforced."""
        # Missing name
        with pytest.raises(ValidationError):
            EntityAttribute(value="test")
        
        # Missing value
        with pytest.raises(ValidationError):
            EntityAttribute(name="test")
    
    @patch('flowlib.providers.graph.models.datetime')
    def test_timestamp_default(self, mock_datetime):
        """Test automatic timestamp generation."""
        mock_datetime.now.return_value.isoformat.return_value = "2023-01-15T10:30:00"
        
        attr = EntityAttribute(name="test", value="test")
        assert attr.timestamp == "2023-01-15T10:30:00"


class TestEntityRelationship:
    """Test EntityRelationship model."""
    
    def test_create_relationship_minimal(self):
        """Test creating EntityRelationship with minimal fields."""
        rel = EntityRelationship(relation_type="friend_of", target_entity="john")
        
        assert rel.relation_type == "friend_of"
        assert rel.target_entity == "john"
        assert rel.confidence == 0.9  # Default
        assert rel.source == "conversation"  # Default
        assert rel.timestamp is not None
    
    def test_create_relationship_full(self):
        """Test creating EntityRelationship with all fields."""
        timestamp = "2023-02-01T14:30:00"
        rel = EntityRelationship(
            relation_type="works_at",
            target_entity="company_abc",
            confidence=0.95,
            source="linkedin",
            timestamp=timestamp
        )
        
        assert rel.relation_type == "works_at"
        assert rel.target_entity == "company_abc"
        assert rel.confidence == 0.95
        assert rel.source == "linkedin"
        assert rel.timestamp == timestamp
    
    def test_confidence_validation(self):
        """Test confidence validation for relationships."""
        # Valid values
        EntityRelationship(relation_type="test", target_entity="test", confidence=0.0)
        EntityRelationship(relation_type="test", target_entity="test", confidence=1.0)
        
        # Invalid values
        with pytest.raises(ValidationError):
            EntityRelationship(relation_type="test", target_entity="test", confidence=-0.1)
        
        with pytest.raises(ValidationError):
            EntityRelationship(relation_type="test", target_entity="test", confidence=1.5)
    
    def test_required_fields(self):
        """Test required fields for relationships."""
        with pytest.raises(ValidationError):
            EntityRelationship(target_entity="test")  # Missing relation_type
        
        with pytest.raises(ValidationError):
            EntityRelationship(relation_type="test")  # Missing target_entity


class TestEntity:
    """Test Entity model."""
    
    def test_create_entity_minimal(self):
        """Test creating Entity with minimal fields."""
        entity = Entity(id="person_1", type="person")
        
        assert entity.id == "person_1"
        assert entity.type == "person"
        assert entity.attributes == {}
        assert entity.relationships == []
        assert entity.tags == []
        assert entity.importance == 0.7  # Default
        assert entity.vector_id is None
        assert entity.last_updated is not None
    
    def test_create_entity_full(self):
        """Test creating Entity with all fields."""
        attributes = {
            "name": EntityAttribute(name="name", value="Alice"),
            "age": EntityAttribute(name="age", value="30")
        }
        relationships = [
            EntityRelationship(relation_type="friend_of", target_entity="bob"),
            EntityRelationship(relation_type="works_at", target_entity="tech_corp")
        ]
        tags = ["important", "customer"]
        timestamp = "2023-03-01T09:00:00"
        
        entity = Entity(
            id="alice_123",
            type="person",
            attributes=attributes,
            relationships=relationships,
            tags=tags,
            importance=0.9,
            vector_id="vec_456",
            last_updated=timestamp
        )
        
        assert entity.id == "alice_123"
        assert entity.type == "person"
        assert len(entity.attributes) == 2
        assert len(entity.relationships) == 2
        assert entity.tags == tags
        assert entity.importance == 0.9
        assert entity.vector_id == "vec_456"
        assert entity.last_updated == timestamp
    
    def test_importance_validation(self):
        """Test importance score validation."""
        # Valid values
        Entity(id="test", type="test", importance=0.0)
        Entity(id="test", type="test", importance=0.5)
        Entity(id="test", type="test", importance=1.0)
        
        # Invalid values
        with pytest.raises(ValidationError):
            Entity(id="test", type="test", importance=-0.1)
        
        with pytest.raises(ValidationError):
            Entity(id="test", type="test", importance=1.2)
    
    def test_required_fields(self):
        """Test required fields for Entity."""
        with pytest.raises(ValidationError):
            Entity(type="person")  # Missing id
        
        with pytest.raises(ValidationError):
            Entity(id="test")  # Missing type
    
    @patch('flowlib.providers.graph.models.datetime')
    def test_last_updated_default(self, mock_datetime):
        """Test automatic last_updated timestamp."""
        mock_datetime.now.return_value.isoformat.return_value = "2023-04-01T15:45:00"
        
        entity = Entity(id="test", type="test")
        assert entity.last_updated == "2023-04-01T15:45:00"


class TestEntityToMemoryItem:
    """Test Entity.to_memory_item method."""
    
    def test_to_memory_item_specific_attribute(self):
        """Test converting specific attribute to memory item."""
        attr = EntityAttribute(
            name="full_name",
            value="John Smith",
            confidence=0.8,
            source="profile",
            timestamp="2023-01-01T12:00:00"
        )
        
        relationships = [
            EntityRelationship(relation_type="friend_of", target_entity="alice"),
            EntityRelationship(relation_type="colleague_of", target_entity="bob")
        ]
        
        entity = Entity(
            id="person_123",
            type="person",
            attributes={"full_name": attr},
            relationships=relationships,
            tags=["vip", "customer"],
            importance=0.9
        )
        
        memory_item = entity.to_memory_item("full_name")
        
        assert memory_item["entity_id"] == "person_123"
        assert memory_item["entity_type"] == "person"
        assert memory_item["attribute"] == "full_name"
        assert memory_item["value"] == "John Smith"
        assert memory_item["confidence"] == 0.8
        assert memory_item["importance"] == 0.9
        assert memory_item["source"] == "profile"
        assert memory_item["tags"] == ["vip", "customer"]
        assert memory_item["timestamp"] == "2023-01-01T12:00:00"
        
        # Check relationships format
        expected_relationships = [
            {"type": "friend_of", "target": "alice"},
            {"type": "colleague_of", "target": "bob"}
        ]
        assert memory_item["relationships"] == expected_relationships
    
    def test_to_memory_item_nonexistent_attribute(self):
        """Test converting nonexistent attribute returns summary."""
        entity = Entity(
            id="test_entity",
            type="test",
            attributes={"name": EntityAttribute(name="name", value="Test")},
            relationships=[],
            last_updated="2023-05-01T10:00:00"
        )
        
        memory_item = entity.to_memory_item("nonexistent")
        
        assert memory_item["entity_id"] == "test_entity"
        assert memory_item["attribute"] == "summary"
        assert memory_item["value"] == "test with 1 attributes and 0 relationships"
        assert memory_item["confidence"] == 1.0
        assert memory_item["source"] == "system"
        assert memory_item["timestamp"] == "2023-05-01T10:00:00"
    
    def test_to_memory_item_no_attribute_specified(self):
        """Test converting entity without specific attribute returns summary."""
        entity = Entity(
            id="summary_test",
            type="location",
            attributes={
                "name": EntityAttribute(name="name", value="New York"),
                "population": EntityAttribute(name="population", value="8000000")
            },
            relationships=[
                EntityRelationship(relation_type="located_in", target_entity="usa"),
                EntityRelationship(relation_type="has_district", target_entity="manhattan")
            ],
            importance=0.95,
            last_updated="2023-06-01T08:30:00"
        )
        
        memory_item = entity.to_memory_item()
        
        assert memory_item["entity_id"] == "summary_test"
        assert memory_item["entity_type"] == "location"
        assert memory_item["attribute"] == "summary"
        assert memory_item["value"] == "location with 2 attributes and 2 relationships"
        assert memory_item["confidence"] == 1.0
        assert memory_item["importance"] == 0.95
        assert memory_item["source"] == "system"
        assert memory_item["timestamp"] == "2023-06-01T08:30:00"
        
        expected_relationships = [
            {"type": "located_in", "target": "usa"},
            {"type": "has_district", "target": "manhattan"}
        ]
        assert memory_item["relationships"] == expected_relationships
    
    def test_to_memory_item_empty_entity(self):
        """Test converting entity with no attributes or relationships."""
        entity = Entity(
            id="empty_entity",
            type="unknown",
            last_updated="2023-07-01T12:00:00"
        )
        
        memory_item = entity.to_memory_item()
        
        assert memory_item["value"] == "unknown with 0 attributes and 0 relationships"
        assert memory_item["relationships"] == []
        assert memory_item["tags"] == []


class TestEntityGetFormattedView:
    """Test Entity.get_formatted_view method."""
    
    def test_get_formatted_view_full_entity(self):
        """Test formatted view with all components."""
        attributes = {
            "name": EntityAttribute(name="name", value="John Doe", confidence=0.9),
            "age": EntityAttribute(name="age", value="35", confidence=0.8),
            "occupation": EntityAttribute(name="occupation", value="Engineer", confidence=0.95)
        }
        
        relationships = [
            EntityRelationship(relation_type="married_to", target_entity="jane_doe", confidence=0.9),
            EntityRelationship(relation_type="works_at", target_entity="tech_company", confidence=0.85)
        ]
        
        entity = Entity(
            id="john_doe_123",
            type="person",
            attributes=attributes,
            relationships=relationships,
            tags=["important", "customer", "engineer"]
        )
        
        formatted = entity.get_formatted_view()
        
        # Check main header
        assert "Entity: john_doe_123 (Type: person)" in formatted
        
        # Check attributes section
        assert "Attributes:" in formatted
        assert "name: John Doe (confidence: 0.90)" in formatted
        assert "age: 35 (confidence: 0.80)" in formatted
        assert "occupation: Engineer (confidence: 0.95)" in formatted
        
        # Check relationships section
        assert "Relationships:" in formatted
        assert "married_to jane_doe (confidence: 0.90)" in formatted
        assert "works_at tech_company (confidence: 0.85)" in formatted
        
        # Check tags section
        assert "Tags: important, customer, engineer" in formatted
    
    def test_get_formatted_view_minimal_entity(self):
        """Test formatted view with minimal entity."""
        entity = Entity(id="minimal", type="test")
        
        formatted = entity.get_formatted_view()
        
        assert formatted == "Entity: minimal (Type: test)"
        assert "Attributes:" not in formatted
        assert "Relationships:" not in formatted
        assert "Tags:" not in formatted
    
    def test_get_formatted_view_attributes_only(self):
        """Test formatted view with only attributes."""
        entity = Entity(
            id="attr_only",
            type="test",
            attributes={
                "key1": EntityAttribute(name="key1", value="value1", confidence=0.7)
            }
        )
        
        formatted = entity.get_formatted_view()
        
        assert "Entity: attr_only (Type: test)" in formatted
        assert "Attributes:" in formatted
        assert "key1: value1 (confidence: 0.70)" in formatted
        assert "Relationships:" not in formatted
        assert "Tags:" not in formatted
    
    def test_get_formatted_view_relationships_only(self):
        """Test formatted view with only relationships."""
        entity = Entity(
            id="rel_only",
            type="test",
            relationships=[
                EntityRelationship(relation_type="connects_to", target_entity="other", confidence=0.6)
            ]
        )
        
        formatted = entity.get_formatted_view()
        
        assert "Entity: rel_only (Type: test)" in formatted
        assert "Attributes:" not in formatted
        assert "Relationships:" in formatted
        assert "connects_to other (confidence: 0.60)" in formatted
        assert "Tags:" not in formatted
    
    def test_get_formatted_view_tags_only(self):
        """Test formatted view with only tags."""
        entity = Entity(
            id="tags_only",
            type="test",
            tags=["tag1", "tag2", "tag3"]
        )
        
        formatted = entity.get_formatted_view()
        
        assert "Entity: tags_only (Type: test)" in formatted
        assert "Attributes:" not in formatted
        assert "Relationships:" not in formatted
        assert "Tags: tag1, tag2, tag3" in formatted
    
    def test_get_formatted_view_confidence_formatting(self):
        """Test that confidence values are properly formatted to 2 decimal places."""
        entity = Entity(
            id="format_test",
            type="test",
            attributes={
                "precise": EntityAttribute(name="precise", value="test", confidence=0.123456)
            },
            relationships=[
                EntityRelationship(relation_type="test", target_entity="target", confidence=0.987654)
            ]
        )
        
        formatted = entity.get_formatted_view()
        
        assert "precise: test (confidence: 0.12)" in formatted
        assert "test target (confidence: 0.99)" in formatted


class TestRelationshipUpdate:
    """Test RelationshipUpdate model."""
    
    def test_create_relationship_update(self):
        """Test creating RelationshipUpdate."""
        update = RelationshipUpdate(type="friend_of", target="person_123")
        
        assert update.type == "friend_of"
        assert update.target == "person_123"
    
    def test_relationship_update_required_fields(self):
        """Test required fields for RelationshipUpdate."""
        with pytest.raises(ValidationError):
            RelationshipUpdate(target="test")  # Missing type
        
        with pytest.raises(ValidationError):
            RelationshipUpdate(type="test")  # Missing target
    
    def test_relationship_update_field_descriptions(self):
        """Test that field descriptions are properly set."""
        # This tests the Pydantic Field descriptions
        update = RelationshipUpdate(type="colleague_of", target="alice")
        
        # Access field info to verify descriptions
        fields = RelationshipUpdate.model_fields
        assert fields['type'].description == "Type of relationship"
        assert fields['target'].description == "ID of the target entity"


class TestIntegration:
    """Integration tests for graph models."""
    
    def test_entity_with_complex_data(self):
        """Test entity with complex nested data structures."""
        # Create a person entity with various attributes and relationships
        person_attrs = {
            "full_name": EntityAttribute(
                name="full_name", 
                value="Dr. Sarah Wilson",
                confidence=0.95,
                source="business_card"
            ),
            "profession": EntityAttribute(
                name="profession",
                value="Data Scientist", 
                confidence=0.8,
                source="linkedin"
            ),
            "location": EntityAttribute(
                name="location",
                value="San Francisco, CA",
                confidence=0.7,
                source="conversation"
            )
        }
        
        person_relationships = [
            EntityRelationship(
                relation_type="works_at",
                target_entity="ai_startup_xyz", 
                confidence=0.9,
                source="linkedin"
            ),
            EntityRelationship(
                relation_type="collaborates_with",
                target_entity="university_research_lab",
                confidence=0.75,
                source="publication"
            ),
            EntityRelationship(
                relation_type="mentor_to",
                target_entity="junior_researcher_bob",
                confidence=0.6,
                source="conversation"
            )
        ]
        
        person = Entity(
            id="dr_sarah_wilson",
            type="person",
            attributes=person_attrs,
            relationships=person_relationships,
            tags=["scientist", "mentor", "ai_expert"],
            importance=0.9,
            vector_id="embedding_vec_789"
        )
        
        # Test memory item conversion for specific attribute
        name_memory = person.to_memory_item("full_name")
        assert name_memory["value"] == "Dr. Sarah Wilson"
        assert name_memory["confidence"] == 0.95
        assert len(name_memory["relationships"]) == 3
        
        # Test formatted view
        formatted = person.get_formatted_view()
        assert "Dr. Sarah Wilson" in formatted
        assert "Data Scientist" in formatted
        assert "works_at ai_startup_xyz" in formatted
        assert "scientist, mentor, ai_expert" in formatted
        
        # Test summary memory item
        summary_memory = person.to_memory_item()
        assert "person with 3 attributes and 3 relationships" in summary_memory["value"]
        assert summary_memory["importance"] == 0.9
    
    def test_model_serialization(self):
        """Test that models can be properly serialized and deserialized."""
        # Create entity with all components
        entity = Entity(
            id="serialization_test",
            type="test_entity",
            attributes={
                "test_attr": EntityAttribute(name="test_attr", value="test_value", confidence=0.8)
            },
            relationships=[
                EntityRelationship(relation_type="test_rel", target_entity="test_target", confidence=0.9)
            ],
            tags=["test_tag"],
            importance=0.85,
            vector_id="test_vector"
        )
        
        # Test model_dump (serialization)
        entity_dict = entity.model_dump()
        assert entity_dict["id"] == "serialization_test"
        assert "attributes" in entity_dict
        assert "relationships" in entity_dict
        
        # Test model reconstruction (deserialization)
        reconstructed = Entity.model_validate(entity_dict)
        assert reconstructed.id == entity.id
        assert reconstructed.type == entity.type
        assert len(reconstructed.attributes) == len(entity.attributes)
        assert len(reconstructed.relationships) == len(entity.relationships)
        assert reconstructed.tags == entity.tags
        assert reconstructed.importance == entity.importance