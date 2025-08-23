#!/usr/bin/env python3
"""Test for validating generated knowledge plugin data quality.

This test can be used to validate any generated knowledge plugin to ensure
the extracted data meets quality standards and has proper structure.
"""

import json
import pytest
from pathlib import Path
from typing import Dict, List, Any


class TestPluginDataValidation:
    """Test class for validating knowledge plugin data."""
    
    def test_plugin_structure(self, plugin_path: str):
        """Test that plugin has proper file structure."""
        
        plugin_dir = Path(plugin_path)
        
        # Required files
        required_files = [
            "manifest.yaml",
            "provider.py", 
            "__init__.py",
            "README.md"
        ]
        
        for file_name in required_files:
            file_path = plugin_dir / file_name
            assert file_path.exists(), f"Missing required file: {file_name}"
        
        # Required directories
        data_dir = plugin_dir / "data"
        assert data_dir.is_dir(), "Missing data directory"
        
        # Required data files
        required_data_files = [
            "documents.json",
            "entities.json", 
            "relationships.json",
            "metadata.json"
        ]
        
        for file_name in required_data_files:
            file_path = data_dir / file_name
            assert file_path.exists(), f"Missing data file: {file_name}"
    
    def test_entity_data_quality(self, plugin_path: str):
        """Test the quality of extracted entities."""
        
        plugin_dir = Path(plugin_path)
        entities_file = plugin_dir / "data" / "entities.json"
        
        with open(entities_file) as f:
            entities = json.load(f)
        
        assert len(entities) > 0, "Should have extracted entities"
        
        # Check entity structure
        for entity in entities:
            required_fields = ["entity_id", "name", "entity_type"]
            for field in required_fields:
                assert field in entity, f"Entity missing required field: {field}"
            
            # Check data types
            assert isinstance(entity.get('confidence', 0), (int, float)), "Confidence should be numeric"
            assert 0 <= entity.get('confidence', 0) <= 1, "Confidence should be 0-1"
            
            # Check entity type is valid
            valid_types = ["concept", "technology", "methodology", "term", "person", "organization"]
            assert entity.get('entity_type') in valid_types, f"Invalid entity type: {entity.get('entity_type')}"
        
        # Quality metrics
        high_conf_entities = [e for e in entities if e.get('confidence', 0) >= 0.7]
        entities_with_desc = [e for e in entities if e.get('description')]
        
        # At least 50% should be high confidence
        conf_ratio = len(high_conf_entities) / len(entities)
        assert conf_ratio >= 0.5, f"Low confidence ratio: {conf_ratio:.2f}"
        
        # At least 80% should have descriptions
        desc_ratio = len(entities_with_desc) / len(entities)
        assert desc_ratio >= 0.8, f"Low description ratio: {desc_ratio:.2f}"
    
    def test_relationship_data_quality(self, plugin_path: str):
        """Test the quality of extracted relationships."""
        
        plugin_dir = Path(plugin_path)
        
        # Load entities for validation
        with open(plugin_dir / "data" / "entities.json") as f:
            entities = json.load(f)
        
        with open(plugin_dir / "data" / "relationships.json") as f:
            relationships = json.load(f)
        
        # Build entity ID lookup
        entity_ids = {e.get('entity_id') for e in entities}
        
        # Validate relationships
        for rel in relationships:
            required_fields = ["relationship_id", "source_entity_id", "target_entity_id", "relationship_type"]
            for field in required_fields:
                assert field in rel, f"Relationship missing required field: {field}"
            
            # Check entity references are valid
            source_id = rel.get('source_entity_id')
            target_id = rel.get('target_entity_id')
            
            assert source_id in entity_ids, f"Invalid source entity ID: {source_id}"
            assert target_id in entity_ids, f"Invalid target entity ID: {target_id}"
            
            # Check confidence
            conf = rel.get('confidence', 0)
            assert isinstance(conf, (int, float)), "Confidence should be numeric"
            assert 0 <= conf <= 1, "Confidence should be 0-1"
            
            # Check relationship type
            valid_rel_types = [
                "relates_to", "implements", "uses", "depends_on", "extends",
                "supports", "derived_from", "part_of", "mentions"
            ]
            rel_type = rel.get('relationship_type')
            assert rel_type in valid_rel_types, f"Invalid relationship type: {rel_type}"
    
    def test_document_processing_completeness(self, plugin_path: str):
        """Test that document processing was complete."""
        
        plugin_dir = Path(plugin_path)
        
        with open(plugin_dir / "data" / "documents.json") as f:
            documents = json.load(f)
        
        with open(plugin_dir / "data" / "metadata.json") as f:
            metadata = json.load(f)
        
        assert len(documents) > 0, "Should have processed documents"
        
        # Check document structure
        for doc in documents:
            assert "document_id" in doc, "Document missing ID"
            assert "status" in doc, "Document missing status"
            assert doc.get('status') == "completed", f"Document not completed: {doc.get('status')}"
    
    def test_knowledge_coverage(self, plugin_path: str, expected_domains: List[str] = None):
        """Test that knowledge covers expected domains."""
        
        plugin_dir = Path(plugin_path)
        
        with open(plugin_dir / "data" / "entities.json") as f:
            entities = json.load(f)
        
        # Extract entity names and descriptions for domain analysis
        all_text = ""
        for entity in entities:
            all_text += f" {entity.get('name', '')} {entity.get('description', '')}"
        
        all_text = all_text.lower()
        
        if expected_domains:
            for domain in expected_domains:
                domain_keywords = {
                    "software_development": ["software", "architecture", "microservices", "api", "service"],
                    "python_programming": ["python", "pandas", "numpy", "library", "framework"],
                    "ai_development": ["neural", "machine learning", "deep learning", "model", "training"],
                    "data_science": ["data", "analysis", "visualization", "statistics", "pandas"]
                }
                
                keywords = domain_keywords.get(domain, [domain])
                found_keywords = [kw for kw in keywords if kw in all_text]
                
                assert len(found_keywords) > 0, f"No coverage for domain '{domain}'. Expected keywords: {keywords}"


def validate_plugin_comprehensive(plugin_path: str) -> Dict[str, Any]:
    """Comprehensive validation of a knowledge plugin."""
    
    plugin_dir = Path(plugin_path)
    
    # Load all data
    with open(plugin_dir / "data" / "entities.json") as f:
        entities = json.load(f)
    
    with open(plugin_dir / "data" / "relationships.json") as f:
        relationships = json.load(f)
    
    with open(plugin_dir / "data" / "documents.json") as f:
        documents = json.load(f)
    
    # Entity analysis
    entity_types = {}
    high_conf_entities = 0
    entities_with_desc = 0
    
    for entity in entities:
        entity_type = entity.get('entity_type', 'unknown')
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        if entity.get('confidence', 0) >= 0.8:
            high_conf_entities += 1
        
        if entity.get('description'):
            entities_with_desc += 1
    
    # Relationship analysis
    rel_types = {}
    high_conf_rels = 0
    valid_rels = 0
    
    entity_ids = {e.get('entity_id') for e in entities}
    
    for rel in relationships:
        rel_type = rel.get('relationship_type', 'unknown')
        rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
        
        if rel.get('confidence', 0) >= 0.7:
            high_conf_rels += 1
        
        # Check if relationship references valid entities
        source_id = rel.get('source_entity_id')
        target_id = rel.get('target_entity_id')
        if source_id in entity_ids and target_id in entity_ids:
            valid_rels += 1
    
    # Calculate quality metrics
    total_entities = len(entities)
    total_rels = len(relationships)
    
    entity_quality = high_conf_entities / total_entities if total_entities > 0 else 0
    entity_completeness = entities_with_desc / total_entities if total_entities > 0 else 0
    relationship_quality = high_conf_rels / total_rels if total_rels > 0 else 0
    relationship_validity = valid_rels / total_rels if total_rels > 0 else 0
    
    overall_quality = (entity_quality + relationship_quality) / 2
    
    return {
        "plugin_path": str(plugin_path),
        "total_entities": total_entities,
        "total_relationships": total_rels,
        "total_documents": len(documents),
        "entity_types": entity_types,
        "relationship_types": rel_types,
        "quality_metrics": {
            "entity_quality": entity_quality,
            "entity_completeness": entity_completeness,
            "relationship_quality": relationship_quality,
            "relationship_validity": relationship_validity,
            "overall_quality": overall_quality
        },
        "validation_passed": (
            entity_quality >= 0.5 and
            entity_completeness >= 0.8 and
            relationship_validity >= 0.9 and
            total_entities > 5
        )
    }


if __name__ == "__main__":
    """Test the plugin we generated."""
    
    import sys
    
    plugin_path = "/tmp/knowledge_output_real"
    
    if len(sys.argv) > 1:
        plugin_path = sys.argv[1]
    
    if not Path(plugin_path).exists():
        print(f"❌ Plugin path does not exist: {plugin_path}")
        sys.exit(1)
    
    print(f"🧪 Validating plugin at: {plugin_path}")
    print("=" * 60)
    
    try:
        validator = TestPluginDataValidation()
        
        # Run all validation tests
        validator.test_plugin_structure(plugin_path)
        print("✅ Plugin structure validation passed")
        
        validator.test_entity_data_quality(plugin_path)
        print("✅ Entity data quality validation passed")
        
        validator.test_relationship_data_quality(plugin_path)
        print("✅ Relationship data quality validation passed")
        
        validator.test_document_processing_completeness(plugin_path)
        print("✅ Document processing validation passed")
        
        validator.test_knowledge_coverage(plugin_path, ["software_development", "python_programming"])
        print("✅ Knowledge coverage validation passed")
        
        # Get comprehensive report
        report = validate_plugin_comprehensive(plugin_path)
        
        print(f"\n📊 COMPREHENSIVE VALIDATION REPORT:")
        print(f"✅ Total entities: {report['total_entities']}")
        print(f"✅ Total relationships: {report['total_relationships']}")
        print(f"✅ Total documents: {report['total_documents']}")
        print(f"✅ Entity quality: {report['quality_metrics']['entity_quality']:.1%}")
        print(f"✅ Entity completeness: {report['quality_metrics']['entity_completeness']:.1%}")
        print(f"✅ Relationship quality: {report['quality_metrics']['relationship_quality']:.1%}")
        print(f"✅ Relationship validity: {report['quality_metrics']['relationship_validity']:.1%}")
        print(f"✅ Overall quality: {report['quality_metrics']['overall_quality']:.1%}")
        
        if report['validation_passed']:
            print(f"\n🎉 PLUGIN VALIDATION PASSED!")
            print(f"The knowledge plugin meets all quality standards.")
        else:
            print(f"\n⚠️  Plugin validation has issues but data is present.")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)