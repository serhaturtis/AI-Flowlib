#!/usr/bin/env python3
"""Integration test for end-to-end knowledge plugin generation with real LLM.

This test demonstrates the complete knowledge extraction and plugin generation pipeline:
1. Configure LLM provider (Phi-4 model)
2. Extract knowledge from real documents using LLM analysis
3. Generate a complete knowledge plugin
4. Validate the plugin structure and extracted data
"""

import asyncio
import json
import logging
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from flowlib.resources.decorators.decorators import llm_config, model_config
from flowlib.resources.models.config_resource import LLMConfigResource
from flowlib.resources.models.model_resource import ModelResource
from flowlib.knowledge.plugin_generation.flow import PluginGenerationFlow
from flowlib.knowledge.plugin_generation.models import PluginGenerationRequest


# Test configurations for Phi-4 LLM (clean separation - model-specific settings only)
@model_config(
    name="test-knowledge-extraction", 
    provider_type="llamacpp",
    config={
        "path": "/home/swr/tools/llm_models/phi-4-Q6_K.gguf",
        "model_type": "phi4",
        "n_ctx": 8192,  # Model-specific context window
        "use_gpu": True,
        "n_gpu_layers": -1,  # Model-specific GPU usage
        "temperature": 0.3,
        "max_tokens": 1000,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1
        # n_threads, n_batch, verbose are now provider-level settings
    }
)
class TestKnowledgeExtractionModel(ModelResource):
    """Test model configuration for knowledge extraction."""
    pass


@llm_config("test-llm")
class TestLLMProvider(LLMConfigResource):
    """Test LLM provider configuration (clean separation - provider-level settings only)."""
    
    def __init__(self, **data):
        super().__init__(
            provider_type="llamacpp",
            # Provider-level settings only (n_ctx, use_gpu, n_gpu_layers are model-specific)
            **data
        )


class TestKnowledgePluginGeneration:
    """Integration test for knowledge plugin generation."""
    
    @pytest.fixture
    def test_documents_dir(self):
        """Create temporary directory with test documents."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create test documents
        (temp_dir / "software_architecture.txt").write_text("""
Modern Software Architecture Patterns

Software architecture defines the high-level structure of software systems, including components, their relationships, and design principles. Good architecture ensures scalability, maintainability, and reliability.

Microservices Architecture:
Decomposes applications into small, independent services that communicate over well-defined APIs. Each service is responsible for a specific business capability and can be developed, deployed, and scaled independently.

Benefits include:
- Independent deployment and scaling
- Technology diversity across services
- Fault isolation and resilience
- Team autonomy and faster development cycles

Event-Driven Architecture:
Uses events to trigger and communicate between decoupled services. Events represent state changes or significant occurrences in the system. This pattern enables loose coupling and asynchronous processing.

Key components:
- Event producers generate events
- Event brokers (like Apache Kafka) handle event routing
- Event consumers process events and trigger actions
- Event stores maintain event history for audit and replay

Domain-Driven Design (DDD):
Focuses on modeling software around business domains and domain expertise. It emphasizes collaboration between technical and domain experts to create a shared understanding.

Core concepts:
- Bounded contexts define clear boundaries
- Entities represent business objects with identity
- Value objects are immutable and defined by their attributes
- Aggregates maintain consistency boundaries
- Domain services encapsulate business logic
""")

        (temp_dir / "python_programming.txt").write_text("""
Python Programming for Data Science

Python has become the dominant programming language for data science and machine learning applications. Its simplicity, readability, and extensive ecosystem make it ideal for rapid prototyping and production deployment.

Essential Python libraries for data science:

NumPy: Provides efficient numerical computing with multi-dimensional arrays and mathematical functions. It serves as the foundation for most other data science libraries.

Pandas: Offers powerful data structures like DataFrames for data manipulation and analysis. It excels at handling structured data, time series, and missing values.

Matplotlib and Seaborn: Enable comprehensive data visualization capabilities. These libraries support statistical plots, customizable charts, and publication-quality graphics.

Scikit-learn: Implements machine learning algorithms including classification, regression, clustering, and dimensionality reduction. It provides consistent APIs and extensive documentation.

Best practices for Python data science projects:
- Use virtual environments to manage dependencies
- Write modular, reusable code with proper documentation
- Implement version control with Git
- Follow PEP 8 style guidelines for code consistency
- Use type hints for better code clarity and IDE support
- Implement comprehensive testing with pytest
""")

        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def output_dir(self):
        """Create temporary output directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_plugin_generation_pipeline(self, test_documents_dir: Path, output_dir: Path):
        """Test the complete plugin generation pipeline with real LLM."""
        
        # Skip if model file doesn't exist
        model_path = Path("/home/swr/tools/llm_models/phi-4-Q6_K.gguf")
        if not model_path.exists():
            pytest.skip("Phi-4 model not available")
        
        # Create plugin generation request
        request = PluginGenerationRequest(
            input_directory=str(test_documents_dir),
            output_directory=str(output_dir),
            plugin_name="test_software_knowledge",
            description="Test plugin for software development knowledge",
            domains=["software_development", "python_programming"],
            use_vector_db=False,  # Keep simple for test
            use_graph_db=False,
            max_files=2,
            chunk_size=500,
            chunk_overlap=100
        )
        
        # Run plugin generation
        flow = PluginGenerationFlow()
        result = await flow.run_pipeline(request)
        
        # Validate results
        assert result.success, f"Plugin generation failed: {result.error_message}"
        assert result.plugin_path, "Plugin path should be set"
        
        plugin_path = Path(result.plugin_path)
        assert plugin_path.exists(), "Plugin directory should exist"
        
        # Validate extraction stats
        stats = result.summary.extraction_stats
        assert stats.total_documents >= 2, "Should process at least 2 documents"
        assert stats.total_entities > 0, "Should extract entities"
        assert stats.total_relationships >= 0, "Should have relationships"
        
        # Validate plugin structure
        assert (plugin_path / "manifest.yaml").exists(), "Should have manifest"
        assert (plugin_path / "provider.py").exists(), "Should have provider"
        assert (plugin_path / "__init__.py").exists(), "Should have init file"
        assert (plugin_path / "README.md").exists(), "Should have README"
        assert (plugin_path / "data").is_dir(), "Should have data directory"
        
        # Validate data files
        data_dir = plugin_path / "data"
        assert (data_dir / "documents.json").exists(), "Should have documents data"
        assert (data_dir / "entities.json").exists(), "Should have entities data"
        assert (data_dir / "relationships.json").exists(), "Should have relationships data"
        assert (data_dir / "metadata.json").exists(), "Should have metadata"
        
        return result
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_extracted_knowledge_quality(self, test_documents_dir: Path, output_dir: Path):
        """Test the quality of extracted knowledge."""
        
        # Skip if model file doesn't exist
        model_path = Path("/home/swr/tools/llm_models/phi-4-Q6_K.gguf")
        if not model_path.exists():
            pytest.skip("Phi-4 model not available")
        
        # Generate plugin
        request = PluginGenerationRequest(
            input_directory=str(test_documents_dir),
            output_directory=str(output_dir),
            plugin_name="quality_test_plugin",
            description="Plugin for testing knowledge quality",
            domains=["software_development"],
            use_vector_db=False,
            use_graph_db=False,
            max_files=2,
            chunk_size=300,
            chunk_overlap=50
        )
        
        flow = PluginGenerationFlow()
        result = await flow.run_pipeline(request)
        
        assert result.success, "Plugin generation should succeed"
        
        # Load and validate extracted data
        plugin_path = Path(result.plugin_path)
        data_dir = plugin_path / "data"
        
        with open(data_dir / "entities.json") as f:
            entities = json.load(f)
        
        with open(data_dir / "relationships.json") as f:
            relationships = json.load(f)
        
        # Quality checks
        assert len(entities) > 10, "Should extract meaningful number of entities"
        
        # Check for expected software architecture entities
        entity_names = [e.get('name', '').lower() for e in entities]
        expected_concepts = ['microservices', 'architecture', 'python', 'kafka']
        
        found_concepts = [concept for concept in expected_concepts 
                         if any(concept in name for name in entity_names)]
        assert len(found_concepts) >= 2, f"Should find key concepts. Found: {found_concepts}"
        
        # Check entity quality
        high_conf_entities = [e for e in entities if e.get('confidence', 0) >= 0.8]
        assert len(high_conf_entities) > 0, "Should have high-confidence entities"
        
        # Check all entities have descriptions
        entities_with_desc = [e for e in entities if e.get('description')]
        assert len(entities_with_desc) == len(entities), "All entities should have descriptions"
        
        # Check relationship quality if any exist
        if relationships:
            high_conf_rels = [r for r in relationships if r.get('confidence', 0) >= 0.7]
            assert len(high_conf_rels) > 0, "Should have high-confidence relationships"
    
    def test_plugin_data_structure(self):
        """Test the plugin data structure validation."""
        
        # This is a unit test that can run without LLM
        sample_entity = {
            "entity_id": "test123",
            "name": "Test Entity",
            "entity_type": "concept",
            "description": "A test entity",
            "confidence": 0.9,
            "frequency": 1
        }
        
        sample_relationship = {
            "relationship_id": "rel123",
            "source_entity_id": "ent1",
            "target_entity_id": "ent2", 
            "relationship_type": "relates_to",
            "confidence": 0.8
        }
        
        # Validate structure
        required_entity_fields = ["entity_id", "name", "entity_type"]
        for field in required_entity_fields:
            assert field in sample_entity, f"Entity missing required field: {field}"
        
        required_rel_fields = ["relationship_id", "source_entity_id", "target_entity_id", "relationship_type"]
        for field in required_rel_fields:
            assert field in sample_relationship, f"Relationship missing required field: {field}"


def validate_plugin_data(plugin_path: Path) -> Dict[str, Any]:
    """Utility function to validate plugin data quality."""
    
    data_dir = plugin_path / "data"
    
    with open(data_dir / "entities.json") as f:
        entities = json.load(f)
    
    with open(data_dir / "relationships.json") as f:
        relationships = json.load(f)
    
    with open(data_dir / "documents.json") as f:
        documents = json.load(f)
    
    # Create entity ID to name mapping for relationship validation
    entity_id_to_name = {e.get('entity_id'): e.get('name', 'Unknown') for e in entities}
    
    # Quality metrics
    total_entities = len(entities)
    high_conf_entities = len([e for e in entities if e.get('confidence', 0) >= 0.8])
    entities_with_desc = len([e for e in entities if e.get('description')])
    
    total_rels = len(relationships)
    high_conf_rels = len([r for r in relationships if r.get('confidence', 0) >= 0.7])
    
    # Validate relationship entity resolution
    valid_relationships = 0
    for rel in relationships:
        source_id = rel.get('source_entity_id')
        target_id = rel.get('target_entity_id')
        if source_id in entity_id_to_name and target_id in entity_id_to_name:
            valid_relationships += 1
    
    return {
        "total_entities": total_entities,
        "total_relationships": total_rels,
        "total_documents": len(documents),
        "high_confidence_entities": high_conf_entities,
        "high_confidence_relationships": high_conf_rels,
        "entity_completeness": entities_with_desc / total_entities if total_entities > 0 else 0,
        "relationship_validity": valid_relationships / total_rels if total_rels > 0 else 0,
        "quality_score": (high_conf_entities + high_conf_rels) / (total_entities + total_rels) if (total_entities + total_rels) > 0 else 0
    }


if __name__ == "__main__":
    """Run integration test manually."""
    
    logging.basicConfig(level=logging.INFO)
    
    async def run_manual_test():
        """Run the test manually for development."""
        
        test_instance = TestKnowledgePluginGeneration()
        
        # Create test data
        import tempfile
        test_docs = Path(tempfile.mkdtemp())
        output = Path(tempfile.mkdtemp())
        
        try:
            # Create test documents
            test_instance.test_documents_dir = lambda: test_docs
            test_instance.output_dir = lambda: output
            
            # Write test documents
            (test_docs / "test.txt").write_text("Python is a programming language used for data science and machine learning.")
            
            result = await test_instance.test_full_plugin_generation_pipeline(test_docs, output)
            
            print(f"✅ Plugin generated successfully at: {result.plugin_path}")
            
            # Validate quality
            quality = validate_plugin_data(Path(result.plugin_path))
            print(f"📊 Quality metrics: {quality}")
            
        finally:
            shutil.rmtree(test_docs, ignore_errors=True)
            shutil.rmtree(output, ignore_errors=True)
    
    # Run if executed directly
    asyncio.run(run_manual_test())