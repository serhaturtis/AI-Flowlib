"""Plugin generation flow."""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import yaml  # type: ignore[import-untyped]

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.knowledge.models import KnowledgeExtractionRequest
from flowlib.knowledge.streaming.flow import KnowledgeExtractionFlow
from flowlib.knowledge.plugin_generation.models import (
    PluginGenerationRequest, PluginGenerationResult, PluginGenerationSummary,
    ExtractionStats, ProcessedDataStats, ProcessedData
)
from flowlib.knowledge.plugin_generation.domain_strategies import domain_strategy_registry

logger = logging.getLogger(__name__)


@flow(name="plugin-generation", description="Generate knowledge plugins from document collections")  # type: ignore[arg-type]
class PluginGenerationFlow:
    """Generator that creates knowledge plugins from document collections."""
    
    def __init__(self) -> None:
        self.templates_dir = Path(__file__).parent / "plugin_templates"
        self.ensure_templates_exist()

    def ensure_templates_exist(self) -> None:
        """Ensure template directory exists."""
        self.templates_dir.mkdir(exist_ok=True)
    
    @pipeline(input_model=PluginGenerationRequest, output_model=PluginGenerationResult)
    async def run_pipeline(self, request: PluginGenerationRequest) -> PluginGenerationResult:
        """Generate a complete knowledge plugin from documents."""
        
        logger.info(f"🚀 Generating knowledge plugin: {request.plugin_name}")
        logger.info(f"📂 Input: {request.input_directory}")
        logger.info(f"📁 Output: {request.output_directory}")
        logger.info(f"🎯 Domain Strategy: {request.domain_strategy.value}")
        
        try:
            # Get domain strategy
            domain_strategy = domain_strategy_registry.get_strategy(
                request.domain_strategy, 
                request.domain_config
            )
            logger.info(f"📋 Using strategy: {domain_strategy.strategy_name}")
            
            # Create output directory
            plugin_dir = Path(request.output_directory)
            plugin_dir.mkdir(parents=True, exist_ok=True)
            
            # Create temporary directory for extraction results
            temp_dir = plugin_dir / "temp_extraction"
            temp_dir.mkdir(exist_ok=True)
            
            try:
                # Step 1: Run knowledge extraction
                logger.info("🔍 Step 1: Extracting knowledge from documents...")
                extraction_stats = await self._run_knowledge_extraction(
                    request=request,
                    temp_dir=temp_dir
                )
                
                # Step 2: Process extraction results
                logger.info("📊 Step 2: Processing extraction results...")
                processed_data = await self._process_extraction_results(temp_dir)
                
                # Step 3: Generate plugin files
                logger.info("📝 Step 3: Generating plugin files...")
                await self._generate_plugin_files(
                    plugin_dir=plugin_dir,
                    request=request,
                    processed_data=processed_data,
                    extraction_stats=extraction_stats
                )
                
                # Step 4: Create database configs and embedded database files
                logger.info("💾 Step 4: Creating database configs and embedded database files...")
                await self._create_database_configs(plugin_dir, request.plugin_name, request.use_vector_db, request.use_graph_db)
                await self._create_embedded_databases(plugin_dir, temp_dir, request.plugin_name, request.use_vector_db, request.use_graph_db)
                await self._create_data_files(plugin_dir, processed_data)
                
                # Clean up temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                # Generate summary
                files_created = [
                    "manifest.yaml",
                    "provider.py", 
                    "__init__.py",
                    "README.md",
                    "test_plugin.py",
                    "data/documents.json",
                    "data/entities.json",
                    "data/relationships.json",
                    "data/chunks.json"
                ]
                
                if request.use_vector_db:
                    files_created.extend(["chromadb_config.yaml", "databases/chromadb/"])
                
                if request.use_graph_db:
                    files_created.extend(["neo4j_config.yaml", "databases/neo4j/"])
                
                summary = PluginGenerationSummary(
                    plugin_name=request.plugin_name,
                    plugin_directory=str(plugin_dir),
                    domains=request.domains,
                    generation_time=datetime.now().isoformat(),
                    extraction_stats=ExtractionStats(**extraction_stats),
                    processed_data_stats=ProcessedDataStats(
                        documents=len(processed_data.documents),
                        entities=len(processed_data.entities),
                        relationships=len(processed_data.relationships),
                        chunks=len(processed_data.chunks)
                    ),
                    files_created=files_created
                )
                
                logger.info("✅ Plugin generation completed successfully!")
                logger.info(f"📦 Plugin created at: {plugin_dir}")
                
                return PluginGenerationResult(
                    success=True,
                    plugin_path=str(plugin_dir),
                    summary=summary
                )
                
            except Exception as e:
                logger.error(f"❌ Plugin generation failed: {e}")
                # Clean up on failure
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                raise
                
        except Exception as e:
            return PluginGenerationResult(
                success=False,
                plugin_path="",
                summary=PluginGenerationSummary(
                    plugin_name=request.plugin_name,
                    plugin_directory="",
                    domains=request.domains,
                    generation_time=datetime.now().isoformat(),
                    extraction_stats=ExtractionStats(extraction_error=str(e)),
                    processed_data_stats=ProcessedDataStats(),
                    files_created=[]
                ),
                error_message=str(e)
            )

    async def _run_knowledge_extraction(
        self,
        request: PluginGenerationRequest,
        temp_dir: Path
    ) -> Dict[str, Any]:
        """Run the knowledge extraction pipeline."""
        
        # Setup database directories for extraction
        if request.use_vector_db:
            chroma_temp_dir = temp_dir / "chroma_data"
            chroma_temp_dir.mkdir(exist_ok=True)
        
        if request.use_graph_db:
            neo4j_temp_dir = temp_dir / "neo4j_data"
            neo4j_temp_dir.mkdir(exist_ok=True)
        
        # Create knowledge extraction request
        extraction_request = KnowledgeExtractionRequest(
            input_directory=request.input_directory,
            output_directory=str(temp_dir),
            collection_name="knowledge_base",
            graph_name="knowledge_graph",
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            extract_entities=True,
            extract_relationships=True,
            create_summaries=True,
            detect_topics=True,
            extraction_domain=request.domains[0] if request.domains else "general",
            llm_model_name="music-album-model",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            vector_dimensions=384,
            vector_provider_name="chroma",
            embedding_provider_name="default-embedding",
            enable_graph_analysis=True,
            min_entity_frequency=2,
            min_relationship_confidence=0.7,
            graph_provider_name="neo4j",
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="password",
            use_vector_db=request.use_vector_db,
            use_graph_db=request.use_graph_db,
            max_files=request.max_files,
            resume_from_checkpoint=False,
            plugin_name_prefix=request.plugin_name
        )
        
        # Create extraction flow
        flow = KnowledgeExtractionFlow()
        
        try:
            # Run extraction pipeline
            result = await flow.run_pipeline(extraction_request)
            
            return ExtractionStats(
                total_documents=result.final_stats.total_documents,
                successful_documents=result.final_stats.successful_documents,
                failed_documents=result.final_stats.failed_documents,
                total_entities=result.final_stats.total_entities,
                total_relationships=result.final_stats.total_relationships,
                total_chunks=result.final_stats.total_chunks,
                processing_time=result.final_stats.processing_time_seconds
            ).model_dump()
            
        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")
            # Don't create plugins with fake data - fail properly
            raise RuntimeError(f"Knowledge extraction pipeline failed: {e}")
    
    async def _process_extraction_results(self, temp_dir: Path) -> ProcessedData:
        """Process the results from knowledge extraction."""
        processed_data = ProcessedData()
        
        # Look for the streaming plugin export directory
        streaming_plugin_dirs = list(temp_dir.glob("*_final_*"))
        if not streaming_plugin_dirs:
            raise RuntimeError(
                "Knowledge extraction failed - no plugin export found. "
                "The streaming extraction flow should create a plugin directory ending with '_final_*'"
            )
        
        streaming_plugin_dir = streaming_plugin_dirs[0]
        logger.info(f"Processing extraction results from: {streaming_plugin_dir}")
        
        # Load data files from the streaming plugin
        data_dir = streaming_plugin_dir / "data"
        if not data_dir.exists():
            raise RuntimeError(
                f"Invalid plugin structure - missing data directory: {data_dir}"
            )
        
        try:
            # Load documents
            documents_file = data_dir / "documents.json"
            if documents_file.exists():
                with open(documents_file, 'r') as f:
                    processed_data.documents = json.load(f)
            
            # Load entities
            entities_file = data_dir / "entities.json"
            if entities_file.exists():
                with open(entities_file, 'r') as f:
                    processed_data.entities = json.load(f)
            
            # Load relationships
            relationships_file = data_dir / "relationships.json"
            if relationships_file.exists():
                with open(relationships_file, 'r') as f:
                    processed_data.relationships = json.load(f)
            
            # Load chunks
            chunks_file = data_dir / "chunks.json"
            if chunks_file.exists():
                with open(chunks_file, 'r') as f:
                    processed_data.chunks = json.load(f)
            
            logger.info(f"Loaded extraction data: {len(processed_data.documents)} docs, {len(processed_data.entities)} entities, {len(processed_data.relationships)} relationships")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load extraction data from {data_dir}: {e}")
        
        if not processed_data.documents:
            raise RuntimeError(
                "Knowledge extraction failed - no documents were processed successfully. "
                "Please check that:\n"
                "1. LLM provider resources are properly configured\n"
                "2. Input directory contains supported document formats (.txt, .md, .pdf, .doc, .docx)\n"
                "3. Documents are readable and not empty\n"
                "4. Required dependencies are installed"
            )
        
        return processed_data
    

    async def _generate_plugin_files(
        self,
        plugin_dir: Path,
        request: PluginGenerationRequest,
        processed_data: ProcessedData,
        extraction_stats: Dict[str, Any]
    ) -> None:
        """Generate all plugin files."""
        
        # Generate manifest.yaml
        await self._generate_manifest(
            plugin_dir, request, extraction_stats
        )
        
        # Generate provider.py
        await self._generate_provider(
            plugin_dir, request, processed_data
        )
        
        # Generate __init__.py
        await self._generate_init_file(plugin_dir, request)
        
        # Generate README.md
        await self._generate_readme(
            plugin_dir, request, extraction_stats
        )
        
        # Generate test script
        await self._generate_test_script(plugin_dir, request)

    # Additional helper methods for file generation would continue here...
    # For brevity, I'm including placeholders for the remaining methods
    
    async def _generate_manifest(self, plugin_dir: Path, request: PluginGenerationRequest, extraction_stats: Dict[str, Any]) -> None:
        """Generate manifest.yaml file."""
        
        manifest_data = {
            "name": request.plugin_name,
            "version": request.version,
            "description": request.description if request.description else f"Knowledge plugin for {', '.join(request.domains)}",
            "author": request.author,
            "domains": request.domains,
            "created_at": datetime.now().isoformat(),
            "requires_llm": True,
            "requires_embedding": True,
            "requires_vector_db": request.use_vector_db,
            "requires_graph_db": request.use_graph_db,
            "extraction_stats": extraction_stats,
            "chunk_settings": {
                "chunk_size": request.chunk_size,
                "chunk_overlap": request.chunk_overlap
            },
            "domain_strategy": request.domain_strategy.value
        }
        
        manifest_path = plugin_dir / "manifest.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False)
        
    async def _generate_provider(self, plugin_dir: Path, request: PluginGenerationRequest, processed_data: ProcessedData) -> None:
        """Generate provider.py file."""
        provider_content = f'''"""Knowledge provider for {request.plugin_name}."""

from typing import Dict, List, Any, Optional
import json
from pathlib import Path

from flowlib.providers.knowledge.base import KnowledgeProvider
from flowlib.providers.knowledge.models.plugin import PluginInfo, PluginCapabilities


class {request.plugin_name.title().replace('_', '')}Provider(KnowledgeProvider):
    """Knowledge provider for {request.plugin_name}."""
    
    def __init__(self, plugin_dir: Path):
        self.plugin_dir = plugin_dir
        self.data_dir = plugin_dir / "data"
        self._load_data()
    
    def _load_data(self):
        """Load plugin data from files."""
        with open(self.data_dir / "documents.json", 'r') as f:
            self.documents = json.load(f)
        
        with open(self.data_dir / "entities.json", 'r') as f:
            self.entities = json.load(f)
        
        with open(self.data_dir / "relationships.json", 'r') as f:
            self.relationships = json.load(f)
        
        with open(self.data_dir / "chunks.json", 'r') as f:
            self.chunks = json.load(f)
        
        with open(self.data_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
    
    def get_plugin_info(self) -> PluginInfo:
        """Get plugin information."""
        return PluginInfo(
            name="{request.plugin_name}",
            version="{request.version}",
            description="{request.description or 'Knowledge plugin for ' + ', '.join(request.domains)}",
            author="{request.author}",
            domains={request.domains},
            capabilities=PluginCapabilities(
                can_query=True,
                can_analyze=True,
                has_vector_search={str(request.use_vector_db).lower()},
                has_graph_queries={str(request.use_graph_db).lower()}
            )
        )
    
    async def query_knowledge(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query knowledge from the plugin."""
        # Simple text-based search for now
        results = {{
            "documents": [],
            "entities": [],
            "relationships": []
        }}
        
        query_lower = query.lower()
        
        # Search documents
        for doc in self.documents:
            content = doc["content"] if "content" in doc else ""
            if query_lower in content.lower():
                results["documents"].append(doc)
        
        # Search entities
        for entity in self.entities:
            name = entity["name"] if "name" in entity else ""
            if query_lower in name.lower():
                results["entities"].append(entity)
        
        # Search relationships
        for rel in self.relationships:
            relationship = rel["relationship"] if "relationship" in rel else ""
            if query_lower in relationship.lower():
                results["relationships"].append(rel)
        
        return results
    
    async def analyze_content(self, content: str, analysis_type: str = "entities") -> Dict[str, Any]:
        """Analyze content using plugin knowledge."""
        # Basic content analysis
        return {{
            "analysis_type": analysis_type,
            "content_length": len(content),
            "related_entities": [],
            "suggested_relationships": []
        }}
'''
        
        provider_path = plugin_dir / "provider.py"
        with open(provider_path, 'w') as f:
            f.write(provider_content)
        
    async def _generate_init_file(self, plugin_dir: Path, request: PluginGenerationRequest) -> None:
        """Generate __init__.py file."""
        init_content = f'''"""Knowledge plugin: {request.plugin_name}."""

from .provider import {request.plugin_name.title().replace('_', '')}Provider

__version__ = "{request.version}"
__author__ = "{request.author}"
__description__ = "{request.description or 'Knowledge plugin for ' + ', '.join(request.domains)}"

__all__ = [
    "{request.plugin_name.title().replace('_', '')}Provider"
]
'''
        
        init_path = plugin_dir / "__init__.py"
        with open(init_path, 'w') as f:
            f.write(init_content)
        
    async def _generate_readme(self, plugin_dir: Path, request: PluginGenerationRequest, extraction_stats: Dict[str, Any]) -> None:
        """Generate README.md file."""
        readme_content = f'''# {request.plugin_name.title().replace('_', ' ')} Knowledge Plugin

{request.description or 'Knowledge plugin for ' + ', '.join(request.domains)}

## Overview

This plugin provides knowledge access and analysis capabilities for:
{chr(10).join(f'- {domain}' for domain in request.domains)}

## Statistics

- **Documents processed:** {extraction_stats['total_documents'] if 'total_documents' in extraction_stats else 'N/A'}
- **Entities extracted:** {extraction_stats['total_entities'] if 'total_entities' in extraction_stats else 'N/A'}
- **Relationships found:** {extraction_stats['total_relationships'] if 'total_relationships' in extraction_stats else 'N/A'}
- **Chunks created:** {extraction_stats['total_chunks'] if 'total_chunks' in extraction_stats else 'N/A'}

## Features

- Text-based knowledge querying
- Entity and relationship extraction
- Content analysis
{'- Vector database support (ChromaDB)' if request.use_vector_db else ''}
{'- Graph database support (Neo4j)' if request.use_graph_db else ''}

## Usage

```python
from flowlib.providers.knowledge.plugin_manager import PluginManager

# Load the plugin
plugin_manager = PluginManager()
plugin = plugin_manager.load_plugin("{request.plugin_name}")

# Query knowledge
results = await plugin.query_knowledge("your query here")

# Analyze content
analysis = await plugin.analyze_content("content to analyze")
```

## Configuration

- **Chunk size:** {request.chunk_size}
- **Chunk overlap:** {request.chunk_overlap}
- **Domain strategy:** {request.domain_strategy.value}

## Generated by

Flowlib Knowledge Plugin Generator v1.0.0
**Author:** {request.author}
**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
'''
        
        readme_path = plugin_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
    async def _generate_test_script(self, plugin_dir: Path, request: PluginGenerationRequest) -> None:
        """Generate test script for the plugin."""
        test_content = f'''#!/usr/bin/env python3
"""Test script for {request.plugin_name} knowledge plugin."""

import asyncio
from pathlib import Path
from provider import {request.plugin_name.title().replace('_', '')}Provider


async def test_plugin():
    """Test the knowledge plugin."""
    plugin_dir = Path(__file__).parent
    provider = {request.plugin_name.title().replace('_', '')}Provider(plugin_dir)
    
    print(f"Testing {request.plugin_name} plugin...")
    
    # Test plugin info
    info = provider.get_plugin_info()
    print(f"Plugin: " + info.name + " v" + info.version)
    print(f"Description: " + info.description)
    print(f"Domains: " + ", ".join(info.domains))
    
    # Test knowledge query
    print("\\nTesting knowledge query...")
    results = await provider.query_knowledge("test query")
    print("Query results: " + str(len(results['documents'])) + " documents, " + str(len(results['entities'])) + " entities")
    
    # Test content analysis
    print("\\nTesting content analysis...")
    analysis = await provider.analyze_content("This is test content for analysis.")
    print("Analysis: " + analysis['analysis_type'])
    
    print("\\nPlugin test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_plugin())
'''
        
        test_path = plugin_dir / "test_plugin.py"
        with open(test_path, 'w') as f:
            f.write(test_content)
        
    async def _create_database_configs(self, plugin_dir: Path, plugin_name: str, use_vector_db: bool, use_graph_db: bool) -> None:
        """Create database configuration files."""
        import yaml
        
        if use_vector_db:
            chromadb_config = {
                "provider_type": "chromadb",
                "collection_name": f"{plugin_name}_vectors",
                "persist_directory": f"./data/chroma_{plugin_name}",
                "embedding_function": "default",
                "metadata_fields": ["source", "chunk_id", "domain"]
            }
            
            chromadb_path = plugin_dir / "chromadb_config.yaml"
            with open(chromadb_path, 'w') as f:
                yaml.dump(chromadb_config, f, default_flow_style=False)
        
        if use_graph_db:
            neo4j_config = {
                "provider_type": "neo4j",
                "uri": "bolt://localhost:7687",
                "database": f"{plugin_name}_graph",
                "node_labels": ["Entity", "Document", "Chunk"],
                "relationship_types": ["RELATED_TO", "CONTAINS", "REFERENCES"]
            }
            
            neo4j_path = plugin_dir / "neo4j_config.yaml"
            with open(neo4j_path, 'w') as f:
                yaml.dump(neo4j_config, f, default_flow_style=False)
        
    async def _create_embedded_databases(self, plugin_dir: Path, temp_dir: Path, plugin_name: str, use_vector_db: bool, use_graph_db: bool) -> None:
        """Copy database files from extraction to create embedded databases in plugin."""
        import shutil
        
        if use_vector_db:
            # Copy ChromaDB data if it exists
            chroma_source = temp_dir / "chroma_data"
            if chroma_source.exists():
                chroma_dest = plugin_dir / "data" / f"chroma_{plugin_name}"
                chroma_dest.mkdir(parents=True, exist_ok=True)
                shutil.copytree(chroma_source, chroma_dest, dirs_exist_ok=True)
                print(f"Copied ChromaDB data to {chroma_dest}")
        
        if use_graph_db:
            # Copy Neo4j data if it exists
            neo4j_source = temp_dir / "neo4j_data"
            if neo4j_source.exists():
                neo4j_dest = plugin_dir / "data" / f"neo4j_{plugin_name}"
                neo4j_dest.mkdir(parents=True, exist_ok=True)
                shutil.copytree(neo4j_source, neo4j_dest, dirs_exist_ok=True)
                print(f"Copied Neo4j data to {neo4j_dest}")
        
    async def _create_data_files(self, plugin_dir: Path, processed_data: ProcessedData) -> None:
        """Create data files for the plugin."""
        # Save documents
        data_dir = plugin_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        with open(data_dir / "documents.json", 'w') as f:
            json.dump(processed_data.documents, f, indent=2)
        
        with open(data_dir / "entities.json", 'w') as f:
            json.dump(processed_data.entities, f, indent=2)
        
        with open(data_dir / "relationships.json", 'w') as f:
            json.dump(processed_data.relationships, f, indent=2)
        
        with open(data_dir / "chunks.json", 'w') as f:
            json.dump(processed_data.chunks, f, indent=2)
        
        with open(data_dir / "metadata.json", 'w') as f:
            json.dump(processed_data.metadata, f, indent=2)
