"""Checkpoint management for streaming knowledge extraction."""

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml  # type: ignore[import-untyped]

from flowlib.knowledge.models import ExtractionState, PluginManifest

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages streaming checkpoints and incremental plugin exports."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.streaming_dir = self.output_dir / "streaming"
        self.checkpoints_dir = self.streaming_dir / "checkpoints"
        self.incremental_plugins_dir = self.streaming_dir / "incremental_plugins"
        self.state_file = self.checkpoints_dir / "streaming_state.json"

        # Create directories
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.incremental_plugins_dir.mkdir(parents=True, exist_ok=True)

        # Persistent database directories
        self.streaming_vector_db_dir = self.streaming_dir / "vector_db"
        self.streaming_graph_db_dir = self.streaming_dir / "graph_db"

        self.streaming_vector_db_dir.mkdir(parents=True, exist_ok=True)
        self.streaming_graph_db_dir.mkdir(parents=True, exist_ok=True)

    async def load_streaming_state(self) -> Optional[ExtractionState]:
        """Load existing streaming state if available."""

        if not self.state_file.exists():
            logger.info("No existing streaming state found")
            return None

        try:
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)

            # Convert back to ExtractionState
            streaming_state = ExtractionState(**state_data)

            # Update database paths to current locations
            streaming_state.streaming_vector_db_path = str(self.streaming_vector_db_dir)
            streaming_state.streaming_graph_db_path = str(self.streaming_graph_db_dir)

            logger.info(f"Loaded streaming state: {len(streaming_state.processed_docs)} docs processed")
            logger.info(f"Progress: {streaming_state.progress.progress_percentage:.1f}%")

            return streaming_state

        except Exception as e:
            logger.error(f"Failed to load streaming state: {e}")
            return None

    async def save_streaming_state(self, state: ExtractionState) -> None:
        """Save current streaming state."""

        try:
            # Update timestamp
            state.progress.last_update = datetime.now()

            # Convert sets to lists for JSON serialization
            state_dict = state.model_dump()
            state_dict['processed_docs'] = list(state.processed_docs)
            state_dict['detected_domains'] = list(state.detected_domains)

            # Save state file
            with open(self.state_file, 'w') as f:
                json.dump(state_dict, f, indent=2, default=str)

            logger.debug("Streaming state saved successfully")

        except Exception as e:
            logger.error(f"Failed to save streaming state: {e}")
            raise

    async def should_export_checkpoint_plugin(self, state: ExtractionState) -> bool:
        """Determine if it's time to export a checkpoint plugin."""

        docs_since_checkpoint = len(state.processed_docs) - state.last_checkpoint_at
        return docs_since_checkpoint >= state.extraction_config.checkpoint_interval

    async def export_incremental_plugin(self, state: ExtractionState) -> str:
        """Export incremental plugin using existing plugin export system."""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        doc_count = len(state.processed_docs)
        plugin_name = f"streaming_checkpoint_{doc_count}_{timestamp}"

        logger.info(f"Exporting incremental plugin: {plugin_name}")

        # Create plugin directory
        plugin_dir = self.incremental_plugins_dir / plugin_name
        plugin_dir.mkdir(exist_ok=True)

        # Create data directory for JSON exports
        data_dir = plugin_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Create databases directory
        databases_dir = plugin_dir / "databases"
        databases_dir.mkdir(exist_ok=True)

        # Export current accumulated data as JSON
        await self._export_json_data(data_dir, state)

        # Copy streaming databases to plugin
        await self._copy_streaming_databases(databases_dir, state)

        # Create plugin manifest
        manifest = await self._create_checkpoint_manifest(plugin_name, state)
        await self._save_manifest(plugin_dir, manifest)

        # Generate provider code
        provider_code = await self._generate_checkpoint_provider(plugin_name, state)
        await self._save_provider_code(plugin_dir, provider_code)

        # Create database configurations
        await self._create_database_configs(plugin_dir)

        # Update state
        state.last_checkpoint_at = len(state.processed_docs)
        state.checkpoint_plugins.append(str(plugin_dir))

        logger.info(f"Incremental plugin exported: {plugin_dir}")
        logger.info(f"Contains {len(state.accumulated_entities)} entities, {len(state.accumulated_relationships)} relationships")

        return str(plugin_dir)

    async def _export_json_data(self, data_dir: Path, state: ExtractionState) -> None:
        """Export accumulated data as JSON files."""

        # Export entities
        entities_data = [entity.model_dump() for entity in state.accumulated_entities]
        with open(data_dir / "entities.json", 'w') as f:
            json.dump(entities_data, f, indent=2, default=str)

        # Export relationships
        relationships_data = [rel.model_dump() for rel in state.accumulated_relationships]
        with open(data_dir / "relationships.json", 'w') as f:
            json.dump(relationships_data, f, indent=2, default=str)

        # Export processing metadata
        metadata = {
            "processed_documents": list(state.processed_docs),
            "failed_documents": state.failed_docs,
            "detected_domains": list(state.detected_domains),
            "progress": state.progress.model_dump(),
            "export_timestamp": datetime.now().isoformat()
        }

        with open(data_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    async def _copy_streaming_databases(self, databases_dir: Path, state: ExtractionState) -> None:
        """Copy streaming database files to plugin."""

        # Copy ChromaDB if exists
        vector_db_source = Path(state.streaming_vector_db_path)
        if vector_db_source.exists() and any(vector_db_source.iterdir()):
            vector_db_dest = databases_dir / "chromadb"
            vector_db_dest.mkdir(exist_ok=True)

            # Copy ChromaDB files
            for item in vector_db_source.iterdir():
                if item.is_file():
                    shutil.copy2(item, vector_db_dest)
                elif item.is_dir():
                    shutil.copytree(item, vector_db_dest / item.name, dirs_exist_ok=True)

        # Copy Neo4j if exists
        graph_db_source = Path(state.streaming_graph_db_path)
        if graph_db_source.exists() and any(graph_db_source.iterdir()):
            graph_db_dest = databases_dir / "neo4j"
            graph_db_dest.mkdir(exist_ok=True)

            # Copy Neo4j files
            for item in graph_db_source.iterdir():
                if item.is_file():
                    shutil.copy2(item, graph_db_dest)
                elif item.is_dir():
                    shutil.copytree(item, graph_db_dest / item.name, dirs_exist_ok=True)

    async def _create_checkpoint_manifest(self, plugin_name: str, state: ExtractionState) -> PluginManifest:
        """Create manifest for checkpoint plugin."""

        provider_class = f"{plugin_name.title().replace('_', '')}Provider"

        # Database configuration
        databases = {}
        if Path(state.streaming_vector_db_path).exists():
            databases["chromadb"] = {
                "enabled": True,
                "config_file": "chromadb_config.yaml"
            }

        if Path(state.streaming_graph_db_path).exists():
            databases["neo4j"] = {
                "enabled": True,
                "config_file": "neo4j_config.yaml"
            }

        # Checkpoint metadata
        checkpoint_info = {
            "is_incremental": True,
            "resumable": True,
            "processed_docs": len(state.processed_docs),
            "total_docs": state.progress.total_documents,
            "progress_percentage": state.progress.progress_percentage,
            "created_at": datetime.now().isoformat(),
            "can_resume_from": True
        }

        return PluginManifest(
            name=plugin_name,
            version="1.0.0",
            description=f"Incremental knowledge checkpoint: {len(state.processed_docs)}/{state.progress.total_documents} documents",
            plugin_type="knowledge_plugin",
            provider_class=provider_class,
            entities_count=len(state.accumulated_entities),
            relationships_count=len(state.accumulated_relationships),
            documents_processed=len(state.processed_docs),
            domains=list(state.detected_domains),
            databases=databases,
            checkpoint=checkpoint_info
        )

    async def _save_manifest(self, plugin_dir: Path, manifest: PluginManifest) -> None:
        """Save plugin manifest."""

        manifest_data = manifest.model_dump()

        with open(plugin_dir / "manifest.yaml", 'w') as f:
            yaml.dump(manifest_data, f, default_flow_style=False)

    async def _generate_checkpoint_provider(self, plugin_name: str, state: ExtractionState) -> str:
        """Generate provider code for checkpoint plugin."""

        class_name = f"{plugin_name.title().replace('_', '')}Provider"
        domains_list = list(state.detected_domains)

        # Build provider code safely without complex f-string templates
        lines = [
            '"""',
            f'Incremental knowledge plugin provider for checkpoint: {plugin_name}',
            f'Generated from streaming extraction of {len(state.processed_docs)} documents.',
            '"""',
            '',
            'import json',
            'import logging',
            'from pathlib import Path',
            'from typing import List, Dict, Any',
            'from flowlib.providers.knowledge.base import KnowledgeProvider, Knowledge',
            '',
            'logger = logging.getLogger(__name__)',
            '',
            '',
            f'class {class_name}(KnowledgeProvider):',
            '    """Incremental knowledge provider from streaming checkpoint."""',
            '',
            '    def __init__(self, plugin_dir: Path):',
            '        super().__init__()',
            '        self.plugin_dir = plugin_dir',
            f'        self.domains = {domains_list}',
            '        self._entities_cache = None',
            '',
            '    def get_description(self) -> str:',
            f'        return "Incremental knowledge provider for checkpoint: {plugin_name}"',
            '',
            '    async def query(self, domain: str, query: str, limit: int = 10) -> List[Knowledge]:',
            '        """Query knowledge from checkpoint data."""',
            '        if domain not in self.domains:',
            '            return []',
            '',
            '        entities = self._load_entities()',
            '        matches = []',
            '        query_lower = query.lower()',
            '',
            '        for entity in entities:',
            '            if "name" in entity and query_lower in entity["name"].lower():',
            '                entity_type = entity.get("entity_type", "unknown")',
            '                entity_desc = entity.get("description", "N/A")',
            '',
            '                content = f"Entity: {entity[\'name\']}\\n"',
            '                content += f"Type: {entity_type}\\n"',
            '                content += f"Description: {entity_desc}"',
            '',
            '                matches.append(Knowledge(',
            '                    content=content,',
            '                    domain=domain,',
            '                    confidence=0.8,',
            '                    source="json_entities",',
            '                    metadata={',
            '                        "entity_id": entity.get("entity_id"),',
            '                        "type": entity.get("entity_type")',
            '                    }',
            '                ))',
            '',
            '                if len(matches) >= limit:',
            '                    break',
            '',
            '        return matches',
            '',
            '    def _load_entities(self) -> List[Dict]:',
            '        """Load entities from checkpoint data."""',
            '        if self._entities_cache is None:',
            '            entities_file = self.plugin_dir / "entities.json"',
            '            if entities_file.exists():',
            '                try:',
            '                    with open(entities_file, "r", encoding="utf-8") as f:',
            '                        self._entities_cache = json.load(f)',
            '                except Exception as e:',
            '                    logger.warning(f"Failed to load entities: {e}")',
            '                    self._entities_cache = []',
            '            else:',
            '                self._entities_cache = []',
            '        return self._entities_cache',
        ]

        return '\n'.join(lines)

    async def _save_provider_code(self, plugin_dir: Path, provider_code: str) -> None:
        """Save provider code to plugin."""

        with open(plugin_dir / "provider.py", 'w') as f:
            f.write(provider_code)

    async def _create_database_configs(self, plugin_dir: Path) -> None:
        """Create database configuration files."""

        import yaml

        # ChromaDB config
        chromadb_config = {
            "persist_directory": "./databases/chromadb",
            "collection_name": "streaming_knowledge",
            "embedding_function": "sentence_transformers"
        }

        with open(plugin_dir / "chromadb_config.yaml", 'w') as f:
            yaml.dump(chromadb_config, f, default_flow_style=False)

        # Neo4j config
        neo4j_config = {
            "database_path": "./databases/neo4j",
            "embedded": True,
            "auto_load": True
        }

        with open(plugin_dir / "neo4j_config.yaml", 'w') as f:
            yaml.dump(neo4j_config, f, default_flow_style=False)

    def create_checksum(self, data: Dict[str, Any]) -> str:
        """Create checksum for data integrity."""

        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()

    async def cleanup_old_checkpoints(self, keep_last: int = 5) -> None:
        """Clean up old checkpoint files to save space."""

        try:
            # Get all checkpoint plugin directories
            plugin_dirs = [d for d in self.incremental_plugins_dir.iterdir() if d.is_dir()]

            # Sort by creation time
            plugin_dirs.sort(key=lambda x: x.stat().st_ctime)

            # Remove old ones, keeping the latest N
            if len(plugin_dirs) > keep_last:
                to_remove = plugin_dirs[:-keep_last]

                for plugin_dir in to_remove:
                    logger.info(f"Removing old checkpoint: {plugin_dir.name}")
                    shutil.rmtree(plugin_dir)

        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")
