"""
Knowledge Plugin Service for GUI integration.

This service uses flowlib flows instead of stub implementations,
providing real knowledge extraction and plugin generation capabilities.
"""

import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime
from pydantic import Field, validator
from flowlib.core.models import StrictBaseModel

from .models import OperationResult, PluginInfo, TestResult
from .error_boundaries import handle_service_errors, ServiceError
from .async_qt_helper import AsyncServiceMixin


class PluginDataFile(StrictBaseModel):
    """Pydantic model for plugin data files with validation."""
    # Inherits strict configuration from StrictBaseModel
    
    documents: List[dict[str, Union[str, int, float, bool]]] = Field(default_factory=list, description="Document data")
    entities: List[dict[str, Union[str, int, float, bool]]] = Field(default_factory=list, description="Entity data")
    relationships: List[dict[str, Union[str, int, float, bool]]] = Field(default_factory=list, description="Relationship data")
    
    @property
    def documents_count(self) -> int:
        return len(self.documents)
    
    @property
    def entities_count(self) -> int:
        return len(self.entities)
    
    @property
    def relationships_count(self) -> int:
        return len(self.relationships)


class PluginStats(StrictBaseModel):
    """Pydantic model for plugin statistics."""
    # Inherits strict configuration from StrictBaseModel
    
    documents_count: int = Field(default=0, description="Number of documents")
    entities_count: int = Field(default=0, description="Number of entities")
    relationships_count: int = Field(default=0, description="Number of relationships")
    has_vector_data: bool = Field(default=False, description="Whether plugin has vector data")
    has_graph_data: bool = Field(default=False, description="Whether plugin has graph data")

logger = logging.getLogger(__name__)


class KnowledgePluginService(AsyncServiceMixin):
    """Real knowledge plugin service using flowlib flows."""
    
    def __init__(self):
        super().__init__()
        self._initialized = False
        self.plugins_directory = Path.home() / ".flowlib" / "knowledge_plugins"
        
    async def initialize(self) -> bool:
        """Initialize the service."""
        try:
            # Ensure plugins directory exists
            self.plugins_directory.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            logger.info("KnowledgePluginService initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgePluginService: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the service."""
        self._initialized = False
    
    @handle_service_errors("generate_plugin")
    async def generate_plugin(self, plugin_config: dict[str, Union[str, int, float, bool]]) -> OperationResult:
        """Generate a knowledge plugin using real flowlib flows."""
        if not self._initialized:
            raise ServiceError("Service not initialized", context={'operation': 'generate_plugin'})
        
        try:
            # Import flowlib components
            from flowlib.knowledge.plugin_generation.flow import PluginGenerationFlow
            from flowlib.knowledge.plugin_generation.models import (
                PluginGenerationRequest, DomainStrategy, DomainStrategyConfig
            )
            
            # Create and validate PluginGenerationRequest using proper Pydantic model
            try:
                request = PluginGenerationRequest(
                    input_directory=plugin_config['input_directory'],
                    output_directory=plugin_config['output_directory'] if 'output_directory' in plugin_config else str(self.plugins_directory),
                    plugin_name=plugin_config['plugin_name'],
                    domains=plugin_config['domains'] if 'domains' in plugin_config else ['general'],
                    description=plugin_config['description'] if 'description' in plugin_config else '',
                    chunk_size=plugin_config['chunk_size'] if 'chunk_size' in plugin_config else 1000,
                    chunk_overlap=plugin_config['chunk_overlap'] if 'chunk_overlap' in plugin_config else 200
                )
            except (KeyError, ValueError, TypeError) as e:
                raise ServiceError(f"Invalid plugin configuration: {e}", context={'plugin_config': plugin_config})
            
            # Verify input directory exists and has files
            input_dir = Path(request.input_directory)
            if not input_dir.exists():
                raise ServiceError(f"Input directory does not exist: {request.input_directory}")
            
            input_files = list(input_dir.glob("*"))
            if not input_files:
                raise ServiceError(f"No files found in input directory: {request.input_directory}")
            
            # Create output directory
            output_dir = Path(request.output_directory) / request.plugin_name
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run plugin generation using proper PluginGenerationFlow
            logger.info(f"Starting plugin generation for '{request.plugin_name}' with {len(input_files)} files")
            flow = PluginGenerationFlow()
            # Run pipeline and convert result to OperationResult format
            result = await flow.run_pipeline(request)
            
            if result.success:
                logger.info(f"Successfully generated plugin '{request.plugin_name}'")
                
                # Return success result with plugin details
                plugin_details = {
                    'name': request.plugin_name,
                    'path': str(output_dir),
                    'description': request.description,
                    'domains': request.domains,
                    'documents_processed': len(input_files),
                    'created_at': datetime.now().isoformat()
                }
                
                # Add extraction stats if available
                if result.summary and result.summary.processed_data_stats:
                    plugin_details['entities_extracted'] = result.summary.processed_data_stats.entities
                    plugin_details['relationships_extracted'] = result.summary.processed_data_stats.relationships
                
                return OperationResult(
                    success=True,
                    data=plugin_details,
                    message=f"Plugin '{request.plugin_name}' generated successfully"
                )
            else:
                error_msg = result.error_message if result.error_message else "Plugin generation failed"
                raise ServiceError(f"Plugin generation failed: {error_msg}", 
                                 context={'plugin_name': request.plugin_name})
                
        except ImportError as e:
            raise ServiceError(f"Flowlib components not available: {e}", 
                             context={'operation': 'generate_plugin'})
        except Exception as e:
            logger.error(f"Plugin generation failed: {e}")
            raise ServiceError(f"Plugin generation failed: {e}", 
                             context={'plugin_name': plugin_config['name'] if 'name' in plugin_config else 'unknown'})
    
    @handle_service_errors("list_plugins")
    async def list_plugins(self) -> OperationResult:
        """List all available knowledge plugins."""
        if not self._initialized:
            raise ServiceError("Service not initialized", context={'operation': 'list_plugins'})
        
        try:
            plugins = []
            
            if self.plugins_directory.exists():
                for plugin_dir in self.plugins_directory.iterdir():
                    if plugin_dir.is_dir():
                        plugin_info = await self._get_plugin_info(plugin_dir)
                        if plugin_info:
                            plugins.append(plugin_info)
            
            return OperationResult(
                success=True,
                data={'plugins': plugins},
                message=f"Found {len(plugins)} plugins"
            )
            
        except Exception as e:
            logger.error(f"Failed to list plugins: {e}")
            raise ServiceError(f"Failed to list plugins: {e}", context={'operation': 'list_plugins'})
    
    async def _get_plugin_info(self, plugin_dir: Path) -> Optional[dict[str, Union[str, int, float, bool]]]:
        """Get information about a plugin from its directory."""
        try:
            manifest_path = plugin_dir / "manifest.yaml"
            if not manifest_path.exists():
                return None
            
            import yaml
            with open(manifest_path, 'r') as f:
                manifest = yaml.safe_load(f)
            
            # Count data files using validated Pydantic model
            data_dir = plugin_dir / "data"
            doc_count = 0
            if data_dir.exists():
                docs_file = data_dir / "documents.json"
                if docs_file.exists():
                    import json
                    try:
                        with open(docs_file, 'r') as f:
                            raw_data = json.load(f)
                        # Use Pydantic validation instead of isinstance() check
                        try:
                            plugin_data = PluginDataFile(documents=raw_data if isinstance(raw_data, list) else [])
                            doc_count = plugin_data.documents_count
                        except Exception:
                            doc_count = 0
                    except Exception:
                        doc_count = 0
            
            return {
                'name': manifest['name'] if 'name' in manifest else plugin_dir.name,
                'version': manifest['version'] if 'version' in manifest else '1.0.0',
                'description': manifest['description'] if 'description' in manifest else '',
                'domain': manifest['domain'] if 'domain' in manifest else 'unknown',
                'domain_strategy': manifest['domain_strategy'] if 'domain_strategy' in manifest else 'generic',
                'status': 'Active',  # Could be enhanced with real status checking
                'documents': doc_count,
                'created': manifest['created_at'] if 'created_at' in manifest else '',
                'path': str(plugin_dir)
            }
            
        except Exception as e:
            logger.warning(f"Failed to read plugin info for {plugin_dir.name}: {e}")
            return None
    
    @handle_service_errors("delete_plugin")
    async def delete_plugin(self, plugin_name: str) -> OperationResult:
        """Delete a knowledge plugin."""
        if not self._initialized:
            raise ServiceError("Service not initialized", context={'operation': 'delete_plugin'})
        
        try:
            plugin_dir = self.plugins_directory / plugin_name
            if not plugin_dir.exists():
                raise ServiceError(f"Plugin '{plugin_name}' not found", 
                                 context={'plugin_name': plugin_name})
            
            shutil.rmtree(plugin_dir)
            logger.info(f"Deleted plugin '{plugin_name}'")
            
            return OperationResult(
                success=True,
                data={'plugin_name': plugin_name},
                message=f"Plugin '{plugin_name}' deleted successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to delete plugin '{plugin_name}': {e}")
            raise ServiceError(f"Failed to delete plugin: {e}", 
                             context={'plugin_name': plugin_name})
    
    @handle_service_errors("get_available_extractors")
    async def get_available_extractors(self) -> OperationResult:
        """Get list of available knowledge extractors."""
        try:
            # Return the available extractors from flowlib
            extractors = [
                "text",
                "streaming", 
                "entity_analysis",
                "concept_extraction"
            ]
            
            return OperationResult(
                success=True,
                data={'extractors': extractors},
                message=f"Found {len(extractors)} extractors"
            )
            
        except Exception as e:
            logger.error(f"Failed to get extractors: {e}")
            raise ServiceError(f"Failed to get extractors: {e}", context={'operation': 'get_extractors'})
    
    @handle_service_errors("validate_plugin")
    async def validate_plugin(self, plugin_name: str) -> OperationResult:
        """Validate a knowledge plugin."""
        if not self._initialized:
            raise ServiceError("Service not initialized", context={'operation': 'validate_plugin'})
        
        try:
            plugin_dir = self.plugins_directory / plugin_name
            if not plugin_dir.exists():
                raise ServiceError(f"Plugin '{plugin_name}' not found", 
                                 context={'plugin_name': plugin_name})
            
            # Basic validation checks
            required_files = ['manifest.yaml', '__init__.py', 'provider.py']
            missing_files = []
            
            for required_file in required_files:
                if not (plugin_dir / required_file).exists():
                    missing_files.append(required_file)
            
            # Check data directory
            data_dir = plugin_dir / "data"
            data_files_exist = data_dir.exists() and any(data_dir.iterdir())
            
            validation_result = {
                'plugin_name': plugin_name,
                'structure_valid': len(missing_files) == 0,
                'missing_files': missing_files,
                'has_data': data_files_exist,
                'validation_passed': len(missing_files) == 0 and data_files_exist
            }
            
            return OperationResult(
                success=True,
                data=validation_result,
                message=f"Plugin validation {'passed' if validation_result['validation_passed'] else 'failed'}"
            )
            
        except Exception as e:
            logger.error(f"Failed to validate plugin '{plugin_name}': {e}")
            raise ServiceError(f"Failed to validate plugin: {e}", 
                             context={'plugin_name': plugin_name})
    
    @handle_service_errors("list_plugins")
    def list_plugins(self) -> List[PluginInfo]:
        """List available knowledge plugins."""
        if not self._initialized:
            raise ServiceError("Service not initialized", context={'operation': 'list_plugins'})
        
        plugins = []
        
        if self.plugins_directory.exists():
            for plugin_path in self.plugins_directory.iterdir():
                if plugin_path.is_dir() and (plugin_path / "manifest.yaml").exists():
                    try:
                        # Read plugin manifest
                        import yaml
                        with open(plugin_path / "manifest.yaml", 'r') as f:
                            manifest = yaml.safe_load(f)
                        
                        plugins.append(PluginInfo(
                            name=plugin_path.name,
                            version=manifest['version'] if 'version' in manifest else '1.0.0',
                            description=manifest['description'] if 'description' in manifest else '',
                            author=manifest['author'] if 'author' in manifest else 'Unknown',
                            enabled=True,
                            path=str(plugin_path)
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to load plugin {plugin_path.name}: {e}")
        
        return plugins
    
    def get_plugin_details(self, plugin_name: str) -> Optional[dict[str, Union[str, int, float, bool]]]:
        """Get detailed plugin information."""
        try:
            plugin_dir = self.plugins_directory / plugin_name
            
            if not plugin_dir.exists():
                return None
            
            # Load plugin manifest
            manifest_path = plugin_dir / "manifest.yaml"
            if not manifest_path.exists():
                return {"name": plugin_name, "error": "No manifest found"}
            
            import yaml
            with open(manifest_path, 'r') as f:
                manifest = yaml.safe_load(f)
            
            # Get file information
            files = []
            for file_path in plugin_dir.rglob('*'):
                if file_path.is_file():
                    files.append({
                        "name": file_path.name,
                        "relative_path": str(file_path.relative_to(plugin_dir)),
                        "size": file_path.stat().st_size,
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
            
            # Get data statistics
            data_stats = self._get_plugin_data_stats(plugin_dir)
            
            return {
                "name": plugin_name,
                "path": str(plugin_dir),
                "manifest": manifest,
                "files": files,
                "file_count": len(files),
                "total_size": sum(f["size"] for f in files),
                "data_statistics": data_stats
            }
            
        except Exception as e:
            return {"name": plugin_name, "error": str(e)}
    
    def _get_plugin_data_stats(self, plugin_dir: Path) -> dict[str, Union[str, int, float, bool]]:
        """Get statistics about plugin data files using validated Pydantic models."""
        try:
            data_dir = plugin_dir / "data"
            if not data_dir.exists():
                return PluginStats().model_dump()
            
            import json
            
            # Initialize data structures
            documents = []
            entities = []
            relationships = []
            
            # Load and validate documents
            docs_file = data_dir / "documents.json"
            if docs_file.exists():
                try:
                    with open(docs_file, 'r') as f:
                        raw_docs = json.load(f)
                    documents = raw_docs if isinstance(raw_docs, list) else []
                except Exception:
                    pass  # Keep empty list for malformed data
            
            # Load and validate entities
            entities_file = data_dir / "entities.json"
            if entities_file.exists():
                try:
                    with open(entities_file, 'r') as f:
                        raw_entities = json.load(f)
                    entities = raw_entities if isinstance(raw_entities, list) else []
                except Exception:
                    pass  # Keep empty list for malformed data
            
            # Load and validate relationships
            relationships_file = data_dir / "relationships.json"
            if relationships_file.exists():
                try:
                    with open(relationships_file, 'r') as f:
                        raw_relationships = json.load(f)
                    relationships = raw_relationships if isinstance(raw_relationships, list) else []
                except Exception:
                    pass  # Keep empty list for malformed data
            
            # Create validated plugin data model
            plugin_data = PluginDataFile(
                documents=documents,
                entities=entities,
                relationships=relationships
            )
            
            # Check for vector and graph data
            has_vector_data = (data_dir / "vectors").exists() or (data_dir / "embeddings.json").exists()
            has_graph_data = (data_dir / "graph").exists() or (data_dir / "graph.json").exists()
            
            # Return validated PluginStats
            return PluginStats(
                documents_count=plugin_data.documents_count,
                entities_count=plugin_data.entities_count,
                relationships_count=plugin_data.relationships_count,
                has_vector_data=has_vector_data,
                has_graph_data=has_graph_data
            ).model_dump()
            
        except Exception as e:
            logger.warning(f"Failed to get data stats for plugin: {e}")
            return PluginStats().model_dump()
    
    def validate_plugin_structure(self, plugin_name: str) -> dict[str, Union[str, int, float, bool]]:
        """Validate plugin structure and files."""
        try:
            plugin_dir = self.plugins_directory / plugin_name
            
            if not plugin_dir.exists():
                return {"valid": False, "errors": ["Plugin directory not found"]}
            
            errors = []
            warnings = []
            
            # Check required files
            required_files = ["manifest.yaml", "__init__.py", "provider.py"]
            for file_name in required_files:
                if not (plugin_dir / file_name).exists():
                    errors.append(f"Missing required file: {file_name}")
            
            # Validate manifest
            manifest_path = plugin_dir / "manifest.yaml"
            if manifest_path.exists():
                try:
                    import yaml
                    with open(manifest_path, 'r') as f:
                        manifest = yaml.safe_load(f)
                    
                    required_fields = ["name", "version", "description"]
                    for field in required_fields:
                        if field not in manifest:
                            errors.append(f"Missing required manifest field: {field}")
                except Exception as e:
                    errors.append(f"Invalid manifest.yaml: {str(e)}")
            
            # Validate Python files
            for py_file in plugin_dir.glob("*.py"):
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                    
                    import ast
                    ast.parse(content)
                except SyntaxError as e:
                    errors.append(f"Syntax error in {py_file.name}: {str(e)}")
                except Exception as e:
                    warnings.append(f"Could not validate {py_file.name}: {str(e)}")
            
            # Check data integrity
            data_issues = self._validate_plugin_data(plugin_dir)
            if data_issues:
                warnings.extend(data_issues)
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "checked_files": len(list(plugin_dir.glob("*.py"))),
                "plugin_path": str(plugin_dir)
            }
            
        except Exception as e:
            return {"valid": False, "errors": [f"Validation error: {str(e)}"]}
    
    def _validate_plugin_data(self, plugin_dir: Path) -> List[str]:
        """Validate plugin data files."""
        issues = []
        
        try:
            data_dir = plugin_dir / "data"
            if not data_dir.exists():
                issues.append("No data directory found")
                return issues
            
            # Check for any data files
            data_files = list(data_dir.glob("*.json"))
            if not data_files:
                issues.append("No data files found in data directory")
            
            # Validate JSON files
            for data_file in data_files:
                try:
                    import json
                    with open(data_file, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    issues.append(f"Invalid JSON in {data_file.name}: {str(e)}")
                except Exception as e:
                    issues.append(f"Cannot read {data_file.name}: {str(e)}")
        
        except Exception as e:
            issues.append(f"Data validation error: {str(e)}")
        
        return issues
    
    def get_plugin_manifest(self, plugin_name: str) -> dict[str, Union[str, int, float, bool]]:
        """Get plugin manifest data."""
        try:
            plugin_dir = self.plugins_directory / plugin_name
            manifest_path = plugin_dir / "manifest.yaml"
            
            if not manifest_path.exists():
                return {}
            
            import yaml
            with open(manifest_path, 'r') as f:
                return yaml.safe_load(f)
                
        except Exception as e:
            logger.error(f"Failed to get manifest for plugin '{plugin_name}': {e}")
            return {}
    
    def update_plugin_manifest(self, plugin_name: str, manifest_data: dict[str, Union[str, int, float, bool]]) -> bool:
        """Update plugin manifest data."""
        try:
            plugin_dir = self.plugins_directory / plugin_name
            manifest_path = plugin_dir / "manifest.yaml"
            
            if not plugin_dir.exists():
                return False
            
            # Add update timestamp
            manifest_data['updated_at'] = datetime.now().isoformat()
            
            import yaml
            with open(manifest_path, 'w') as f:
                yaml.dump(manifest_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Updated manifest for plugin '{plugin_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update manifest for plugin '{plugin_name}': {e}")
            return False