"""Comprehensive tests for knowledge plugin manager module."""

import pytest
import os
import yaml
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock, mock_open
import tempfile
import shutil

from flowlib.providers.knowledge.plugin_manager import KnowledgePluginManager, plugin_manager
from flowlib.providers.knowledge.base import KnowledgeProvider, Knowledge


# Test helper classes
class MockKnowledgeProvider(KnowledgeProvider):
    """Mock knowledge provider for testing."""
    
    def __init__(self, domains: List[str] = None, name: str = "mock_provider"):
        self.domains = domains or ["test_domain"]
        self.name = name
        self._initialized = False
        self._shutdown = False
        self.databases = {}
    
    async def initialize(self, databases: Dict[str, Any] = None):
        """Initialize the provider."""
        self.databases = databases or {}
        self._initialized = True
    
    async def query(self, domain: str, query: str, limit: int = 10) -> List[Knowledge]:
        """Mock query implementation."""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        
        if not self.supports_domain(domain):
            return []
        
        # Return mock results
        return [
            Knowledge(
                domain=domain,
                content=f"Mock result for '{query}' from {self.name}",
                source=self.name,
                confidence=0.8,
                metadata={"query": query}
            )
        ]
    
    def supports_domain(self, domain: str) -> bool:
        """Check if domain is supported."""
        return domain in self.domains
    
    async def shutdown(self):
        """Shutdown the provider."""
        self._shutdown = True


class MockFailingProvider(KnowledgeProvider):
    """Mock provider that fails for testing error handling."""
    
    def __init__(self, fail_on: str = "initialize"):
        self.domains = ["failing_domain"]
        self.fail_on = fail_on
    
    async def initialize(self, databases: Dict[str, Any] = None):
        if self.fail_on == "initialize":
            raise Exception("Initialization failed")
    
    async def query(self, domain: str, query: str, limit: int = 10) -> List[Knowledge]:
        if self.fail_on == "query":
            raise Exception("Query failed")
        return []
    
    def supports_domain(self, domain: str) -> bool:
        return domain in self.domains
    
    async def shutdown(self):
        if self.fail_on == "shutdown":
            raise Exception("Shutdown failed")


class TestKnowledgePluginManager:
    """Test KnowledgePluginManager class."""
    
    def test_plugin_manager_creation(self):
        """Test creating plugin manager instance."""
        manager = KnowledgePluginManager()
        
        assert manager.loaded_plugins == {}
        assert manager.plugin_configs == {}
        assert manager._initialized is False
    
    @pytest.mark.asyncio
    async def test_plugin_manager_initialize_basic(self):
        """Test basic initialization."""
        manager = KnowledgePluginManager()
        
        with patch.object(manager, 'discover_and_load_plugins', new_callable=AsyncMock) as mock_discover:
            await manager.initialize()
            
            assert manager._initialized is True
            mock_discover.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_plugin_manager_initialize_already_initialized(self):
        """Test that initialize is idempotent."""
        manager = KnowledgePluginManager()
        manager._initialized = True
        
        with patch.object(manager, 'discover_and_load_plugins', new_callable=AsyncMock) as mock_discover:
            await manager.initialize()
            
            # Should not call discover again
            mock_discover.assert_not_called()
    
    def test_get_plugin_discovery_paths_basic(self):
        """Test getting plugin discovery paths."""
        manager = KnowledgePluginManager()
        
        with patch('flowlib.providers.knowledge.plugin_manager.Path') as mock_path_class:
            # Create mock Path objects that support the / operator
            mock_builtin_path = Mock()
            mock_user_home = Mock()
            mock_user_path = Mock()
            
            # Mock Path(__file__).parent.parent.parent 
            mock_file_path = Mock()
            mock_file_path.parent.parent.parent.__truediv__ = Mock(return_value=mock_builtin_path)
            
            # Mock Path.home() - need to support chained / operators
            mock_path_class.home.return_value = mock_user_home
            mock_flowlib_path = Mock()
            mock_user_home.__truediv__ = Mock(return_value=mock_flowlib_path)
            mock_flowlib_path.__truediv__ = Mock(return_value=mock_user_path)
            
            # Mock Path constructor
            def path_constructor(arg):
                if str(arg).endswith('.py'):  # This is __file__
                    return mock_file_path
                return Mock()
            
            mock_path_class.side_effect = path_constructor
            
            with patch.dict(os.environ, {}, clear=True):
                paths = manager._get_plugin_discovery_paths()
                
                # Should include at least builtin and user paths
                assert len(paths) >= 2
    
    def test_get_plugin_discovery_paths_with_env_var(self):
        """Test getting paths with environment variable."""
        manager = KnowledgePluginManager()
        
        env_paths = "/custom/path1:/custom/path2"
        with patch.dict(os.environ, {"FLOWLIB_KNOWLEDGE_PLUGINS": env_paths}):
            with patch('flowlib.providers.knowledge.plugin_manager.Path') as mock_path_class:
                # Create mock Path objects that support the / operator
                mock_builtin_path = Mock()
                mock_user_home = Mock()
                mock_flowlib_path = Mock()
                mock_user_path = Mock()
                
                # Mock Path(__file__).parent.parent.parent 
                mock_file_path = Mock()
                mock_file_path.parent.parent.parent.__truediv__ = Mock(return_value=mock_builtin_path)
                
                # Mock Path.home() - need to support chained / operators
                mock_path_class.home.return_value = mock_user_home
                mock_user_home.__truediv__ = Mock(return_value=mock_flowlib_path)
                mock_flowlib_path.__truediv__ = Mock(return_value=mock_user_path)
                
                # Mock environment variable paths
                mock_env_path1 = Mock()
                mock_env_path2 = Mock()
                
                # Mock Path constructor
                def path_constructor(arg):
                    if str(arg).endswith('.py'):  # This is __file__
                        return mock_file_path
                    elif arg == "/custom/path1":
                        return mock_env_path1
                    elif arg == "/custom/path2":
                        return mock_env_path2
                    return Mock()
                
                mock_path_class.side_effect = path_constructor
                
                paths = manager._get_plugin_discovery_paths()
                
                # Should include environment paths
                assert len(paths) >= 4  # builtin + user + 2 env paths
    
    def test_get_plugin_discovery_paths_with_project_config(self):
        """Test getting paths with project configuration."""
        manager = KnowledgePluginManager()
        
        project_config_content = {
            "plugins": {
                "test_plugin": {
                    "enabled": True,
                    "path": "/project/plugin/path"
                },
                "disabled_plugin": {
                    "enabled": False,
                    "path": "/disabled/path"
                }
            }
        }
        
        with patch('flowlib.providers.knowledge.plugin_manager.Path') as mock_path_class:
            # Create mock Path objects that support the / operator
            mock_builtin_path = Mock()
            mock_user_home = Mock()
            mock_flowlib_path = Mock()
            mock_user_path = Mock()
            mock_project_config = Mock()
            mock_project_config.exists.return_value = True
            
            # Mock Path(__file__).parent.parent.parent
            mock_file_path = Mock()
            mock_file_path.parent.parent.parent.__truediv__ = Mock(return_value=mock_builtin_path)
            
            # Mock Path.home() - need to support chained / operators
            mock_path_class.home.return_value = mock_user_home
            mock_user_home.__truediv__ = Mock(return_value=mock_flowlib_path)
            mock_flowlib_path.__truediv__ = Mock(return_value=mock_user_path)
            
            # Mock Path constructor
            def path_constructor(arg):
                if str(arg).endswith('.py'):  # This is __file__
                    return mock_file_path
                elif arg == "knowledge_plugins.yaml":
                    return mock_project_config
                return Mock(resolve=Mock(return_value=Mock()))
            
            mock_path_class.side_effect = path_constructor
            
            with patch("builtins.open", mock_open(read_data=yaml.dump(project_config_content))):
                with patch('yaml.safe_load', return_value=project_config_content):
                    paths = manager._get_plugin_discovery_paths()
                    
                    # Should include enabled project plugin path
                    assert len(paths) >= 3
    
    def test_get_plugin_priority_default(self):
        """Test getting plugin priority with default value."""
        manager = KnowledgePluginManager()
        
        manifest_content = {"name": "test_plugin"}
        plugin_path = Mock(spec=Path)
        manifest_file = Mock()
        plugin_path.__truediv__ = Mock(return_value=manifest_file)
        
        with patch("builtins.open", mock_open(read_data=yaml.dump(manifest_content))):
            with patch('yaml.safe_load', return_value=manifest_content):
                priority = manager._get_plugin_priority(plugin_path)
                
                assert priority == 50  # default priority
    
    def test_get_plugin_priority_custom(self):
        """Test getting plugin priority with custom value."""
        manager = KnowledgePluginManager()
        
        manifest_content = {"name": "test_plugin", "priority": 10}
        plugin_path = Mock(spec=Path)
        manifest_file = Mock()
        plugin_path.__truediv__ = Mock(return_value=manifest_file)
        
        with patch("builtins.open", mock_open(read_data=yaml.dump(manifest_content))):
            with patch('yaml.safe_load', return_value=manifest_content):
                priority = manager._get_plugin_priority(plugin_path)
                
                assert priority == 10
    
    def test_get_plugin_priority_error(self):
        """Test getting plugin priority with error."""
        manager = KnowledgePluginManager()
        
        plugin_path = Mock()
        
        with patch("builtins.open", side_effect=FileNotFoundError):
            priority = manager._get_plugin_priority(plugin_path)
            
            assert priority == 100  # low priority for malformed plugins
    
    @pytest.mark.asyncio
    async def test_discover_and_load_plugins_basic(self):
        """Test basic plugin discovery and loading."""
        manager = KnowledgePluginManager()
        
        mock_paths = [Mock(spec=Path)]
        mock_paths[0].exists.return_value = True
        mock_plugin_dir = Mock(spec=Path)
        mock_plugin_dir.is_dir.return_value = True
        mock_plugin_dir.name = "test_plugin"
        mock_manifest = Mock()
        mock_manifest.exists.return_value = True
        mock_plugin_dir.__truediv__ = Mock(return_value=mock_manifest)
        mock_paths[0].iterdir.return_value = [mock_plugin_dir]
        
        with patch.object(manager, '_get_plugin_discovery_paths', return_value=mock_paths):
            with patch.object(manager, '_get_plugin_priority', return_value=50):
                with patch.object(manager, '_load_plugin', new_callable=AsyncMock, return_value=True) as mock_load:
                    await manager.discover_and_load_plugins()
                    
                    mock_load.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_discover_and_load_plugins_with_errors(self):
        """Test plugin discovery with loading errors."""
        manager = KnowledgePluginManager()
        
        mock_paths = [Mock(spec=Path)]
        mock_paths[0].exists.return_value = True
        mock_plugin_dir = Mock(spec=Path)
        mock_plugin_dir.is_dir.return_value = True
        mock_plugin_dir.name = "test_plugin"
        mock_manifest = Mock()
        mock_manifest.exists.return_value = True
        mock_plugin_dir.__truediv__ = Mock(return_value=mock_manifest)
        mock_paths[0].iterdir.return_value = [mock_plugin_dir]
        
        with patch.object(manager, '_get_plugin_discovery_paths', return_value=mock_paths):
            with patch.object(manager, '_get_plugin_priority', return_value=50):
                with patch.object(manager, '_load_plugin', new_callable=AsyncMock, side_effect=Exception("Load error")):
                    # Should not raise exception, just log error
                    await manager.discover_and_load_plugins()
    
    @pytest.mark.asyncio
    async def test_load_plugin_success(self):
        """Test successful plugin loading."""
        manager = KnowledgePluginManager()
        
        plugin_path = Mock(spec=Path)
        manifest_file = Mock()
        plugin_path.__truediv__ = Mock(return_value=manifest_file)
        manifest_content = {
            "name": "test_plugin",
            "auto_load": True,
            "domains": ["test_domain"],
            "provider_class": "TestProvider",
            "databases": {}
        }
        
        mock_provider_class = Mock()
        mock_provider = MockKnowledgeProvider(["test_domain"], "test_plugin")
        mock_provider_class.return_value = mock_provider
        
        with patch("builtins.open", mock_open(read_data=yaml.dump(manifest_content))):
            with patch('yaml.safe_load', return_value=manifest_content):
                with patch.object(manager, '_load_database_configs', new_callable=AsyncMock, return_value={}):
                    with patch.object(manager, '_import_provider_class', return_value=mock_provider_class):
                        result = await manager._load_plugin(plugin_path)
                        
                        assert result is True
                        assert "test_plugin" in manager.loaded_plugins
                        assert manager.loaded_plugins["test_plugin"] == mock_provider
                        assert "test_plugin" in manager.plugin_configs
    
    @pytest.mark.asyncio
    async def test_load_plugin_auto_load_false(self):
        """Test plugin loading with auto_load=false."""
        manager = KnowledgePluginManager()
        
        plugin_path = Mock(spec=Path)
        manifest_content = {
            "name": "test_plugin",
            "auto_load": False,
            "provider_class": "TestProvider"
        }
        
        with patch("builtins.open", mock_open(read_data=yaml.dump(manifest_content))):
            with patch('yaml.safe_load', return_value=manifest_content):
                result = await manager._load_plugin(plugin_path)
                
                assert result is False
                assert "test_plugin" not in manager.loaded_plugins
    
    @pytest.mark.asyncio
    async def test_load_plugin_already_loaded(self):
        """Test loading plugin that's already loaded."""
        manager = KnowledgePluginManager()
        manager.loaded_plugins["test_plugin"] = Mock()
        
        plugin_path = Mock(spec=Path)
        manifest_content = {
            "name": "test_plugin",
            "provider_class": "TestProvider"
        }
        
        with patch("builtins.open", mock_open(read_data=yaml.dump(manifest_content))):
            with patch('yaml.safe_load', return_value=manifest_content):
                result = await manager._load_plugin(plugin_path)
                
                assert result is False
    
    @pytest.mark.asyncio
    async def test_load_database_configs_chromadb(self):
        """Test loading ChromaDB configuration."""
        manager = KnowledgePluginManager()
        
        plugin_path = Mock(spec=Path)
        manifest = {
            "databases": {
                "chromadb": {
                    "enabled": True,
                    "config_file": "chromadb_config.yaml"
                }
            }
        }
        
        chromadb_config = {"host": "localhost", "port": 8000}
        config_path = Mock()
        config_path.exists.return_value = True
        plugin_path.__truediv__ = Mock(return_value=config_path)
        
        with patch("builtins.open", mock_open(read_data=yaml.dump(chromadb_config))):
            with patch('yaml.safe_load', return_value=chromadb_config):
                databases = await manager._load_database_configs(plugin_path, manifest)
                
                assert "chromadb" in databases
                assert databases["chromadb"] == chromadb_config
    
    @pytest.mark.asyncio
    async def test_load_database_configs_neo4j(self):
        """Test loading Neo4j configuration."""
        manager = KnowledgePluginManager()
        
        plugin_path = Mock(spec=Path)
        manifest = {
            "databases": {
                "neo4j": {
                    "enabled": True,
                    "config_file": "neo4j_config.yaml"
                }
            }
        }
        
        neo4j_config = {"uri": "bolt://localhost:7687", "user": "neo4j"}
        config_path = Mock()
        config_path.exists.return_value = True
        plugin_path.__truediv__ = Mock(return_value=config_path)
        
        with patch("builtins.open", mock_open(read_data=yaml.dump(neo4j_config))):
            with patch('yaml.safe_load', return_value=neo4j_config):
                databases = await manager._load_database_configs(plugin_path, manifest)
                
                assert "neo4j" in databases
                assert databases["neo4j"] == neo4j_config
    
    @pytest.mark.asyncio
    async def test_load_database_configs_missing_file(self):
        """Test loading database config with missing file."""
        manager = KnowledgePluginManager()
        
        plugin_path = Mock(spec=Path)
        manifest = {
            "databases": {
                "chromadb": {
                    "enabled": True,
                    "config_file": "missing_config.yaml"
                }
            }
        }
        
        config_path = Mock()
        config_path.exists.return_value = False
        plugin_path.__truediv__ = Mock(return_value=config_path)
        
        # Should not raise exception, just log warning
        databases = await manager._load_database_configs(plugin_path, manifest)
        
        assert "chromadb" not in databases
    
    def test_import_provider_class_success(self):
        """Test successful provider class import."""
        manager = KnowledgePluginManager()
        
        plugin_path = Mock(spec=Path)
        manifest = {
            "name": "test_plugin",
            "provider_class": "TestProvider"
        }
        
        provider_file = Mock()
        provider_file.exists.return_value = True
        plugin_path.__truediv__ = Mock(return_value=provider_file)
        
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_module = Mock()
        mock_provider_class = Mock()
        mock_module.TestProvider = mock_provider_class
        
        with patch('importlib.util.spec_from_file_location', return_value=mock_spec):
            with patch('importlib.util.module_from_spec', return_value=mock_module):
                with patch.object(mock_loader, 'exec_module'):
                    provider_class = manager._import_provider_class(plugin_path, manifest)
                    
                    assert provider_class == mock_provider_class
    
    def test_import_provider_class_missing_file(self):
        """Test importing provider class with missing file."""
        manager = KnowledgePluginManager()
        
        plugin_path = Mock(spec=Path)
        manifest = {"provider_class": "TestProvider"}
        
        provider_file = Mock()
        provider_file.exists.return_value = False
        plugin_path.__truediv__ = Mock(return_value=provider_file)
        
        with pytest.raises(ImportError) as exc_info:
            manager._import_provider_class(plugin_path, manifest)
        
        assert "Provider file not found" in str(exc_info.value)
    
    def test_import_provider_class_missing_class(self):
        """Test importing provider class when class doesn't exist."""
        manager = KnowledgePluginManager()
        
        plugin_path = Mock(spec=Path)
        manifest = {"name": "test_plugin", "provider_class": "MissingProvider"}
        
        provider_file = Mock()
        provider_file.exists.return_value = True
        plugin_path.__truediv__ = Mock(return_value=provider_file)
        
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_module = Mock()
        # Don't add MissingProvider to mock_module
        
        with patch('importlib.util.spec_from_file_location', return_value=mock_spec):
            with patch('importlib.util.module_from_spec', return_value=mock_module):
                with patch.object(mock_loader, 'exec_module'):
                    # Ensure hasattr returns False for the missing class
                    with patch('builtins.hasattr', side_effect=lambda obj, attr: attr != 'MissingProvider'):
                        with pytest.raises(ImportError) as exc_info:
                            manager._import_provider_class(plugin_path, manifest)
                        
                        assert "Provider class MissingProvider not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_query_domain_success(self):
        """Test successful domain query."""
        manager = KnowledgePluginManager()
        manager._initialized = True
        
        # Add mock providers
        provider1 = MockKnowledgeProvider(["test_domain"], "provider1")
        provider2 = MockKnowledgeProvider(["test_domain", "other_domain"], "provider2")
        manager.loaded_plugins = {
            "provider1": provider1,
            "provider2": provider2
        }
        
        await provider1.initialize()
        await provider2.initialize()
        
        results = await manager.query_domain("test_domain", "test query", limit=10)
        
        assert len(results) == 2  # One result from each provider
        assert all(isinstance(r, Knowledge) for r in results)
        assert all(r.domain == "test_domain" for r in results)
    
    @pytest.mark.asyncio
    async def test_query_domain_auto_initialize(self):
        """Test domain query with auto-initialization."""
        manager = KnowledgePluginManager()
        manager._initialized = False
        
        with patch.object(manager, 'initialize', new_callable=AsyncMock) as mock_init:
            with patch.object(manager, 'loaded_plugins', {}):
                results = await manager.query_domain("test_domain", "test query")
                
                mock_init.assert_called_once()
                assert results == []  # No providers
    
    @pytest.mark.asyncio
    async def test_query_domain_empty_domain(self):
        """Test domain query with empty domain."""
        manager = KnowledgePluginManager()
        manager._initialized = True
        
        with pytest.raises(ValueError) as exc_info:
            await manager.query_domain("", "test query")
        
        assert "Domain cannot be empty" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_query_domain_empty_query(self):
        """Test domain query with empty query."""
        manager = KnowledgePluginManager()
        manager._initialized = True
        
        with pytest.raises(ValueError) as exc_info:
            await manager.query_domain("test_domain", "   ")
        
        assert "Query cannot be empty" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_query_domain_no_relevant_plugins(self):
        """Test domain query with no relevant plugins."""
        manager = KnowledgePluginManager()
        manager._initialized = True
        manager.loaded_plugins = {
            "provider1": MockKnowledgeProvider(["other_domain"], "provider1")
        }
        
        results = await manager.query_domain("test_domain", "test query")
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_query_plugin_safely_success(self):
        """Test safe plugin querying with success."""
        manager = KnowledgePluginManager()
        
        provider = MockKnowledgeProvider(["test_domain"], "test_provider")
        await provider.initialize()
        
        results = await manager._query_plugin_safely(
            "test_provider", provider, "test_domain", "test query", 10
        )
        
        assert len(results) == 1
        assert results[0].content == "Mock result for 'test query' from test_provider"
    
    @pytest.mark.asyncio
    async def test_query_plugin_safely_error(self):
        """Test safe plugin querying with error."""
        manager = KnowledgePluginManager()
        
        provider = MockFailingProvider("query")
        
        results = await manager._query_plugin_safely(
            "failing_provider", provider, "test_domain", "test query", 10
        )
        
        assert results == []  # Should return empty list on error
    
    def test_merge_plugin_results_basic(self):
        """Test basic result merging."""
        manager = KnowledgePluginManager()
        
        results = [
            Knowledge(domain="test", content="Result 1", source="test", confidence=0.9),
            Knowledge(domain="test", content="Result 2", source="test", confidence=0.7),
            Knowledge(domain="test", content="Result 3", source="test", confidence=0.8)
        ]
        
        merged = manager._merge_plugin_results(results, limit=2)
        
        assert len(merged) == 2
        assert merged[0].confidence == 0.9  # Highest confidence first
        assert merged[1].confidence == 0.8
    
    def test_merge_plugin_results_deduplication(self):
        """Test result deduplication."""
        manager = KnowledgePluginManager()
        
        results = [
            Knowledge(domain="test", content="Same result content", source="test", confidence=0.8),
            Knowledge(domain="test", content="Same result content", source="test", confidence=0.9),  # Higher confidence
            Knowledge(domain="test", content="Different result", source="test", confidence=0.7)
        ]
        
        merged = manager._merge_plugin_results(results, limit=10)
        
        assert len(merged) == 2  # Duplicates removed
        # Should keep the one with higher confidence
        same_content_result = next(r for r in merged if "Same result" in r.content)
        assert same_content_result.confidence == 0.9
    
    def test_merge_plugin_results_empty(self):
        """Test merging empty results."""
        manager = KnowledgePluginManager()
        
        merged = manager._merge_plugin_results([], limit=10)
        
        assert merged == []
    
    def test_get_available_domains(self):
        """Test getting available domains."""
        manager = KnowledgePluginManager()
        
        provider1 = MockKnowledgeProvider(["domain1", "domain2"], "provider1")
        provider2 = MockKnowledgeProvider(["domain2", "domain3"], "provider2")
        manager.loaded_plugins = {
            "provider1": provider1,
            "provider2": provider2
        }
        
        domains = manager.get_available_domains()
        
        assert set(domains) == {"domain1", "domain2", "domain3"}
        assert domains == sorted(domains)  # Should be sorted
    
    def test_get_plugin_info(self):
        """Test getting plugin information."""
        manager = KnowledgePluginManager()
        
        manager.plugin_configs = {
            "plugin1": {
                "domains": ["domain1"],
                "description": "Test plugin 1",
                "version": "1.0.0",
                "databases": {"chromadb": {"enabled": True}},
                "priority": 10
            },
            "plugin2": {
                "domains": ["domain2"],
                "description": "Test plugin 2", 
                "version": "2.0.0"
            }
        }
        
        info = manager.get_plugin_info()
        
        assert len(info) == 2
        assert info["plugin1"]["domains"] == ["domain1"]
        assert info["plugin1"]["priority"] == 10
        assert info["plugin1"]["databases"] == ["chromadb"]
        assert info["plugin2"]["priority"] == 50  # default
        assert info["plugin2"]["databases"] == []
    
    def test_get_domain_plugins(self):
        """Test getting plugins for specific domain."""
        manager = KnowledgePluginManager()
        
        provider1 = MockKnowledgeProvider(["domain1", "domain2"], "provider1")
        provider2 = MockKnowledgeProvider(["domain2", "domain3"], "provider2")
        provider3 = MockKnowledgeProvider(["domain3"], "provider3")
        
        manager.loaded_plugins = {
            "provider1": provider1,
            "provider2": provider2,
            "provider3": provider3
        }
        
        domain2_plugins = manager.get_domain_plugins("domain2")
        
        assert set(domain2_plugins) == {"provider1", "provider2"}
    
    @pytest.mark.asyncio
    async def test_shutdown_success(self):
        """Test successful shutdown."""
        manager = KnowledgePluginManager()
        manager._initialized = True
        
        provider1 = MockKnowledgeProvider(["domain1"], "provider1")
        provider2 = MockKnowledgeProvider(["domain2"], "provider2")
        
        manager.loaded_plugins = {
            "provider1": provider1,
            "provider2": provider2
        }
        manager.plugin_configs = {
            "provider1": {},
            "provider2": {}
        }
        
        await manager.shutdown()
        
        assert manager.loaded_plugins == {}
        assert manager.plugin_configs == {}
        assert manager._initialized is False
        assert provider1._shutdown is True
        assert provider2._shutdown is True
    
    @pytest.mark.asyncio
    async def test_shutdown_with_errors(self):
        """Test shutdown with provider errors."""
        manager = KnowledgePluginManager()
        manager._initialized = True
        
        failing_provider = MockFailingProvider("shutdown")
        normal_provider = MockKnowledgeProvider(["domain1"], "normal")
        
        manager.loaded_plugins = {
            "failing": failing_provider,
            "normal": normal_provider
        }
        manager.plugin_configs = {
            "failing": {},
            "normal": {}
        }
        
        # Should not raise exception
        await manager.shutdown()
        
        assert manager.loaded_plugins == {}
        assert manager.plugin_configs == {}
        assert manager._initialized is False
    
    @pytest.mark.asyncio
    async def test_shutdown_plugin_safely_success(self):
        """Test safe plugin shutdown with success."""
        manager = KnowledgePluginManager()
        
        provider = MockKnowledgeProvider(["domain1"], "test_provider")
        
        await manager._shutdown_plugin_safely("test_provider", provider)
        
        assert provider._shutdown is True
    
    @pytest.mark.asyncio
    async def test_shutdown_plugin_safely_error(self):
        """Test safe plugin shutdown with error."""
        manager = KnowledgePluginManager()
        
        provider = MockFailingProvider("shutdown")
        
        # Should not raise exception
        await manager._shutdown_plugin_safely("failing_provider", provider)


class TestKnowledgePluginManagerIntegration:
    """Test integration aspects of the plugin manager."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete plugin manager workflow."""
        # Create temporary plugin directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            plugin_dir = temp_path / "test_plugin"
            plugin_dir.mkdir()
            
            # Create manifest
            manifest = {
                "name": "test_plugin",
                "version": "1.0.0",
                "description": "Test plugin",
                "domains": ["test_domain"],
                "provider_class": "TestProvider",
                "priority": 10,
                "databases": {}
            }
            
            with open(plugin_dir / "manifest.yaml", "w") as f:
                yaml.dump(manifest, f)
            
            # Create provider file content
            provider_content = '''
from flowlib.providers.knowledge.base import KnowledgeProvider, Knowledge

class TestProvider(KnowledgeProvider):
    def __init__(self):
        self.domains = ["test_domain"]
    
    async def initialize(self, databases=None):
        pass
    
    async def query(self, domain, query, limit=10):
        return [Knowledge(domain=domain, content=f"Result for {query}", source="test", confidence=0.8)]
    
    def supports_domain(self, domain):
        return domain in self.domains
    
    async def shutdown(self):
        pass
'''
            
            with open(plugin_dir / "provider.py", "w") as f:
                f.write(provider_content)
            
            # Test the manager
            manager = KnowledgePluginManager()
            
            with patch.object(manager, '_get_plugin_discovery_paths', return_value=[temp_path]):
                await manager.initialize()
                
                # Verify plugin was loaded
                assert "test_plugin" in manager.loaded_plugins
                assert "test_plugin" in manager.plugin_configs
                
                # Test querying
                results = await manager.query_domain("test_domain", "test query")
                assert len(results) > 0
                assert results[0].content == "Result for test query"
                
                # Test domain listing
                domains = manager.get_available_domains()
                assert "test_domain" in domains
                
                # Test plugin info
                info = manager.get_plugin_info()
                assert "test_plugin" in info
                assert info["test_plugin"]["version"] == "1.0.0"
                
                # Test shutdown
                await manager.shutdown()
                assert manager.loaded_plugins == {}
    
    def test_global_plugin_manager_instance(self):
        """Test that global plugin manager instance exists."""
        from flowlib.providers.knowledge.plugin_manager import plugin_manager
        
        assert isinstance(plugin_manager, KnowledgePluginManager)
        assert plugin_manager.loaded_plugins == {}
        assert plugin_manager._initialized is False
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self):
        """Test concurrent queries to multiple plugins."""
        manager = KnowledgePluginManager()
        manager._initialized = True
        
        # Create multiple providers
        providers = {}
        for i in range(5):
            provider = MockKnowledgeProvider([f"domain{i}", "shared_domain"], f"provider{i}")
            await provider.initialize()
            providers[f"provider{i}"] = provider
        
        manager.loaded_plugins = providers
        
        # Query shared domain that all providers support
        results = await manager.query_domain("shared_domain", "concurrent test", limit=10)
        
        assert len(results) == 5  # One result from each provider
        assert all(r.domain == "shared_domain" for r in results)
    
    @pytest.mark.asyncio
    async def test_plugin_priority_ordering(self):
        """Test that plugins are loaded in priority order."""
        manager = KnowledgePluginManager()
        
        # Mock plugin paths with different priorities
        plugin_paths = [Mock(spec=Path) for _ in range(3)]
        priorities = [100, 10, 50]  # Should load in order: 10, 50, 100
        
        expected_order = [plugin_paths[1], plugin_paths[2], plugin_paths[0]]
        
        # Mock the discovery process
        mock_plugin_dir = Mock(spec=Path)
        mock_plugin_dir.exists.return_value = True
        mock_plugin_dir.iterdir.return_value = plugin_paths
        
        for i, path in enumerate(plugin_paths):
            path.is_dir.return_value = True
            path.name = f"plugin_{i}"
            manifest = Mock()
            manifest.exists.return_value = True
            path.__truediv__ = Mock(return_value=manifest)
        
        with patch.object(manager, '_get_plugin_discovery_paths', return_value=[mock_plugin_dir]):
            with patch.object(manager, '_get_plugin_priority', side_effect=priorities):
                with patch.object(manager, '_load_plugin', new_callable=AsyncMock, return_value=True) as mock_load:
                    await manager.discover_and_load_plugins()
                    
                    # Verify plugins were loaded in priority order
                    call_args = [call[0][0] for call in mock_load.call_args_list]
                    assert call_args == expected_order
    
    def test_path_deduplication(self):
        """Test that duplicate paths are removed."""
        manager = KnowledgePluginManager()
        
        # Mock paths with duplicates
        with patch('flowlib.providers.knowledge.plugin_manager.Path') as mock_path_class:
            # Create mock paths that resolve to same absolute path
            same_absolute_path = Mock()
            
            # Create mock Path objects that support the / operator
            mock_builtin_path = Mock()
            mock_builtin_path.resolve.return_value = same_absolute_path
            mock_user_home = Mock()
            mock_user_path = Mock()
            mock_user_path.resolve.return_value = same_absolute_path
            
            # Mock Path(__file__).parent.parent.parent
            mock_file_path = Mock()
            mock_file_path.parent.parent.parent.__truediv__ = Mock(return_value=mock_builtin_path)
            
            # Mock Path.home() - need to support chained / operators
            mock_path_class.home.return_value = mock_user_home
            mock_flowlib_path = Mock()
            mock_user_home.__truediv__ = Mock(return_value=mock_flowlib_path)
            mock_flowlib_path.__truediv__ = Mock(return_value=mock_user_path)
            
            # Mock Path constructor
            def path_constructor(arg):
                if str(arg).endswith('.py'):  # This is __file__
                    return mock_file_path
                return Mock(resolve=Mock(return_value=same_absolute_path))
            
            mock_path_class.side_effect = path_constructor
            
            with patch.dict(os.environ, {}, clear=True):
                paths = manager._get_plugin_discovery_paths()
                
                # Should deduplicate paths with same absolute path
                resolved_paths = [p.resolve() for p in paths if hasattr(p, 'resolve')]
                unique_absolute_paths = set(str(p) for p in resolved_paths)
                assert len(unique_absolute_paths) <= 2  # At most builtin and user paths