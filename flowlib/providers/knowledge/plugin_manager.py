"""Plugin manager for knowledge providers."""

import asyncio
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from flowlib.providers.knowledge.models.plugin import PluginManifest, ProjectConfig

from .base import Knowledge, KnowledgeProvider

logger = logging.getLogger(__name__)


class KnowledgePluginManager:
    """Manages discovery and loading of knowledge plugins.

    The plugin manager automatically discovers plugins from multiple paths,
    loads them in priority order, and provides a unified interface for
    querying across all loaded plugins.
    """

    def __init__(self) -> None:
        """Initialize the plugin manager."""
        self.loaded_plugins: dict[str, KnowledgeProvider] = {}
        self.plugin_configs: dict[str, dict] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the plugin manager and discover plugins."""
        if self._initialized:
            return

        logger.info("Initializing Knowledge Plugin Manager")
        await self.discover_and_load_plugins()
        self._initialized = True

    async def discover_and_load_plugins(self) -> None:
        """Discover and load all available plugins."""
        plugin_paths = self._get_plugin_discovery_paths()
        logger.info(f"Searching for plugins in {len(plugin_paths)} directories")

        # Collect all plugin manifests
        plugins_to_load = []
        for plugin_dir in plugin_paths:
            if plugin_dir.exists():
                logger.debug(f"Scanning plugin directory: {plugin_dir}")
                for plugin_path in plugin_dir.iterdir():
                    if plugin_path.is_dir():
                        manifest_path = plugin_path / "manifest.yaml"
                        if manifest_path.exists():
                            plugins_to_load.append(plugin_path)
                            logger.debug(f"Found plugin: {plugin_path.name}")

        logger.info(f"Found {len(plugins_to_load)} potential plugins")

        # Sort by priority (lower number = higher priority)
        plugins_to_load.sort(key=lambda p: self._get_plugin_priority(p))

        # Load plugins in priority order
        loaded_count = 0
        for plugin_path in plugins_to_load:
            try:
                success = await self._load_plugin(plugin_path)
                if success:
                    loaded_count += 1
            except Exception as e:
                logger.error(f"Failed to load plugin at {plugin_path}: {e}")

        logger.info(f"Successfully loaded {loaded_count} knowledge plugins")

    def _get_plugin_discovery_paths(self) -> list[Path]:
        """Get all possible plugin discovery paths in priority order."""
        paths = []

        # 1. Built-in plugins (highest priority)
        builtin_path = Path(__file__).parent.parent.parent / "knowledge_plugins"
        paths.append(builtin_path)
        logger.debug(f"Built-in plugin path: {builtin_path}")

        # 2. User plugins
        user_home = Path.home()
        user_path = user_home / ".flowlib" / "knowledge_plugins"
        paths.append(user_path)
        logger.debug(f"User plugin path: {user_path}")

        # 3. Environment variable paths
        env_path = (
            os.environ.get("FLOWLIB_KNOWLEDGE_PLUGINS")
            if "FLOWLIB_KNOWLEDGE_PLUGINS" in os.environ
            else None
        )
        if env_path:
            env_paths = [Path(p.strip()) for p in env_path.split(":") if p.strip()]
            paths.extend(env_paths)
            logger.debug(f"Environment plugin paths: {env_paths}")

        # 4. Project-specific configuration
        project_config = Path("knowledge_plugins.yaml")
        if project_config.exists():
            try:
                with open(project_config) as f:
                    config_raw = yaml.safe_load(f)
                    if "plugins" not in config_raw:
                        logger.warning("Project config missing 'plugins' field")
                        return paths

                    # Validate project config structure
                    project_config_obj = ProjectConfig(**config_raw)

                    for _plugin_name, plugin_config in project_config_obj.plugins.items():
                        if plugin_config.enabled:
                            plugin_path = Path(plugin_config.path).expanduser()
                            paths.append(plugin_path)
                            logger.debug(f"Project plugin path: {plugin_path}")
            except Exception as e:
                logger.warning(f"Failed to load project plugin config: {e}")

        # Remove duplicates while preserving order
        unique_paths = []
        seen = set()
        for path in paths:
            abs_path = path.resolve()
            if abs_path not in seen:
                unique_paths.append(abs_path)
                seen.add(abs_path)

        return unique_paths

    def _get_plugin_priority(self, plugin_path: Path) -> int:
        """Get plugin loading priority from manifest."""
        try:
            with open(plugin_path / "manifest.yaml") as f:
                manifest_raw = yaml.safe_load(f)
                manifest = PluginManifest.model_validate(manifest_raw)
                return manifest.priority
        except Exception as e:
            logger.warning(f"Could not read priority from {plugin_path}: {e}")
            return 100  # Low priority for malformed plugins

    async def _load_plugin(self, plugin_path: Path) -> bool:
        """Load a single plugin.

        Args:
            plugin_path: Path to the plugin directory

        Returns:
            True if plugin loaded successfully, False otherwise
        """
        try:
            # Load and validate manifest
            manifest_path = plugin_path / "manifest.yaml"
            with open(manifest_path) as f:
                manifest_raw = yaml.safe_load(f)

            # Parse manifest using Pydantic model for validation
            manifest = PluginManifest.model_validate(manifest_raw)

            # Skip if auto_load is false
            if not manifest.auto_load:
                logger.info(f"Skipping plugin {manifest.name} (auto_load=false)")
                return False

            # Check if already loaded (higher priority plugin with same name)
            if manifest.name in self.loaded_plugins:
                logger.warning(f"Plugin {manifest.name} already loaded, skipping duplicate")
                return False

            # Load database configurations
            databases = await self._load_database_configs(plugin_path, manifest)

            # Import and instantiate provider
            provider_class = self._import_provider_class(plugin_path, manifest)
            provider = provider_class()

            # Initialize provider with database configs
            await provider.initialize(databases)

            # Register plugin
            self.loaded_plugins[manifest.name] = provider
            self.plugin_configs[manifest.name] = manifest_raw

            logger.info(f"Loaded knowledge plugin: {manifest.name} (domains: {manifest.domains})")
            return True

        except Exception as e:
            logger.error(f"Failed to load plugin at {plugin_path}: {e}")
            return False

    async def _load_database_configs(self, plugin_path: Path, manifest: Any) -> dict[str, Any]:
        """Load database configuration files for the plugin.

        Args:
            plugin_path: Path to plugin directory
            manifest: Plugin manifest (PluginManifest object or dict)

        Returns:
            Dictionary with database configurations
        """
        databases = {}

        # Handle both dict and PluginManifest objects
        if isinstance(manifest, dict):
            manifest_data = manifest
        else:
            manifest_data = manifest.model_dump()

        # Load ChromaDB config if enabled
        if "databases" in manifest_data and "chromadb" in manifest_data["databases"]:
            chromadb_config = manifest_data["databases"]["chromadb"]
            if "enabled" in chromadb_config and chromadb_config["enabled"]:
                config_file = (
                    chromadb_config["config_file"] if "config_file" in chromadb_config else None
                )
                if config_file:
                    config_path = plugin_path / config_file
                    if config_path.exists():
                        with open(config_path) as f:
                            databases["chromadb"] = yaml.safe_load(f)
                        logger.debug(f"Loaded ChromaDB config for plugin at {plugin_path}")
                    else:
                        logger.warning(f"ChromaDB config file not found: {config_path}")

        # Load Neo4j config if enabled
        if "databases" in manifest_data and "neo4j" in manifest_data["databases"]:
            neo4j_config = manifest_data["databases"]["neo4j"]
            if "enabled" in neo4j_config and neo4j_config["enabled"]:
                config_file = neo4j_config["config_file"] if "config_file" in neo4j_config else None
                if config_file:
                    config_path = plugin_path / config_file
                    if config_path.exists():
                        with open(config_path) as f:
                            databases["neo4j"] = yaml.safe_load(f)
                        logger.debug(f"Loaded Neo4j config for plugin at {plugin_path}")
                    else:
                        logger.warning(f"Neo4j config file not found: {config_path}")

        return databases

    def _import_provider_class(self, plugin_path: Path, manifest: Any) -> Any:
        """Dynamically import the provider class from the plugin.

        Args:
            plugin_path: Path to plugin directory
            manifest: Plugin manifest (PluginManifest object or dict)

        Returns:
            Provider class

        Raises:
            ImportError: If provider class cannot be imported
        """
        provider_file = plugin_path / "provider.py"

        # Handle both dict and PluginManifest objects
        if isinstance(manifest, dict):
            provider_class_name = manifest.get("provider_class")
            module_name_base = manifest.get("name")
        else:
            provider_class_name = manifest.provider_class
            module_name_base = manifest.name

        if not provider_class_name:
            raise ImportError("Provider class name not specified in manifest")

        if not provider_file.exists():
            raise ImportError(f"Provider file not found: {provider_file}")

        # Load module
        module_name = f"{module_name_base}_provider"
        spec = importlib.util.spec_from_file_location(module_name, provider_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {provider_file}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get provider class
        if not hasattr(module, provider_class_name):
            raise ImportError(f"Provider class {provider_class_name} not found in {provider_file}")

        return getattr(module, provider_class_name)

    async def query_domain(self, domain: str, query: str, limit: int = 10) -> list[Knowledge]:
        """Query all plugins that handle the specified domain.

        Args:
            domain: Knowledge domain to search
            query: Search query
            limit: Maximum results across all plugins

        Returns:
            Merged and ranked knowledge results
        """
        if not self._initialized:
            await self.initialize()

        if not domain:
            raise ValueError("Domain cannot be empty")

        if not query.strip():
            raise ValueError("Query cannot be empty")

        # Find plugins that support this domain
        relevant_plugins = [
            (name, provider)
            for name, provider in self.loaded_plugins.items()
            if provider.supports_domain(domain)
        ]

        if not relevant_plugins:
            logger.warning(f"No plugins found for domain: {domain}")
            return []

        logger.info(f"Querying {len(relevant_plugins)} plugins for domain '{domain}'")

        # Query all relevant plugins concurrently
        tasks = []
        for plugin_name, provider in relevant_plugins:
            task = self._query_plugin_safely(plugin_name, provider, domain, query, limit)
            tasks.append(task)

        # Wait for all queries to complete
        plugin_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results
        all_results = []
        for i, result in enumerate(plugin_results):
            plugin_name = relevant_plugins[i][0]
            if isinstance(result, Exception):
                logger.error(f"Query failed for plugin {plugin_name}: {result}")
            elif isinstance(result, list):
                all_results.extend(result)
                logger.debug(f"Plugin {plugin_name} returned {len(result)} results")

        # Merge and rank results from all plugins
        merged_results = self._merge_plugin_results(all_results, limit)
        logger.info(f"Domain query for '{domain}' returned {len(merged_results)} total results")

        return merged_results

    async def _query_plugin_safely(
        self, plugin_name: str, provider: KnowledgeProvider, domain: str, query: str, limit: int
    ) -> list[Knowledge]:
        """Query a single plugin with error handling.

        Args:
            plugin_name: Name of the plugin
            provider: Provider instance
            domain: Knowledge domain
            query: Search query
            limit: Maximum results

        Returns:
            Knowledge results from the plugin
        """
        try:
            return await provider.query(domain, query, limit)
        except Exception as e:
            logger.error(f"Query failed for plugin {plugin_name}: {e}")
            return []

    def _merge_plugin_results(self, results: list[Knowledge], limit: int) -> list[Knowledge]:
        """Merge and rank results from multiple plugins.

        Args:
            results: Results from all plugins
            limit: Maximum results to return

        Returns:
            Deduplicated and ranked knowledge results
        """
        if not results:
            return []

        # Remove duplicates based on content similarity
        unique_results: dict[str, Any] = {}
        for result in results:
            # Create a key based on first 100 chars and domain
            key = f"{result.content[:100].strip()}_{result.domain}"

            # Keep the result with higher confidence
            if key not in unique_results or result.confidence > unique_results[key].confidence:
                unique_results[key] = result

        # Sort by confidence score (descending) and return top results
        sorted_results = sorted(unique_results.values(), key=lambda x: x.confidence, reverse=True)

        return sorted_results[:limit]

    def get_available_domains(self) -> list[str]:
        """Get all available knowledge domains across all plugins.

        Returns:
            Sorted list of unique domains
        """
        domains = set()
        for provider in self.loaded_plugins.values():
            domains.update(provider.domains)
        return sorted(domains)

    def get_plugin_info(self) -> dict[str, dict]:
        """Get information about all loaded plugins.

        Returns:
            Dictionary mapping plugin names to their info
        """
        # Return plugin info with strict access - no fallbacks
        plugin_info = {}
        for name, config in self.plugin_configs.items():
            # Validate required fields
            if "domains" not in config:
                logger.warning(f"Plugin {name} missing 'domains' field")
                continue
            if "description" not in config:
                logger.warning(f"Plugin {name} missing 'description' field")
                continue
            if "version" not in config:
                logger.warning(f"Plugin {name} missing 'version' field")
                continue

            # Get database keys with explicit check
            database_keys = []
            if "databases" in config and isinstance(config["databases"], dict):
                database_keys = list(config["databases"].keys())

            # Get priority with explicit check
            priority = 50  # Default priority
            if "priority" in config:
                priority = config["priority"]

            plugin_info[name] = {
                "domains": config["domains"],
                "description": config["description"],
                "version": config["version"],
                "databases": database_keys,
                "priority": priority,
            }

        return plugin_info

    def get_domain_plugins(self, domain: str) -> list[str]:
        """Get names of plugins that support a specific domain.

        Args:
            domain: Domain to check

        Returns:
            List of plugin names that support the domain
        """
        return [
            name
            for name, provider in self.loaded_plugins.items()
            if provider.supports_domain(domain)
        ]

    async def shutdown(self) -> None:
        """Shutdown all loaded plugins."""
        logger.info("Shutting down Knowledge Plugin Manager")

        # Shutdown all plugins concurrently
        shutdown_tasks = []
        for plugin_name, provider in self.loaded_plugins.items():
            task = self._shutdown_plugin_safely(plugin_name, provider)
            shutdown_tasks.append(task)

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        self.loaded_plugins.clear()
        self.plugin_configs.clear()
        self._initialized = False
        logger.info("Knowledge Plugin Manager shutdown complete")

    async def _shutdown_plugin_safely(self, plugin_name: str, provider: KnowledgeProvider) -> None:
        """Shutdown a single plugin with error handling.

        Args:
            plugin_name: Name of the plugin
            provider: Provider instance
        """
        try:
            await provider.shutdown()
            logger.debug(f"Plugin {plugin_name} shutdown successfully")
        except Exception as e:
            logger.error(f"Error shutting down plugin {plugin_name}: {e}")

    async def load_plugin_by_path(self, plugin_path: Path) -> tuple[bool, str]:
        """Load a single plugin by path.

        Args:
            plugin_path: Path to the plugin directory

        Returns:
            Tuple of (success, message)
        """
        try:
            success = await self._load_plugin(plugin_path)
            if success:
                return True, f"Plugin loaded successfully from {plugin_path.name}"
            else:
                return False, f"Failed to load plugin from {plugin_path.name}"
        except Exception as e:
            logger.error(f"Error loading plugin from {plugin_path}: {e}")
            return False, f"Error loading plugin: {str(e)}"

    async def unload_plugin(self, plugin_name: str) -> tuple[bool, str]:
        """Unload a plugin by name.

        Args:
            plugin_name: Name of the plugin to unload

        Returns:
            Tuple of (success, message)
        """
        try:
            if plugin_name not in self.loaded_plugins:
                return False, f"Plugin '{plugin_name}' is not loaded"

            # Get provider before removing from dict
            provider = self.loaded_plugins[plugin_name]

            # Shutdown the plugin
            await self._shutdown_plugin_safely(plugin_name, provider)

            # Remove from loaded plugins
            del self.loaded_plugins[plugin_name]
            if plugin_name in self.plugin_configs:
                del self.plugin_configs[plugin_name]

            logger.info(f"Plugin '{plugin_name}' unloaded successfully")
            return True, f"Plugin '{plugin_name}' unloaded successfully"

        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False, f"Error unloading plugin: {str(e)}"

    async def test_plugin(self, plugin_name: str) -> tuple[bool, str, dict[str, Any]]:
        """Test a plugin by running basic operations.

        Args:
            plugin_name: Name of the plugin to test

        Returns:
            Tuple of (success, message, test_results)
        """
        try:
            if plugin_name not in self.loaded_plugins:
                return False, f"Plugin '{plugin_name}' is not loaded", {}

            provider = self.loaded_plugins[plugin_name]
            test_results: dict[str, Any] = {}

            # Test 1: Check domains
            domains = provider.domains
            test_results["domains"] = {
                "status": "PASS" if domains else "FAIL",
                "value": domains,
                "message": f"Plugin supports {len(domains)} domains"
                if domains
                else "No domains supported",
            }

            # Test 2: Check if provider responds to domain queries
            domain_tests: dict[str, dict[str, str]] = {}
            if domains:
                for domain in domains[:3]:  # Test first 3 domains
                    try:
                        # Test with a simple query
                        test_query = "test"
                        results = await provider.query(domain, test_query, 1)
                        domain_tests[domain] = {
                            "status": "PASS",
                            "message": f"Query returned {len(results)} results",
                        }
                    except Exception as e:
                        domain_tests[domain] = {
                            "status": "FAIL",
                            "message": f"Query failed: {str(e)}",
                        }

            test_results["domain_queries"] = domain_tests

            # Test 3: Check plugin configuration
            if plugin_name not in self.plugin_configs:
                config = {}
            else:
                config = self.plugin_configs[plugin_name]

            databases_list = []
            if config and "databases" in config:
                databases_list = list(config["databases"].keys())

            test_results["configuration"] = {
                "status": "PASS" if config else "FAIL",
                "message": "Configuration loaded" if config else "No configuration found",
                "databases": databases_list,
            }

            # Overall test result
            all_tests_passed = (
                test_results["domains"]["status"] == "PASS"
                and all(t["status"] == "PASS" for t in domain_tests.values())
                and test_results["configuration"]["status"] == "PASS"
            )

            overall_message = "All tests passed" if all_tests_passed else "Some tests failed"

            return True, overall_message, test_results

        except Exception as e:
            logger.error(f"Error testing plugin {plugin_name}: {e}")
            return False, f"Error testing plugin: {str(e)}", {}

    def get_available_plugins(self) -> list[dict[str, Any]]:
        """Get list of available but not loaded plugins.

        Returns:
            List of available plugin information
        """
        available_plugins = []
        plugin_paths = self._get_plugin_discovery_paths()
        loaded_names = set(self.loaded_plugins.keys())

        for plugin_dir in plugin_paths:
            if not plugin_dir.exists():
                continue

            for plugin_path in plugin_dir.iterdir():
                if not plugin_path.is_dir():
                    continue

                manifest_path = plugin_path / "manifest.yaml"
                if not manifest_path.exists():
                    continue

                try:
                    with open(manifest_path) as f:
                        manifest_raw = yaml.safe_load(f)

                    manifest = PluginManifest.model_validate(manifest_raw)

                    # Skip if already loaded
                    if manifest.name in loaded_names:
                        continue

                    available_plugins.append(
                        {
                            "name": manifest.name,
                            "description": manifest.description,
                            "version": manifest.version,
                            "domains": manifest.domains,
                            "path": str(plugin_path),
                            "auto_load": manifest.auto_load,
                            "priority": manifest.priority,
                        }
                    )

                except Exception as e:
                    logger.warning(f"Could not read manifest from {plugin_path}: {e}")
                    # Add as unknown plugin
                    available_plugins.append(
                        {
                            "name": plugin_path.name,
                            "description": "Unknown plugin (manifest error)",
                            "version": "Unknown",
                            "domains": [],
                            "path": str(plugin_path),
                            "auto_load": False,
                            "priority": 100,
                            "error": str(e),
                        }
                    )

        return available_plugins

    def is_plugin_loaded(self, plugin_name: str) -> bool:
        """Check if a plugin is currently loaded.

        Args:
            plugin_name: Name of the plugin

        Returns:
            True if plugin is loaded, False otherwise
        """
        return plugin_name in self.loaded_plugins


# Global plugin manager instance
plugin_manager = KnowledgePluginManager()
