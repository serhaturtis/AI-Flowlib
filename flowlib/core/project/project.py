"""Project management for flowlib configurations and extensions.

This module provides the Project class which handles loading of configurations,
tools, flows, and agents from a project directory structure.
"""

import importlib.util
import logging
import shutil
import sys
from pathlib import Path

from flowlib.config.required_alias_validator import RequiredAliasValidator
from flowlib.core.project.validator import ProjectValidator

logger = logging.getLogger(__name__)


class Project:
    """Project management for flowlib configurations and extensions.

    A project can be either:
    - Default project at ~/.flowlib/ (when no project path specified)
    - Custom project at /path/to/project/.flowlib/

    Both use identical structure and loading mechanisms.
    """

    def __init__(self, project_path: str | None = None) -> None:
        """Initialize project with specified or default path.

        Args:
            project_path: Optional project root path. If None, uses ~/.flowlib/
        """
        if project_path:
            self.root_path = Path(project_path)
            # For custom projects, the project path IS the flowlib path
            self.flowlib_path = self.root_path
            self.is_default_project = False
        else:
            # Default to home directory "project"
            self.root_path = Path.home()
            self.flowlib_path = Path.home() / ".flowlib"
            self.is_default_project = True

        # Define project structure paths
        self.configs_path = self.flowlib_path / "configs"
        self.tools_path = self.flowlib_path / "tools"
        self.agents_path = self.flowlib_path / "agents"
        self.flows_path = self.flowlib_path / "flows"
        self.providers_path = self.configs_path / "providers"
        self.resources_path = self.configs_path / "resources"
        self.knowledge_plugins_path = self.flowlib_path / "knowledge_plugins"
        self.logs_path = self.flowlib_path / "logs"
        self.temp_path = self.flowlib_path / "temp"
        self.backups_path = self.flowlib_path / "backups"

        # Track loaded modules to prevent duplicates
        self._loaded_modules: set[str] = set()
        self._initialized = False
        self._validator = ProjectValidator()

    def initialize(self) -> None:
        """Initialize project structure and copy templates if needed.

        Creates directory structure and copies example configurations
        if this is a new project.
        """
        if self._initialized:
            return

        try:
            # Ensure directory structure exists
            self._ensure_directories()

            # Copy example configs if this is a new project
            if self._is_new_project():
                self._copy_example_configs()

            self._initialized = True
            logger.info(f"Initialized project at {self.flowlib_path}")

        except Exception as e:
            logger.error(f"Failed to initialize project: {e}")
            raise

    def load_configurations(self) -> None:
        """Load all project configurations into registries.

        This method loads configurations in the following order:
        1. Provider configs from configs/providers/
        2. Alias bindings from configs/aliases.py
        3. Agent configs from agents/
        4. Custom tools from tools/
        5. Custom flows from flows/
        """
        if not self._initialized:
            self.initialize()

        try:
            logger.info(f"Loading configurations from project at {self.flowlib_path}")

            validation_result = self._validator.validate(self.flowlib_path)
            if not validation_result.is_valid:
                issues = "\n".join(
                    f"- {issue.path}: {issue.message}" for issue in validation_result.issues
                )
                raise RuntimeError(f"Project validation failed:\n{issues}")

            # Add project paths to Python path temporarily
            paths_added = []
            # Add project root first so imports like "from tools.module import ..." work
            root_path_str = str(self.root_path)
            if root_path_str not in sys.path:
                sys.path.insert(0, root_path_str)
                paths_added.append(root_path_str)

            for path in [
                self.configs_path,
                self.providers_path,
                self.resources_path,
                self.tools_path,
                self.agents_path,
                self.flows_path,
            ]:
                if path.exists():
                    path_str = str(path)
                    if path_str not in sys.path:
                        sys.path.insert(0, path_str)
                        paths_added.append(path_str)

            try:
                # Load provider configurations from configs/providers/
                self._load_configs()

                # Load alias bindings (requires provider configs)
                self._load_aliases()

                # Note: Alias validation is deferred until agent launch time
                # This allows users to build projects incrementally

                # Load agent configurations
                self._load_agents()

                # Load custom tools
                self._load_tools()

                # Load custom flows
                self._load_flows()

                logger.info(f"Project loading complete: loaded {len(self._loaded_modules)} modules")

            finally:
                # Clean up path modifications
                for path_str in paths_added:
                    if path_str in sys.path:
                        sys.path.remove(path_str)

        except Exception as e:
            logger.error(f"Project configuration loading failed: {e}")
            raise

    def _ensure_directories(self) -> None:
        """Create project directory structure if it doesn't exist."""
        directories = [
            self.flowlib_path,
            self.configs_path,
            self.providers_path,
            self.resources_path,
            self.tools_path,
            self.agents_path,
            self.flows_path,
            self.knowledge_plugins_path,
            self.logs_path,
            self.temp_path,
            self.backups_path,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

        # Create __init__.py files for configs and providers directories
        configs_init = self.configs_path / "__init__.py"
        if not configs_init.exists():
            configs_init.write_text('"""Project configurations."""\n')

        providers_init = self.providers_path / "__init__.py"
        if not providers_init.exists():
            providers_init.write_text('"""Project provider configurations."""\n')

        resources_init = self.resources_path / "__init__.py"
        if not resources_init.exists():
            resources_init.write_text('"""Project resource configurations."""\n')

    def _is_new_project(self) -> bool:
        """Check if this is a new project that needs initialization.

        Returns:
            True if example configs haven't been copied yet
        """
        if not self.configs_path.exists():
            return True

        # Check if example configs were already copied by looking for specific marker files
        # We check multiple directories where example configs get copied to
        import flowlib.resources.example_configs as example_configs

        # Check if any of the target files from EXAMPLE_TO_TARGET exist
        for _example_file, target_file in example_configs.EXAMPLE_TO_TARGET.items():
            if target_file.startswith("../agents/"):
                target_path = self.agents_path / target_file.replace("../agents/", "")
            elif target_file.startswith("providers/"):
                target_path = self.providers_path / target_file.replace("providers/", "")
            elif target_file.startswith("resources/"):
                target_path = self.resources_path / target_file.replace("resources/", "")
            else:
                target_path = self.configs_path / target_file

            # If any target file exists, examples were already copied
            if target_path.exists():
                return False

        # If no target files exist, this is a new project
        return True

    def _copy_example_configs(self) -> None:
        """Copy example configuration files to project.

        Only copies files that don't already exist to prevent overwriting user customizations.
        """
        try:
            # Import example configs module to get file list
            import flowlib.resources.example_configs as example_configs

            # Get the directory where example configs are stored
            example_dir = Path(example_configs.__file__).parent

            # Copy each example file based on mapping
            copied_count = 0
            skipped_count = 0

            for example_file, target_file in example_configs.EXAMPLE_TO_TARGET.items():
                source_path = example_dir / example_file

                # Handle special cases for different target directories
                if target_file.startswith("../agents/"):
                    target_path = self.agents_path / target_file.replace("../agents/", "")
                elif target_file.startswith("providers/"):
                    target_path = self.providers_path / target_file.replace("providers/", "")
                elif target_file.startswith("resources/"):
                    target_path = self.resources_path / target_file.replace("resources/", "")
                else:
                    target_path = self.configs_path / target_file

                # Only copy if source exists and target doesn't exist
                if source_path.exists():
                    if not target_path.exists():
                        # Ensure target directory exists
                        target_path.parent.mkdir(parents=True, exist_ok=True)

                        shutil.copy2(source_path, target_path)
                        logger.info(f"Copied example config: {target_file}")
                        copied_count += 1
                    else:
                        logger.debug(f"Skipped existing file: {target_file}")
                        skipped_count += 1
                else:
                    logger.warning(f"Source example file not found: {example_file}")

            if copied_count > 0:
                logger.info(f"Copied {copied_count} example configuration files to project")
                logger.info(f"Edit configurations in {self.flowlib_path} as needed")

            if skipped_count > 0:
                logger.debug(f"Skipped {skipped_count} existing configuration files")

            if copied_count == 0 and skipped_count == 0:
                logger.debug("No example configs needed to be copied")

        except ImportError as e:
            raise RuntimeError(f"Failed to import example configs: {e}") from e

        except Exception as e:
            raise RuntimeError(f"Failed to copy example configs: {e}") from e

    def _load_configs(self) -> None:
        """Load provider and resource configurations."""
        self._load_provider_configs()
        self._load_resource_configs()

    def _load_provider_configs(self) -> None:
        provider_files = self._find_python_files(self.providers_path)

        if not provider_files:
            logger.debug("No configuration files found in configs/providers/")
        else:
            logger.info(f"Found {len(provider_files)} configuration files in configs/providers/")
            for config_file in provider_files:
                self._import_module(config_file, "provider_config")

    def _load_resource_configs(self) -> None:
        resource_files = self._find_python_files(self.resources_path)

        if not resource_files:
            logger.debug("No configuration files found in configs/resources/")
            return

        logger.info(f"Found {len(resource_files)} configuration files in configs/resources/")

        for resource_file in resource_files:
            self._import_module(resource_file, "resource_config")

    def _load_agents(self) -> None:
        """Load agent configurations from agents/."""
        if not self.agents_path.exists():
            logger.debug("No agents directory found")
            return

        agent_files = self._find_python_files(self.agents_path)
        if not agent_files:
            logger.debug("No agent files found")
            return

        for agent_file in agent_files:
            self._import_module(agent_file, "agent")

    def _load_tools(self) -> None:
        """Load custom tool implementations from tools/."""
        if not self.tools_path.exists():
            logger.debug("No tools directory found")
            return

        # Find all tool.py files in tool subdirectories
        tool_files = list(self.tools_path.rglob("*/tool.py"))

        if not tool_files:
            logger.debug("No custom tools found")
            return

        logger.info(f"Found {len(tool_files)} custom tools")

        for tool_file in tool_files:
            self._import_module(tool_file, "tool")

    def _load_flows(self) -> None:
        """Load custom flows from flows/."""
        if not self.flows_path.exists():
            logger.debug("No flows directory found")
            return

        flow_files = self._find_python_files(self.flows_path)

        if not flow_files:
            logger.debug("No custom flows found")
            return

        logger.info(f"Found {len(flow_files)} custom flows")

        for flow_file in flow_files:
            self._import_module(flow_file, "flow")

    def _load_aliases(self) -> None:
        """Load alias bindings from configs/aliases.py."""
        aliases_file = self.configs_path / "aliases.py"

        if not aliases_file.exists():
            raise RuntimeError(
                f"Missing required alias bindings file at '{aliases_file}'. "
                "Create configs/aliases.py to define canonical configuration aliases."
            )

        self._import_module(aliases_file, "aliases")
        logger.info("Loaded configuration aliases")

    def _validate_required_aliases(self) -> None:
        """Validate required aliases after loading (Phase 2: strict mode).

        Phase 2 implementation: Raises ConfigurationError if required aliases are missing.
        Logs warnings for type mismatches and other issues.
        """
        validation_result = self.validate_required_aliases()

        # Log warnings (non-fatal issues)
        if validation_result.warnings:
            for warning in validation_result.warnings:
                logger.warning(f"Required alias validation warning: {warning}")

        # Raise error for missing required aliases (fatal issues)
        if not validation_result.valid:
            error_message = (
                f"Project '{self.flowlib_path}' has missing required aliases:\n"
                f"{validation_result.get_error_message()}\n\n"
                "Please configure the missing aliases in your configs/aliases.py file."
            )
            raise RuntimeError(error_message)

        logger.info("Required alias validation passed")

    def validate_required_aliases(self):
        """Validate all required aliases are configured.

        Returns:
            ValidationResult with validation details.
        """
        from flowlib.config.required_alias_validator import ValidationResult

        return RequiredAliasValidator.validate_project()

    def _find_python_files(self, directory: Path) -> list[Path]:
        """Find all Python files in a directory (excluding __init__.py).

        Args:
            directory: Directory to search

        Returns:
            List of Python file paths, sorted alphabetically
        """
        if not directory.exists():
            return []

        py_files = [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix == ".py" and f.name != "__init__.py"
        ]

        return sorted(py_files, key=lambda p: p.name)

    def _import_module(self, file_path: Path, module_type: str) -> None:
        """Import a Python module file.

        Args:
            file_path: Path to the Python file
            module_type: Type of module for logging
        """
        module_name = file_path.stem

        # For tool.py files, include parent directory name for uniqueness
        if module_type == "tool" and module_name == "tool":
            # Use parent directory name as the tool name
            parent_dir = file_path.parent.name
            module_name = f"{parent_dir}_tool"

        # Create unique module name to avoid conflicts
        if self.is_default_project:
            full_module_name = f"project_default_{module_type}_{module_name}"
        else:
            project_name = self.root_path.name.replace("-", "_").replace(".", "_")
            full_module_name = f"project_{project_name}_{module_type}_{module_name}"

        if full_module_name in self._loaded_modules:
            logger.debug(f"Module {module_name} already loaded, skipping")
            return

        try:
            # Validate the file before importing
            self._validate_module_file(file_path)

            # Import the module using importlib
            spec = importlib.util.spec_from_file_location(full_module_name, file_path)
            if spec is None or spec.loader is None:
                logger.warning(f"Could not load module spec for {file_path}")
                return

            module = importlib.util.module_from_spec(spec)

            # Execute the module - this triggers decorator registration
            spec.loader.exec_module(module)

            self._loaded_modules.add(full_module_name)
            logger.info(f"Loaded {module_type}: {module_name}")

        except Exception as e:
            logger.error(f"Failed to import {module_type} file {file_path}: {e}")
            # Continue with other files rather than failing completely

    def _validate_module_file(self, file_path: Path) -> None:
        """Basic validation of a Python file before importing.

        Args:
            file_path: Path to the Python file
        """
        try:
            content = file_path.read_text(encoding="utf-8")

            # Basic security checks
            suspicious_patterns = [
                "eval(",
                "exec(",
                "__import__",
                "compile(",
                "subprocess",
                "os.system",
                "os.popen",
            ]

            for pattern in suspicious_patterns:
                if pattern in content:
                    logger.warning(f"File {file_path} contains suspicious pattern: {pattern}")
                    # Don't raise - just log the warning

        except Exception as e:
            logger.warning(f"Could not validate file {file_path}: {e}")

    def get_loaded_modules(self) -> set[str]:
        """Get the set of successfully loaded module names.

        Returns:
            Set of module names that were successfully loaded
        """
        return self._loaded_modules.copy()

    def is_initialized(self) -> bool:
        """Check if project has been initialized.

        Returns:
            True if project has been initialized
        """
        return self._initialized


def get_project(project_path: str | None = None) -> Project:
    """Create a project instance.

    Args:
        project_path: Optional project path

    Returns:
        Project instance
    """
    return Project(project_path)
