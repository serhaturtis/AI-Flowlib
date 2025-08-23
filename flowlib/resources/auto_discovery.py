"""
Auto-discovery system for .flowlib configurations.

This module handles automatic discovery and loading of configuration files
from the user's .flowlib directory, following CLAUDE.md principles with
fail-fast validation and no fallbacks.
"""

import os
import sys
import shutil
import logging
import importlib.util
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


class FlowlibAutoDiscovery:
    """Auto-discovery system for .flowlib user configurations."""
    
    def __init__(self):
        self._initialized = False
        self._flowlib_path: Optional[Path] = None
        self._configs_path: Optional[Path] = None
        self._loaded_modules: Set[str] = set()
    
    @property
    def flowlib_path(self) -> Path:
        """Get the .flowlib directory path."""
        if self._flowlib_path is None:
            self._flowlib_path = Path.home() / '.flowlib'
        return self._flowlib_path
    
    @property 
    def configs_path(self) -> Path:
        """Get the .flowlib/configs directory path."""
        if self._configs_path is None:
            self._configs_path = self.flowlib_path / 'configs'
        return self._configs_path
    
    @property
    def knowledge_plugins_path(self) -> Path:
        """Get the .flowlib/knowledge_plugins directory path."""
        return self.flowlib_path / 'knowledge_plugins'
    
    def ensure_directory_structure(self) -> None:
        """Create .flowlib directory structure if it doesn't exist.
        
        Creates:
        - ~/.flowlib/
        - ~/.flowlib/configs/
        - ~/.flowlib/active_configs/
        - ~/.flowlib/knowledge_plugins/
        - ~/.flowlib/logs/
        - ~/.flowlib/temp/
        - ~/.flowlib/backups/
        """
        try:
            directories = [
                self.flowlib_path,
                self.flowlib_path / 'configs',
                self.flowlib_path / 'active_configs',
                self.flowlib_path / 'knowledge_plugins',
                self.flowlib_path / 'logs', 
                self.flowlib_path / 'temp',
                self.flowlib_path / 'backups'
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
            
            # Create __init__.py in configs directory only (needed for package imports)
            configs_init_file = self.configs_path / '__init__.py'
            if not configs_init_file.exists():
                configs_init_file.write_text('"""Flowlib user configurations."""\n')
                logger.debug(f"Created {configs_init_file}")
            
            # Note: We don't create __init__.py in active_configs to avoid auto-importing configs
            
            # Copy example configuration files if none exist
            self._copy_example_configs()
                
        except Exception as e:
            logger.error(f"Failed to create .flowlib directory structure: {e}")
            # Don't raise - this is not critical for flowlib operation
    
    def discover_and_load_configurations(self) -> None:
        """Discover and load all configuration files from .flowlib/configs.
        
        This method:
        1. Ensures directory structure exists
        2. Scans for .py files in ~/.flowlib/configs/
        3. Safely imports each file
        4. Lets decorators handle registration
        """
        if self._initialized:
            return  # Already loaded
        
        try:
            # Ensure directory structure exists
            self.ensure_directory_structure()
            
            # Find all Python configuration files
            config_files = self._find_config_files()
            
            if not config_files:
                logger.debug("No configuration files found in .flowlib/configs")
                self._initialized = True
                return
            
            logger.info(f"Discovered {len(config_files)} configuration files in .flowlib/configs")
            
            # Add configs directory to Python path temporarily
            configs_path_str = str(self.configs_path)
            if configs_path_str not in sys.path:
                sys.path.insert(0, configs_path_str)
                path_added = True
            else:
                path_added = False
            
            try:
                # Import each configuration file
                for config_file in config_files:
                    self._import_config_file(config_file)
                
            finally:
                # Clean up path modification
                if path_added and configs_path_str in sys.path:
                    sys.path.remove(configs_path_str)
            
            self._initialized = True
            logger.info(f"Auto-discovery complete: loaded {len(self._loaded_modules)} configuration modules")
            
        except Exception as e:
            logger.error(f"Configuration auto-discovery failed: {e}")
            # Don't raise - flowlib should work without user configs
            self._initialized = True  # Prevent retry loops
    
    def _find_config_files(self) -> List[Path]:
        """Find all Python configuration files in .flowlib/configs.
        
        Returns:
            List of .py file paths, sorted alphabetically
        """
        config_files = []
        
        if not self.configs_path.exists():
            return config_files
        
        try:
            for file_path in self.configs_path.iterdir():
                if (file_path.is_file() and 
                    file_path.suffix == '.py' and 
                    file_path.name != '__init__.py'):
                    
                    config_files.append(file_path)
            
            # Sort for consistent loading order
            config_files.sort(key=lambda p: p.name)
            
        except Exception as e:
            logger.warning(f"Error scanning .flowlib/configs directory: {e}")
        
        return config_files
    
    def _import_config_file(self, config_file: Path) -> None:
        """Safely import a configuration file.
        
        Args:
            config_file: Path to the Python configuration file
        """
        module_name = config_file.stem  # Filename without extension
        
        if module_name in self._loaded_modules:
            logger.debug(f"Module {module_name} already loaded, skipping")
            return
        
        try:
            # Validate the file before importing
            self._validate_config_file(config_file)
            
            # Import the module using importlib
            spec = importlib.util.spec_from_file_location(module_name, config_file)
            if spec is None or spec.loader is None:
                logger.warning(f"Could not load module spec for {config_file}")
                return
            
            module = importlib.util.module_from_spec(spec)
            
            # Execute the module - this triggers decorator registration
            spec.loader.exec_module(module)
            
            self._loaded_modules.add(module_name)
            logger.info(f"Successfully loaded configuration module: {module_name}")
            
        except Exception as e:
            logger.error(f"Failed to import configuration file {config_file}: {e}")
            # Continue with other files rather than failing completely
    
    def _validate_config_file(self, config_file: Path) -> None:
        """Validate a configuration file before importing.
        
        Args:
            config_file: Path to the configuration file
            
        Raises:
            ValueError: If file contains suspicious content
        """
        try:
            # Read the file content for basic validation
            content = config_file.read_text(encoding='utf-8')
            
            # Basic security checks
            suspicious_patterns = [
                'eval(', 'exec(', '__import__', 'compile(',
                'subprocess', 'os.system', 'os.popen'
            ]
            
            for pattern in suspicious_patterns:
                if pattern in content:
                    logger.warning(f"Configuration file {config_file} contains suspicious pattern: {pattern}")
                    # Don't raise - just log the warning and let admin decide
            
            # Check for required imports (basic validation)
            has_flowlib_import = any([
                'from flowlib' in content,
                'import flowlib' in content,
                '@llm_config' in content,
                '@model_config' in content
            ])
            
            if not has_flowlib_import:
                logger.debug(f"Configuration file {config_file} doesn't appear to contain flowlib configurations")
                
        except Exception as e:
            logger.warning(f"Could not validate configuration file {config_file}: {e}")
            # Don't raise - let the import attempt proceed
    
    def get_loaded_modules(self) -> Set[str]:
        """Get the set of successfully loaded module names.
        
        Returns:
            Set of module names that were successfully loaded
        """
        return self._loaded_modules.copy()
    
    def is_initialized(self) -> bool:
        """Check if auto-discovery has been initialized.
        
        Returns:
            True if auto-discovery has completed (successfully or not)
        """
        return self._initialized
    
    def force_reload(self) -> None:
        """Force reload of all configurations.
        
        This clears the initialization state and reloads all configs.
        Useful for development/testing.
        """
        self._initialized = False
        self._loaded_modules.clear()
        self.discover_and_load_configurations()
    
    def _copy_example_configs(self) -> None:
        """Copy example configuration files to ~/.flowlib/active_configs if directory is empty.
        
        This provides users with ready-to-modify configuration templates.
        Only copies if active_configs directory is empty to avoid overwriting user configs.
        """
        try:
            active_configs_dir = self.flowlib_path / 'active_configs'
            
            # Check if active_configs has any .py files (excluding __init__.py)
            existing_configs = [
                f for f in active_configs_dir.iterdir() 
                if f.is_file() and f.suffix == '.py' and f.name != '__init__.py'
            ] if active_configs_dir.exists() else []
            
            if existing_configs:
                logger.debug(f"Active configs already exist ({len(existing_configs)} files), skipping example copy")
                return
            
            # Import example configs module to get file list
            try:
                import flowlib.resources.example_configs as example_configs
                
                # Get the directory where example configs are stored
                example_dir = Path(example_configs.__file__).parent
                
                # Copy each example file
                copied_count = 0
                for example_file, target_file in example_configs.EXAMPLE_TO_TARGET.items():
                    source_path = example_dir / example_file
                    target_path = active_configs_dir / target_file
                    
                    if source_path.exists() and not target_path.exists():
                        shutil.copy2(source_path, target_path)
                        logger.info(f"Copied example config: {target_file}")
                        copied_count += 1
                
                if copied_count > 0:
                    logger.info(f"Copied {copied_count} example configuration files to ~/.flowlib/active_configs/")
                    logger.info("Edit these files to configure your providers, then restart flowlib applications")
                else:
                    logger.debug("No example configs needed to be copied")
                    
            except ImportError as e:
                logger.debug(f"Could not import example configs: {e}")
                # This is not critical - examples might not be available in all installations
                
        except Exception as e:
            logger.debug(f"Failed to copy example configs (non-critical): {e}")
            # Don't raise - this is not critical for flowlib operation


# Global auto-discovery instance
_auto_discovery = FlowlibAutoDiscovery()


def get_auto_discovery() -> FlowlibAutoDiscovery:
    """Get the global auto-discovery instance.
    
    Returns:
        Global FlowlibAutoDiscovery instance
    """
    return _auto_discovery


def ensure_flowlib_structure() -> None:
    """Ensure .flowlib directory structure exists.
    
    This is a convenience function for external use.
    """
    _auto_discovery.ensure_directory_structure()


def discover_configurations() -> None:
    """Trigger configuration auto-discovery.
    
    This is a convenience function for external use.
    """
    _auto_discovery.discover_and_load_configurations()