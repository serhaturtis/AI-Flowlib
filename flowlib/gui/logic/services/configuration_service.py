"""
Configuration Management Service for GUI.

Handles configuration CRUD operations, validation, and metadata management
with clean service architecture.
"""

import ast
import logging
import asyncio
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from .models import (
    ConfigurationItem, ConfigurationDetails, ConfigurationValidationResult,
    ConfigurationType, ConfigurationStatus, ValidationError, OperationResult,
    ConfigurationCreateData, ConfigurationAnalysis, AutoFixResult, ClassInfo, ClassAttribute
)
from .flowlib_integration_service import FlowlibIntegrationService
from .async_qt_helper import AsyncServiceMixin, ensure_async_context
from .error_boundaries import handle_service_errors, ServiceError

logger = logging.getLogger(__name__)


class ConfigurationService(AsyncServiceMixin):
    """
    Service for comprehensive configuration management.
    
    Provides high-level configuration operations including validation,
    metadata extraction, and detailed analysis.
    Follows flowlib async-first design principles.
    """
    
    def __init__(self, flowlib_integration: FlowlibIntegrationService):
        super().__init__()  # Initialize AsyncServiceMixin
        self.flowlib_integration = flowlib_integration
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the configuration service."""
        try:
            if not self.flowlib_integration._initialized:
                await self.flowlib_integration.initialize()
            
            self._initialized = True
            logger.info("ConfigurationService initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ConfigurationService: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the service."""
        self._initialized = False
        logger.info("ConfigurationService shutdown")
    
    def _ensure_initialized(self):
        """Ensure service is initialized using proper async context."""
        if not self._initialized:
            try:
                # Use the async helper for proper Qt-async integration
                ensure_async_context(self.initialize())
            except Exception as e:
                # Propagate error following CLAUDE.md principles - no silent failures
                raise ServiceError(f"Service initialization failed: {e}", 
                                 context={'operation': 'ensure_initialized'})
    
    async def list_configurations(self) -> List[ConfigurationItem]:
        """List all configurations with enhanced metadata."""
        self._ensure_initialized()
        
        try:
            list_items = await self.flowlib_integration.list_configurations()
            
            # Convert ConfigurationListItem to ConfigurationItem with enhanced metadata
            configuration_items = []
            for item in list_items:
                try:
                    # Determine status by validating configuration
                    status = ConfigurationStatus.ACTIVE
                    if item.file_path:
                        try:
                            content = await self.flowlib_integration.load_configuration(item.name)
                            if content:
                                validation_result = await self.validate_configuration_content(content)
                                status = (
                                    ConfigurationStatus.ACTIVE if validation_result.is_valid 
                                    else ConfigurationStatus.INVALID
                                )
                        except Exception as e:
                            logger.warning(f"Failed to enhance metadata for {item.name}: {e}")
                            status = ConfigurationStatus.INVALID
                    
                    # Convert string type to ConfigurationType enum
                    try:
                        config_type = ConfigurationType(item.type.upper())
                    except ValueError:
                        # Handle unknown types gracefully
                        config_type = ConfigurationType.LLM  # Default fallback
                    
                    # Create ConfigurationItem from ConfigurationListItem
                    config_item = ConfigurationItem(
                        name=item.name,
                        type=config_type,
                        provider="unknown",  # Will be enhanced later if needed
                        status=status,
                        file_path=item.file_path,
                        description=f"Configuration for {item.name}"
                    )
                    
                    configuration_items.append(config_item)
                    
                except Exception as e:
                    logger.warning(f"Failed to convert configuration item {item.name}: {e}")
                    # Create minimal valid item for problematic configs
                    config_item = ConfigurationItem(
                        name=item.name,
                        type=ConfigurationType.LLM,
                        provider="unknown",
                        status=ConfigurationStatus.INVALID,
                        file_path=item.file_path,
                        description=f"Error loading {item.name}"
                    )
                    configuration_items.append(config_item)
            
            return configuration_items
            
        except Exception as e:
            logger.error(f"Failed to list configurations: {e}")
            return []
    
    async def get_configuration_details(self, name: str) -> Optional[ConfigurationDetails]:
        """Get detailed configuration information."""
        self._ensure_initialized()
        
        try:
            # Load configuration content
            content = await self.flowlib_integration.load_configuration(name)
            if not content:
                return None
            
            # Get basic configuration item
            configurations = await self.flowlib_integration.list_configurations()
            config_item = None
            for config in configurations:
                if config.name == name:
                    config_item = config
                    break
            
            if not config_item:
                return None
            
            # Analyze content
            analysis = self._analyze_configuration_content(content)
            
            # Convert class info objects to strings for the model
            class_strings = []
            for class_info in analysis.classes:
                if hasattr(class_info, 'string_representation'):
                    class_strings.append(class_info.string_representation)
                elif isinstance(class_info, str):
                    class_strings.append(class_info)
                else:
                    # Fallback for unexpected format
                    class_strings.append(str(class_info))
            
            # Convert ConfigurationListItem to ConfigurationItem
            from .models import ConfigurationItem, ConfigurationType, ConfigurationStatus
            from datetime import datetime
            
            # Map type string to enum
            config_type_enum = ConfigurationType.LLM  # Default
            try:
                config_type_enum = ConfigurationType(config_item.type.lower())
            except ValueError:
                # Handle unknown types
                if 'model' in config_item.type.lower():
                    config_type_enum = ConfigurationType.MODEL
                elif 'vector' in config_item.type.lower():
                    config_type_enum = ConfigurationType.VECTOR
                elif 'embedding' in config_item.type.lower():
                    config_type_enum = ConfigurationType.EMBEDDING
            
            configuration_item = ConfigurationItem(
                name=config_item.name,
                type=config_type_enum,
                provider="unknown",  # ConfigurationListItem doesn't have provider info
                status=ConfigurationStatus.ACTIVE,
                created_at=datetime.now(),
                modified_at=datetime.now(),
                description="",
                file_path=config_item.file_path
            )
            
            details = ConfigurationDetails(
                item=configuration_item,
                content=content,
                size=len(content.encode('utf-8')),
                line_count=len(content.split('\n')),
                imports=analysis.imports,
                decorators=analysis.decorators,
                classes=class_strings
            )
            
            return details
            
        except Exception as e:
            logger.error(f"Failed to get configuration details for '{name}': {e}")
            return None
    
    async def save_configuration(self, name: str, code: str, config_type: Optional[str] = None) -> OperationResult:
        """Save a configuration with validation."""
        self._ensure_initialized()
        
        try:
            # Validate configuration content first
            validation_result = await self.validate_configuration_content(code)
            
            if not validation_result.is_valid:
                return OperationResult(
                    success=False,
                    message=f"Configuration validation failed: {', '.join([e.message for e in validation_result.errors])}"
                )
            
            # Save the configuration - let result speak for itself
            result = await self.flowlib_integration.save_configuration(name, code)
            
            # Don't check result.success - return result and let controller handle status
            return result
            
        except Exception as e:
            logger.error(f"Failed to save configuration '{name}': {e}")
            return OperationResult(
                success=False,
                message=f"Failed to save configuration: {str(e)}"
            )
    
    async def delete_configuration(self, name: str) -> OperationResult:
        """Delete a configuration."""
        self._ensure_initialized()
        
        try:
            result = await self.flowlib_integration.delete_configuration(name)
            
            # Return result - let controller handle success status
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete configuration '{name}': {e}")
            return OperationResult(
                success=False,
                message=f"Failed to delete configuration: {str(e)}"
            )
    
    async def duplicate_configuration(self, original_name: str, new_name: str) -> OperationResult:
        """Duplicate an existing configuration."""
        self._ensure_initialized()
        
        try:
            # Check if source exists
            if not await self.flowlib_integration.configuration_exists(original_name):
                return OperationResult(
                    success=False,
                    message=f"Source configuration '{original_name}' not found"
                )
            
            # Check if target already exists
            if await self.flowlib_integration.configuration_exists(new_name):
                return OperationResult(
                    success=False,
                    message=f"Configuration '{new_name}' already exists"
                )
            
            # Load original content
            content = await self.flowlib_integration.load_configuration(original_name)
            if not content:
                return OperationResult(
                    success=False,
                    message=f"Failed to load source configuration '{original_name}'"
                )
            
            # Update configuration name in content
            updated_content = self._update_configuration_name_in_content(content, new_name)
            
            # Save duplicated configuration
            result = await self.save_configuration(new_name, updated_content)
            
            # Return result - let controller handle success status
            return result
            
        except Exception as e:
            logger.error(f"Failed to duplicate configuration '{original_name}': {e}")
            return OperationResult(
                success=False,
                message=f"Failed to duplicate configuration: {str(e)}"
            )
    
    async def validate_configuration(self, name: str) -> ConfigurationValidationResult:
        """Validate a configuration by name."""
        self._ensure_initialized()
        
        try:
            content = await self.flowlib_integration.load_configuration(name)
            if not content:
                return ConfigurationValidationResult(
                    is_valid=False,
                    errors=[ValidationError(message=f"Configuration '{name}' not found", severity="error")]
                )
            
            return await self.validate_configuration_content(content)
            
        except Exception as e:
            logger.error(f"Failed to validate configuration '{name}': {e}")
            return ConfigurationValidationResult(
                is_valid=False,
                errors=[ValidationError(message=f"Validation error: {str(e)}", severity="error")]
            )
    
    async def validate_configuration_content(self, content: str) -> ConfigurationValidationResult:
        """Validate configuration content with enhanced error detection."""
        start_time = datetime.now()
        errors = []
        warnings = []
        suggestions = []
        
        try:
            # Parse Python syntax
            try:
                parsed = ast.parse(content)
            except SyntaxError as e:
                error_msg = f"Syntax error: {e.msg}"
                
                # Enhanced syntax error detection with suggestions
                if "invalid syntax" in str(e.msg).lower():
                    error_msg += ". Check for missing colons, parentheses, or quotes."
                elif "unexpected indent" in str(e.msg).lower():
                    error_msg += ". Check indentation - use 4 spaces consistently."
                elif "unexpected character" in str(e.msg).lower():
                    error_msg += ". Check for special characters or encoding issues."
                
                errors.append(ValidationError(
                    line_number=e.lineno,
                    column=e.offset,
                    message=error_msg,
                    severity="error"
                ))
                return ConfigurationValidationResult(
                    is_valid=False,
                    errors=errors,
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Analyze AST for flowlib patterns
            analysis = self._analyze_ast(parsed, content)
            
            # Enhanced validation checks
            self._validate_flowlib_patterns(analysis, content, errors, warnings, suggestions)
            self._validate_imports(content, analysis.imports, errors, warnings, suggestions)
            self._validate_decorators(analysis.decorators, errors, warnings, suggestions)
            self._validate_class_structure(analysis.classes, errors, warnings, suggestions)
            
            # Check for common configuration mistakes
            self._check_common_mistakes(content, errors, warnings, suggestions)
            
            # Check for decorators
            decorators = analysis.decorators
            if not decorators:
                errors.append(ValidationError(
                    message="No configuration decorators found (e.g., @llm_config)",
                    severity="error"
                ))
            elif len(decorators) > 1:
                warnings.append(ValidationError(
                    message="Multiple configuration decorators found",
                    severity="warning"
                ))
            
            # Check for class definitions
            classes = analysis.classes
            if not classes:
                errors.append(ValidationError(
                    message="No configuration class found",
                    severity="error"
                ))
            elif len(classes) > 1:
                warnings.append(ValidationError(
                    message="Multiple classes found, ensure only one configuration class",
                    severity="warning"
                ))
            
            # Add suggestions
            if not errors:
                suggestions.append(ValidationError(
                    message="Configuration structure looks good!",
                    line_number=0,
                    column=0,
                    severity="info"
                ))
                
                # Check for docstring - analysis should always have docstring key from _analyze_ast
                if 'docstring' not in analysis:
                    suggestions.append(ValidationError(
                        message="Configuration analysis missing docstring information",
                        severity="info"
                    ))
                elif not analysis.docstring:
                    suggestions.append(ValidationError(
                        message="Consider adding a class docstring for documentation",
                        line_number=0,
                        column=0,
                        severity="info"
                    ))
                
                if len(content.split('\n')) < 10:
                    suggestions.append(ValidationError(
                        message="Consider adding more configuration parameters",
                        line_number=0,
                        column=0,
                        severity="info"
                    ))
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ConfigurationValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ConfigurationValidationResult(
                is_valid=False,
                errors=[ValidationError(message=f"Validation failed: {str(e)}", severity="error")],
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def create_configuration(self, config_data: ConfigurationCreateData) -> OperationResult:
        """Create a new configuration."""
        self._ensure_initialized()
        
        try:
            # Use validated Pydantic model attributes
            name = config_data.name
            content = config_data.content
            config_type = config_data.type
            
            if not name:
                return OperationResult(
                    success=False,
                    message="Configuration name is required"
                )
            
            # Use save_configuration to create the new config
            return await self.save_configuration(name, content, config_type)
            
        except Exception as e:
            logger.error(f"Failed to create configuration: {e}")
            return OperationResult(
                success=False,
                message=f"Failed to create configuration: {str(e)}"
            )
    
    async def get_all_configurations(self) -> List[ConfigurationItem]:
        """Get all configurations (alias for list_configurations)."""
        return await self.list_configurations()
    
    async def get_configurations_by_type(self, config_type: ConfigurationType) -> List[ConfigurationItem]:
        """Get configurations filtered by type."""
        self._ensure_initialized()
        
        try:
            all_configs = await self.list_configurations()
            return [config for config in all_configs if config.type == config_type]
            
        except Exception as e:
            logger.error(f"Failed to get configurations by type '{config_type}': {e}")
            return []
    
    def _analyze_configuration_content(self, content: str) -> ConfigurationAnalysis:
        """Analyze configuration content for metadata."""
        try:
            parsed = ast.parse(content)
            return self._analyze_ast(parsed, content)
        except Exception as e:
            logger.warning(f"Failed to analyze configuration content: {e}")
            return ConfigurationAnalysis(
                syntax_valid=False,
                imports=[],
                classes=[],
                decorators=[],
                line_count=len(content.splitlines()),
                complexity_score=0.0
            )
    
    def _analyze_ast(self, parsed: ast.AST, content: str) -> ConfigurationAnalysis:
        """Analyze AST for configuration patterns."""
        imports = []
        decorators = []
        classes = []
        docstring = None
        
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"from {module} import {alias.name}")
            elif isinstance(node, ast.ClassDef):
                # Collect class information with details
                class_info = {
                    'name': node.name,
                    'bases': [],
                    'attributes': []
                }
                
                # Get base classes
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        class_info['bases'].append(base.id)
                    elif isinstance(base, ast.Attribute):
                        # Handle dotted names like module.ClassName
                        class_info['bases'].append(ast.unparse(base))
                
                # Get class attributes
                attributes = []
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        attributes.append(ClassAttribute(
                            name=item.target.id,
                            annotation=ast.unparse(item.annotation) if item.annotation else None
                        ))
                
                # Create string representation
                class_str = f"{node.name}"
                if class_info['bases']:
                    class_str += f"({', '.join(class_info['bases'])})"
                
                # Create ClassInfo object
                classes.append(ClassInfo(
                    name=node.name,
                    bases=class_info['bases'],
                    attributes=attributes,
                    string_representation=class_str
                ))
                
                # Check for decorators
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorators.append(decorator.id)
                    elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                        decorators.append(f"{decorator.func.id}({ast.unparse(decorator)})")
                
                # Extract docstring
                if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant) and
                    isinstance(node.body[0].value.value, str)):
                    docstring = node.body[0].value.value
        
        return ConfigurationAnalysis(
            syntax_valid=True,
            imports=imports,
            classes=classes,
            decorators=decorators,
            line_count=len(content.splitlines()),
            complexity_score=float(len(classes) + len(decorators) + len(imports)),
            docstring=docstring
        )
    
    def _get_required_imports(self, content: str) -> List[str]:
        """Get required imports based on content analysis."""
        required = []
        
        content_lower = content.lower()
        
        if '@llm_config' in content_lower:
            required.extend([
                'flowlib.providers.llm.base',
                'flowlib.core.decorators'
            ])
        elif '@database_config' in content_lower:
            required.extend([
                'flowlib.providers.db.base',
                'flowlib.core.decorators'
            ])
        elif '@vector_config' in content_lower:
            required.extend([
                'flowlib.providers.vector.base',
                'flowlib.core.decorators'
            ])
        elif '@cache_config' in content_lower:
            required.extend([
                'flowlib.providers.cache.base',
                'flowlib.core.decorators'
            ])
        
        return required
    
    def _update_configuration_name_in_content(self, content: str, new_name: str) -> str:
        """Update configuration name references in content."""
        import re
        
        # Update decorator parameter
        content = re.sub(
            r'(@\w+_config\s*\(\s*["\'])([^"\']+)(["\'])',
            rf'\1{new_name}\3',
            content
        )
        
        # Update class name if it matches a pattern
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('class ') and 'Config' in line:
                # Try to update class name to match new config name
                class_name = f"{new_name.replace('-', '_').replace(' ', '_').title()}Config"
                lines[i] = re.sub(r'class\s+\w+Config', f'class {class_name}', line)
                break
        
        return '\n'.join(lines)
    
    def _validate_flowlib_patterns(self, analysis: ConfigurationAnalysis, content: str, errors: List, warnings: List, suggestions: List):
        """Validate flowlib-specific patterns."""
        # Check for required decorator
        decorators = analysis.decorators
        if not decorators:
            errors.append(ValidationError(
                message="Missing required decorator (@llm_config, @database_config, etc.)",
                severity="error"
            ))
            suggestions.append(ValidationError(
                message="Add a decorator like @llm_config('config-name') above your class",
                severity="suggestion"
            ))
        
        # Check for proper config class inheritance
        classes = analysis.classes
        if not classes:
            errors.append(ValidationError(
                message="No configuration class found",
                severity="error"
            ))
        else:
            for class_info in classes:
                class_name = class_info.name
                
                # Check inheritance
                bases = class_info.bases
                if not any(base.endswith('ConfigResource') for base in bases):
                    warnings.append(ValidationError(
                        message=f"Class {class_name} should inherit from a ConfigResource",
                        severity="warning"
                    ))
    
    def _validate_imports(self, content: str, imports: List[str], errors: List, warnings: List, suggestions: List):
        """Validate import statements."""
        required_patterns = {
            '@llm_config': 'from flowlib.core.decorators import llm_config',
            '@database_config': 'from flowlib.core.decorators import database_config',
            '@vector_config': 'from flowlib.core.decorators import vector_config',
            '@cache_config': 'from flowlib.core.decorators import cache_config',
            'LLMConfigResource': 'from flowlib.providers.llm.base import LLMConfigResource',
            'DatabaseConfigResource': 'from flowlib.providers.db.base import DatabaseConfigResource',
        }
        
        for pattern, required_import in required_patterns.items():
            if pattern in content and not any(required_import in imp for imp in imports):
                warnings.append(ValidationError(
                    message=f"Missing import: {required_import}",
                    severity="warning"
                ))
    
    def _validate_decorators(self, decorators: List[str], errors: List, warnings: List, suggestions: List):
        """Validate decorator usage."""
        config_decorators = ['llm_config', 'database_config', 'vector_db_config', 'cache_config', 'storage_config', 'graph_db_config', 'model_config', 'embedding_config', 'message_queue_config']
        
        found_config_decorator = False
        for decorator in decorators:
            if any(config_dec in decorator for config_dec in config_decorators):
                found_config_decorator = True
                
                # Check decorator format
                if not '(' in decorator or not ')' in decorator:
                    errors.append(ValidationError(
                        message=f"Decorator {decorator} missing parentheses - should be @{decorator}('name')",
                        severity="error"
                    ))
                elif '""' in decorator or "''" in decorator:
                    warnings.append(ValidationError(
                        message=f"Decorator {decorator} has empty name",
                        severity="warning"
                    ))
        
        if not found_config_decorator:
            errors.append(ValidationError(
                message="No configuration decorator found (e.g., @llm_config, @database_config)",
                severity="error"
            ))
    
    def _validate_class_structure(self, classes: List, errors: List, warnings: List, suggestions: List):
        """Validate class structure."""
        if not classes:
            return
        
        for class_info in classes:
            # Handle ClassInfo objects from _analyze_ast
            if not hasattr(class_info, 'name'):
                errors.append(ValidationError(
                    message="Invalid class structure - missing name",
                    severity="error"
                ))
                continue
            
            class_name = class_info.name
            
            # Check class naming convention
            if not class_name.endswith('Config'):
                suggestions.append(ValidationError(
                    message=f"Class {class_name} should end with 'Config' for clarity",
                    severity="suggestion"
                ))
            
            # Check for required fields
            if not hasattr(class_info, 'attributes'):
                warnings.append(ValidationError(
                    message=f"Class {class_name} has no attributes information",
                    severity="warning"
                ))
                continue
                
            attributes = class_info.attributes
            attribute_names = []
            for attr in attributes:
                if hasattr(attr, 'name'):
                    attribute_names.append(attr.name)
            
            if 'provider_type' not in attribute_names:
                warnings.append(ValidationError(
                    message=f"Class {class_name} missing 'provider_type' attribute",
                    severity="warning"
                ))
    
    def _check_common_mistakes(self, content: str, errors: List, warnings: List, suggestions: List):
        """Check for common configuration mistakes."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for hardcoded secrets
            secret_patterns = ['password', 'api_key', 'secret', 'token']
            for pattern in secret_patterns:
                if pattern in line.lower() and ('=' in line) and ('"' in line or "'" in line):
                    # Check if it looks like a hardcoded value
                    if not ('os.environ' in line or 'getenv' in line):
                        warnings.append(ValidationError(
                            line_number=i,
                            message=f"Possible hardcoded secret on line {i}. Consider using environment variables.",
                            severity="warning"
                        ))
            
            # Check for inconsistent indentation
            if line.strip() and line.startswith(' '):
                spaces = len(line) - len(line.lstrip())
                if spaces % 4 != 0:
                    warnings.append(ValidationError(
                        line_number=i,
                        message=f"Inconsistent indentation on line {i}. Use 4 spaces per level.",
                        severity="warning"
                    ))
            
            # Check for missing type hints
            if line.strip().startswith('def ') and '->' not in line and '__init__' not in line:
                suggestions.append(ValidationError(
                    line_number=i,
                    message=f"Consider adding type hints to function on line {i}",
                    severity="suggestion"
                ))
    
    async def auto_fix_configuration(self, content: str) -> AutoFixResult:
        """Attempt to automatically fix common configuration issues."""
        self._ensure_initialized()
        
        fixed_content = content
        fixes_applied = []
        
        try:
            # Fix common indentation issues
            lines = content.split('\n')
            fixed_lines = []
            
            for line in lines:
                if line.strip():
                    # Fix indentation to multiples of 4
                    spaces = len(line) - len(line.lstrip())
                    if spaces % 4 != 0:
                        new_spaces = ((spaces // 4) + 1) * 4
                        fixed_line = ' ' * new_spaces + line.lstrip()
                        fixed_lines.append(fixed_line)
                        fixes_applied.append(f"Fixed indentation: '{line.strip()}'")
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            
            fixed_content = '\n'.join(fixed_lines)
            
            # Add missing imports based on content analysis
            if '@llm_config' in fixed_content and 'from flowlib.core.decorators import llm_config' not in fixed_content:
                import_line = 'from flowlib.core.decorators import llm_config\n'
                fixed_content = import_line + fixed_content
                fixes_applied.append("Added missing llm_config import")
            
            if 'LLMConfigResource' in fixed_content and 'from flowlib.providers.llm.base import LLMConfigResource' not in fixed_content:
                import_line = 'from flowlib.providers.llm.base import LLMConfigResource\n'
                fixed_content = import_line + fixed_content
                fixes_applied.append("Added missing LLMConfigResource import")
            
            return AutoFixResult(
                success=True,
                fixed_content=fixed_content,
                changes_made=fixes_applied,
                issues_found=[]
            )
            
        except Exception as e:
            logger.error(f"Failed to auto-fix configuration: {e}")
            return AutoFixResult(
                success=False,
                fixed_content=content,
                changes_made=[],
                issues_found=[str(e)]
            )