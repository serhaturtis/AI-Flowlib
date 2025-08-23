"""
Pydantic models for GUI services.

Clean implementation following CLAUDE.md principles:
- No fallbacks, no workarounds, single Pydantic contracts
- Type safety everywhere - NO Dict[str, Any] usage
- Strict validation with extra="forbid"
- No legacy code, no backward compatibility
"""

from enum import Enum
from typing import List, Optional, Union, Literal
from datetime import datetime
from pathlib import Path
from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel


class ConfigurationType(str, Enum):
    """Configuration type enumeration."""
    LLM = "llm"
    MODEL = "model"
    DATABASE = "database"
    VECTOR = "vector"
    CACHE = "cache"
    STORAGE = "storage"
    EMBEDDING = "embedding"
    GRAPH = "graph"
    MESSAGE_QUEUE = "mq"


class ConfigurationStatus(str, Enum):
    """Configuration status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    INVALID = "invalid"
    DRAFT = "draft"


class PluginStatus(str, Enum):
    """Knowledge plugin status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    LOADING = "loading"


class TestStatus(str, Enum):
    """Test execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class OperationType(str, Enum):
    """Operation type enumeration."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    IMPORT = "import"
    EXPORT = "export"
    VALIDATE = "validate"
    TEST = "test"


class ConfigurationItem(StrictBaseModel):
    """Configuration item model with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(..., description="Configuration name")
    type: ConfigurationType = Field(..., description="Configuration type")
    provider: str = Field(..., description="Provider type")
    status: ConfigurationStatus = Field(default=ConfigurationStatus.ACTIVE)
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: datetime = Field(default_factory=datetime.now)
    description: str = Field(default="", description="Configuration description")
    file_path: Optional[str] = Field(default=None, description="Configuration file path")


class ConfigurationCreateData(StrictBaseModel):
    """Data for creating new configurations with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(..., min_length=1, description="Configuration name")
    content: str = Field(..., min_length=1, description="Configuration content/code")
    type: ConfigurationType = Field(default=ConfigurationType.LLM, description="Configuration type")
    description: str = Field(default="", description="Configuration description")


class ConfigurationEditData(StrictBaseModel):
    """Data for editing configurations with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    content: str = Field(..., min_length=1, description="Updated configuration content")
    type: Optional[ConfigurationType] = Field(default=None, description="Configuration type")
    description: Optional[str] = Field(default=None, description="Configuration description")


class ConfigurationDetails(StrictBaseModel):
    """Detailed configuration information with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    item: ConfigurationItem
    content: str = Field(..., description="Configuration file content")
    size: int = Field(..., ge=0, description="File size in bytes")
    line_count: int = Field(..., ge=0, description="Number of lines")
    imports: List[str] = Field(default_factory=list, description="Import statements")
    decorators: List[str] = Field(default_factory=list, description="Decorator usage")
    classes: List[str] = Field(default_factory=list, description="Class definitions")


class ValidationError(StrictBaseModel):
    """Validation error with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    message: str = Field(..., description="Error message")
    severity: str = Field(..., description="Error severity")
    line_number: Optional[int] = Field(default=None, description="Line number if applicable")
    column: Optional[int] = Field(default=None, description="Column number if applicable")


class ConfigurationValidationResult(StrictBaseModel):
    """Configuration validation result with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    is_valid: bool = Field(..., description="Whether configuration is valid")
    errors: List[ValidationError] = Field(default_factory=list, description="Validation errors")
    warnings: List[ValidationError] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[ValidationError] = Field(default_factory=list, description="Improvement suggestions")
    execution_time: float = Field(default=0.0, ge=0.0, description="Validation execution time in seconds")


class OperationData(StrictBaseModel):
    """Operation result data with flexible validation for dynamic content."""
    # CLAUDE.md compliance: Single source of truth, no page-specific fields
    
    model_config = ConfigDict(extra="allow")  # Allow dynamic fields for service-specific data
    
    # Core framework fields only - no page-specific fields
    record_count: Optional[int] = Field(default=None, description="Number of records processed")
    file_path: Optional[str] = Field(default=None, description="File path if applicable")
    error_details: Optional[str] = Field(default=None, description="Error details if applicable")
    error_code: Optional[str] = Field(default=None, description="Error code for categorization")
    context: Optional[object] = Field(default=None, description="Error context object")
    error_type: Optional[str] = Field(default=None, description="Error type name for unexpected errors")


class OperationResult(StrictBaseModel):
    """Generic operation result with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    success: bool = Field(..., description="Whether operation succeeded")
    message: str = Field(default="", description="Operation message")
    operation_type: Optional[OperationType] = Field(default=None, description="Type of operation")
    execution_time: float = Field(default=0.0, ge=0.0, description="Execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Operation timestamp")
    data: Optional[OperationData] = Field(default=None, description="Operation result data")


class ConfigurationOperationResult(OperationResult):
    """Configuration-specific operation result."""
    # Inherits strict configuration from StrictBaseModel
    
    config_name: Optional[str] = Field(default=None, description="Configuration name")
    config_type: Optional[ConfigurationType] = Field(default=None, description="Configuration type")
    file_path: Optional[str] = Field(default=None, description="File path")


class PluginInfo(StrictBaseModel):
    """Knowledge plugin information with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(..., description="Plugin name")
    version: str = Field(..., description="Plugin version")
    description: str = Field(default="", description="Plugin description")
    author: str = Field(default="", description="Plugin author")
    status: PluginStatus = Field(default=PluginStatus.INACTIVE)
    path: str = Field(..., description="Plugin directory path")
    manifest_path: str = Field(..., description="Manifest file path")
    created_at: datetime = Field(default_factory=datetime.now)
    dependencies: List[str] = Field(default_factory=list, description="Plugin dependencies")


class PluginManifest(StrictBaseModel):
    """Plugin manifest with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(..., description="Plugin name")
    version: str = Field(..., description="Plugin version")
    description: str = Field(..., description="Plugin description")
    author: str = Field(..., description="Plugin author")
    entry_point: str = Field(..., description="Plugin entry point file")
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")
    flowlib_version: str = Field(..., description="Required flowlib version")
    knowledge_domain: str = Field(..., description="Knowledge domain")


class PluginGenerationRequest(StrictBaseModel):
    """Plugin generation request with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    plugin_name: str = Field(..., min_length=1, description="Plugin name")
    description: str = Field(..., min_length=1, description="Plugin description")
    knowledge_domain: str = Field(..., min_length=1, description="Knowledge domain")
    author: str = Field(default="", description="Plugin author")
    use_vector_db: bool = Field(default=True, description="Use vector database")
    use_graph_db: bool = Field(default=False, description="Use graph database")
    custom_extractors: List[str] = Field(default_factory=list, description="Custom extractor types")


class TestCase(StrictBaseModel):
    """Test case definition with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(..., description="Test case name")
    description: str = Field(default="", description="Test description")
    test_type: str = Field(..., description="Type of test")
    target: str = Field(..., description="Test target (config name, etc.)")
    expected_result: bool = Field(..., description="Expected test result")
    timeout_seconds: int = Field(default=30, ge=1, description="Test timeout")


class TestResult(StrictBaseModel):
    """Test execution result with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    test_case: TestCase
    status: TestStatus = Field(..., description="Test execution status")
    passed: bool = Field(..., description="Whether test passed")
    execution_time: float = Field(..., ge=0.0, description="Execution time in seconds")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    output: str = Field(default="", description="Test output")
    timestamp: datetime = Field(default_factory=datetime.now)


class TestSuite(StrictBaseModel):
    """Test suite with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(..., description="Test suite name")
    description: str = Field(default="", description="Test suite description")
    test_cases: List[TestCase] = Field(default_factory=list, description="Test cases in suite")
    created_at: datetime = Field(default_factory=datetime.now)


class TestSuiteResult(StrictBaseModel):
    """Test suite execution result with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    test_suite: TestSuite
    results: List[TestResult] = Field(default_factory=list, description="Individual test results")
    total_tests: int = Field(..., ge=0, description="Total number of tests")
    passed_tests: int = Field(..., ge=0, description="Number of passed tests")
    failed_tests: int = Field(..., ge=0, description="Number of failed tests")
    skipped_tests: int = Field(..., ge=0, description="Number of skipped tests")
    execution_time: float = Field(..., ge=0.0, description="Total execution time")
    timestamp: datetime = Field(default_factory=datetime.now)


class ImportExportOperation(StrictBaseModel):
    """Import/export operation data with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    operation_type: OperationType = Field(..., description="Import or export operation")
    file_path: str = Field(..., description="File path for operation")
    config_names: List[str] = Field(default_factory=list, description="Configuration names")
    overwrite_existing: bool = Field(default=False, description="Overwrite existing configurations")
    backup_before_import: bool = Field(default=True, description="Create backup before import")


class BackupInfo(StrictBaseModel):
    """Backup information with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    backup_name: str = Field(..., description="Backup name")
    backup_path: str = Field(..., description="Backup file path")
    created_at: datetime = Field(default_factory=datetime.now)
    config_count: int = Field(..., ge=0, description="Number of configurations in backup")
    file_size: int = Field(..., ge=0, description="Backup file size in bytes")
    description: str = Field(default="", description="Backup description")


class PresetInfo(StrictBaseModel):
    """Preset configuration information with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(..., description="Preset name")
    description: str = Field(default="", description="Preset description")
    category: str = Field(..., description="Preset category")
    file_path: str = Field(..., description="Preset file path")
    is_builtin: bool = Field(default=False, description="Whether preset is built-in")
    created_at: datetime = Field(default_factory=datetime.now)


class SystemInfo(StrictBaseModel):
    """System information with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    flowlib_version: str = Field(..., description="Flowlib version")
    python_version: str = Field(..., description="Python version")
    gui_version: str = Field(..., description="GUI version")
    config_directory: str = Field(..., description="Configuration directory path")
    total_configurations: int = Field(..., ge=0, description="Total number of configurations")
    active_configurations: int = Field(..., ge=0, description="Number of active configurations")
    total_plugins: int = Field(..., ge=0, description="Total number of plugins")
    active_plugins: int = Field(..., ge=0, description="Number of active plugins")


class ServiceStatus(StrictBaseModel):
    """Service status information with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    service_name: str = Field(..., description="Service name")
    is_running: bool = Field(..., description="Whether service is running")
    status_message: str = Field(default="", description="Status message")
    last_operation: Optional[str] = Field(default=None, description="Last operation performed")
    operation_count: int = Field(default=0, ge=0, description="Total operations performed")
    error_count: int = Field(default=0, ge=0, description="Total errors encountered")
    uptime_seconds: float = Field(default=0.0, ge=0.0, description="Service uptime in seconds")


class VariableConfig(StrictBaseModel):
    """Pydantic model for preset variable configuration."""
    # Inherits strict configuration from StrictBaseModel
    
    type: Literal['string', 'integer', 'number', 'boolean'] = Field(description="Variable data type")
    description: str = Field(description="Variable description")
    required: bool = Field(default=False, description="Whether variable is required")
    default: Optional[Union[str, int, float, bool]] = Field(default=None, description="Default value")
    min: Optional[float] = Field(default=None, description="Minimum value for numeric types")
    max: Optional[float] = Field(default=None, description="Maximum value for numeric types")
    sensitive: bool = Field(default=False, description="Whether variable contains sensitive data")


class ConfigurationTemplate(StrictBaseModel):
    """Configuration template with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    class_name: str = Field(description="Configuration class name")
    provider_type: str = Field(description="Provider type")
    settings: dict[str, Union[str, int, float, bool]] = Field(default_factory=dict, description="Template settings")


class PresetData(StrictBaseModel):
    """Pydantic model for preset data structure."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(description="Preset display name")
    type: str = Field(description="Configuration type (e.g., 'llm_config', 'database_config')")
    description: str = Field(default="", description="Preset description")
    category: str = Field(default="other", description="Preset category")
    template: ConfigurationTemplate = Field(description="Configuration template")
    variables: dict[str, VariableConfig] = Field(default_factory=dict, description="Variable definitions")


class PresetSummary(StrictBaseModel):
    """Pydantic model for preset summary information."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(description="Preset display name")
    type: str = Field(description="Configuration type")
    description: str = Field(default="", description="Preset description")
    category: str = Field(default="other", description="Preset category")
    variable_count: int = Field(default=0, ge=0, description="Number of variables")


class KnowledgePluginConfig(StrictBaseModel):
    """Pydantic model for knowledge plugin configuration."""
    # Inherits strict configuration from StrictBaseModel
    
    plugin_name: str = Field(description="Plugin name")
    description: str = Field(default="", description="Plugin description")
    knowledge_domain: str = Field(description="Knowledge domain")
    use_vector_db: bool = Field(default=True, description="Use vector database")
    use_graph_db: bool = Field(default=False, description="Use graph database")
    custom_extractors: List[str] = Field(default_factory=list, description="Custom extractor types")


class TestConfig(StrictBaseModel):
    """Pydantic model for test configuration."""
    # Inherits strict configuration from StrictBaseModel
    
    test_name: str = Field(description="Test name")
    test_type: str = Field(description="Test type")
    target: str = Field(description="Test target")
    timeout_seconds: int = Field(default=30, ge=1, description="Test timeout")
    expected_result: bool = Field(default=True, description="Expected test result")


class AnalysisResult(StrictBaseModel):
    """Generic analysis result for various operations."""
    # Inherits strict configuration from StrictBaseModel
    
    status: str = Field(description="Analysis status")
    details: str = Field(default="", description="Analysis details")
    metrics: dict[str, Union[str, int, float, bool]] = Field(default_factory=dict, description="Analysis metrics")
    suggestions: List[str] = Field(default_factory=list, description="Analysis suggestions")


class ClassAttribute(StrictBaseModel):
    """Class attribute information."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(description="Attribute name")
    annotation: Optional[str] = Field(default=None, description="Type annotation")


class ClassInfo(StrictBaseModel):
    """Class information with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(description="Class name")
    bases: List[str] = Field(default_factory=list, description="Base classes")
    attributes: List[ClassAttribute] = Field(default_factory=list, description="Class attributes")
    string_representation: str = Field(description="String representation of class")


class ConfigurationAnalysis(StrictBaseModel):
    """Configuration analysis result."""
    # Inherits strict configuration from StrictBaseModel
    
    syntax_valid: bool = Field(description="Whether syntax is valid")
    imports: List[str] = Field(default_factory=list, description="Import statements found")
    classes: List[ClassInfo] = Field(default_factory=list, description="Class definitions found")
    decorators: List[str] = Field(default_factory=list, description="Decorators found")
    line_count: int = Field(default=0, description="Total lines")
    complexity_score: float = Field(default=0.0, description="Complexity score")
    docstring: Optional[str] = Field(default=None, description="Module docstring")


class AutoFixResult(StrictBaseModel):
    """Auto-fix operation result."""
    # Inherits strict configuration from StrictBaseModel
    
    fixed_content: str = Field(description="Fixed configuration content")
    changes_made: List[str] = Field(default_factory=list, description="List of changes made")
    issues_found: List[str] = Field(default_factory=list, description="Issues that were found")
    success: bool = Field(description="Whether auto-fix was successful")


class PresetVariables(StrictBaseModel):
    """Preset variables with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    values: dict[str, Union[str, int, float, bool]] = Field(default_factory=dict, description="Variable values")


class PluginConfig(StrictBaseModel):
    """Plugin configuration with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    plugin_name: str = Field(description="Plugin name")
    description: str = Field(default="", description="Plugin description")
    knowledge_domain: str = Field(description="Knowledge domain")
    settings: dict[str, Union[str, int, float, bool]] = Field(default_factory=dict, description="Plugin settings")


class TestOptions(StrictBaseModel):
    """Test options with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    timeout_seconds: int = Field(default=30, description="Test timeout")
    verbose: bool = Field(default=False, description="Verbose output")
    options: dict[str, Union[str, int, float, bool]] = Field(default_factory=dict, description="Additional options")


class ValidationDetails(StrictBaseModel):
    """Validation details with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    is_valid: bool = Field(description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    details: dict[str, Union[str, int, float, bool]] = Field(default_factory=dict, description="Additional details")