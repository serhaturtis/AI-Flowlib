"""
Import/Export controller for business logic.

Handles import/export operations and coordinates
between the UI layer and service layer.
"""

import logging
from typing import List, Optional, Union
from PySide6.QtCore import Signal
from pydantic import Field
from flowlib.core.models import StrictBaseModel

from ..services.base_controller import BaseController

logger = logging.getLogger(__name__)


class ExportResult(StrictBaseModel):
    """Pydantic model for export operation results."""
    # Inherits strict configuration from StrictBaseModel with frozen=True
    
    success: bool = Field(description="Export success status")
    export_path: str = Field(description="Path to exported file")
    message: str = Field(default="", description="Export message")
    file_count: int = Field(default=0, description="Number of files exported")
    export_size: int = Field(default=0, description="Size of exported data in bytes")


class ImportResult(StrictBaseModel):
    """Pydantic model for import operation results."""
    # Inherits strict configuration from StrictBaseModel with frozen=True
    
    success: bool = Field(description="Import success status")
    message: str = Field(default="", description="Import message")
    imported_count: int = Field(default=0, description="Number of configurations imported")
    skipped_count: int = Field(default=0, description="Number of configurations skipped")
    error_count: int = Field(default=0, description="Number of import errors")


class BackupResult(StrictBaseModel):
    """Pydantic model for backup operation results."""
    # Inherits strict configuration from StrictBaseModel with frozen=True
    
    success: bool = Field(description="Backup success status")
    backup_path: str = Field(description="Path to backup file")
    message: str = Field(default="", description="Backup message")
    backup_size: int = Field(default=0, description="Size of backup in bytes")


class FileAnalysisResult(StrictBaseModel):
    """Pydantic model for file analysis results."""
    # Inherits strict configuration from StrictBaseModel with frozen=True
    
    success: bool = Field(description="Analysis success status")
    file_type: str = Field(description="Detected file type")
    config_count: int = Field(default=0, description="Number of configurations detected")
    message: str = Field(default="", description="Analysis message")
    analysis_details: dict[str, Union[str, int, float, bool]] = Field(default_factory=dict, description="Detailed analysis data")


class PreviewResult(StrictBaseModel):
    """Pydantic model for preview operation results."""
    # Inherits strict configuration from StrictBaseModel with frozen=True
    
    success: bool = Field(description="Preview generation success status")
    preview_type: str = Field(description="Type of preview (import/export)")
    preview_data: dict[str, Union[str, int, float, bool]] = Field(description="Preview data")
    message: str = Field(default="", description="Preview message")


class ImportExportController(BaseController):
    """
    Controller for import/export operations.
    
    Provides business logic for:
    - Configuration export in various formats
    - Configuration import with validation
    - Repository backup and restore
    """
    
    # Specific signals for import/export operations
    export_completed = Signal(str)  # export_path
    import_completed = Signal(dict)  # import_result
    backup_completed = Signal(str)  # backup_path
    file_analyzed = Signal(dict)  # file_analysis_result
    preview_generated = Signal(str, dict)  # operation_type, preview_data
    
    def __init__(self, service_factory):
        super().__init__(service_factory)
        # Initialize service immediately - fail fast, no workarounds
        import asyncio
        self.import_export_service = asyncio.run(self.service_factory.get_import_export_service())
        asyncio.run(self.import_export_service.initialize())
        logger.info("ImportExportController initialized successfully")
    
    def export_configurations(self, export_options: dict[str, Union[str, int, float, bool]]):
        """Export configurations with specified options."""
        logger.info(f"Export configurations requested: {export_options}")
        self.start_operation("export_configurations", self.import_export_service.export_configurations, export_options)
    
    def import_configurations(self, import_options: dict[str, Union[str, int, float, bool]]):
        """Import configurations with specified options."""
        logger.info(f"Import configurations requested: {import_options}")
        self.start_operation("import_configurations", self.import_export_service.import_configurations, import_options)
    
    def create_backup(self, backup_options: dict[str, Union[str, int, float, bool]]):
        """Create a repository backup."""
        logger.info(f"Create backup requested: {backup_options}")
        self.start_operation("create_backup", self.import_export_service.create_backup, backup_options)
    
    def analyze_import_file(self, file_path: str):
        """Analyze an import file to determine its contents."""
        logger.info(f"Analyze import file requested: {file_path}")
        self.start_operation("analyze_file", self.import_export_service.analyze_import_file, file_path)
    
    def generate_export_preview(self, export_options: dict[str, Union[str, int, float, bool]]):
        """Generate a preview of export operation."""
        logger.info(f"Export preview requested: {export_options}")
        self.start_operation("export_preview", self.import_export_service.generate_export_preview, export_options)
    
    def generate_import_preview(self, import_options: dict[str, Union[str, int, float, bool]]):
        """Generate a preview of import operation."""
        logger.info(f"Import preview requested: {import_options}")
        self.start_operation("import_preview", self.import_export_service.generate_import_preview, import_options)
    
    def _on_operation_finished(self, operation_name: str, success: bool, result: Union[str, int, float, bool, dict, list], performance_monitor=None):
        """Handle operation completion with specific logic."""
        super()._on_operation_finished(operation_name, success, result, performance_monitor)
        
        if not success:
            return
        
        # Emit specific signals based on operation type using validated Pydantic models
        try:
            if operation_name == "export_configurations":
                # Result is extracted as dict by base controller - single source of truth
                export_result = ExportResult(**result)
                self.export_completed.emit(export_result.export_path)
                    
            elif operation_name == "import_configurations":
                # Result is extracted as dict by base controller - single source of truth  
                import_result = ImportResult(**result)
                self.import_completed.emit(import_result.model_dump())
                    
            elif operation_name == "create_backup":
                # Result is extracted as dict by base controller - single source of truth
                backup_result = BackupResult(**result)
                self.backup_completed.emit(backup_result.backup_path)
                    
            elif operation_name == "analyze_file":
                # Result is extracted as dict by base controller - single source of truth
                analysis_result = FileAnalysisResult(**result)
                self.file_analyzed.emit(analysis_result.model_dump())
                    
            elif operation_name == "export_preview":
                # Base controller automatically extracts data from OperationResult
                self.preview_generated.emit("export", result)
                    
            elif operation_name == "import_preview":
                # Base controller automatically extracts data from OperationResult
                self.preview_generated.emit("import", result)
                    
        except Exception as e:
            logger.error(f"Failed to validate {operation_name} result: {e}")
            # Emit error recovery signals following CLAUDE.md error propagation
            if operation_name == "export_configurations":
                self.export_completed.emit("unknown")
            elif operation_name == "import_configurations":
                self.import_completed.emit({'success': False, 'message': f'Validation error: {e}'})
            elif operation_name == "create_backup":
                self.backup_completed.emit("unknown")
            elif operation_name == "analyze_file":
                self.file_analyzed.emit({'success': False, 'message': f'Validation error: {e}'})
            elif operation_name.endswith("_preview"):
                preview_type = operation_name.replace("_preview", "")
                self.preview_generated.emit(preview_type, {'success': False, 'message': f'Validation error: {e}'})
    
        
    def get_supported_formats(self) -> List[str]:
        """Get list of supported import/export formats dynamically from service."""
        try:
            # Get formats from import/export service capabilities
            return self.import_export_service.get_supported_formats()
        except Exception as e:
            logger.error(f"Failed to get supported formats from service: {e}")
            # Fail fast - no hardcoded fallbacks
            raise RuntimeError("Cannot determine supported export formats - service unavailable")