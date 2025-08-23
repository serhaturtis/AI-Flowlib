"""
Refactored Import/Export Page following MVC pattern.

This demonstrates the new architecture with business logic controllers
and pure presentation layer.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                               QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
                               QLineEdit, QTextEdit, QTabWidget, QGroupBox, QGridLayout,
                               QMessageBox, QDialog, QDialogButtonBox, QFormLayout,
                               QCheckBox, QSpinBox, QFileDialog, QProgressBar, QListWidget,
                               QListWidgetItem, QSplitter, QRadioButton, QButtonGroup,
                               QScrollArea)
from PySide6.QtCore import Qt, Signal
import logging
import json
from pathlib import Path
from typing import List, Union

from flowlib.gui.logic.config_manager.import_export_controller import ImportExportController
from flowlib.gui.logic.services.service_factory import ServiceFactory
from flowlib.gui.logic.services.base_controller import ControllerManager

logger = logging.getLogger(__name__)


class ImportExportPage(QWidget):
    """
    Import/Export Page - Pure Presentation Layer
    
    Handles only UI concerns and delegates business logic to the controller.
    """
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing Import/Export page")
        
        # Initialize MVC components
        self.service_factory = ServiceFactory()
        self.controller_manager = ControllerManager(self.service_factory)
        self.controller = self.controller_manager.get_controller_sync(ImportExportController)
        
        # Controller is initialized in constructor - no deferred initialization
        
        # UI state
        self.export_preview = None
        self.import_preview = None
        self.selected_file = None
        
        # Initialize UI
        self.init_ui()
        
        # Connect controller signals
        self.connect_controller_signals()
    
    def connect_controller_signals(self):
        """Connect controller signals to UI update methods."""
        if not self.controller:
            return
            
        # Connect business logic signals to UI updates
        self.controller.export_completed.connect(self.on_export_completed)
        self.controller.import_completed.connect(self.on_import_completed)
        self.controller.backup_completed.connect(self.on_backup_completed)
        self.controller.file_analyzed.connect(self.on_file_analyzed)
        self.controller.preview_generated.connect(self.on_preview_generated)
        
        # Connect common operation signals
        self.controller.operation_started.connect(self.on_operation_started)
        self.controller.operation_completed.connect(self.on_operation_completed)
        self.controller.operation_failed.connect(self.on_operation_failed)
        self.controller.progress_updated.connect(self.on_progress_updated)
        self.controller.status_updated.connect(self.on_status_updated)
    
    def get_title(self):
        """Get page title for navigation."""
        return "Import/Export"
    
    def page_visible(self):
        """Called when page becomes visible."""
        logger.debug("Import/Export page became visible")
        # No specific refresh action needed for import/export
    
    def get_state(self):
        """Get current page state for persistence."""
        return {
            "current_tab": self.tab_widget.currentIndex() if hasattr(self, 'tab_widget') else 0,
            "export_format": self.export_format_combo.currentText() if hasattr(self, 'export_format_combo') else "JSON",
            "import_format": self.import_format_combo.currentText() if hasattr(self, 'import_format_combo') else "JSON"
        }
    
    def set_state(self, state):
        """Set page state when loading."""
        if hasattr(self, 'tab_widget') and "current_tab" in state:
            self.tab_widget.setCurrentIndex(state["current_tab"])
        if hasattr(self, 'export_format_combo') and "export_format" in state:
            self.export_format_combo.setCurrentText(state["export_format"])
        if hasattr(self, 'import_format_combo') and "import_format" in state:
            self.import_format_combo.setCurrentText(state["import_format"])
    
    def init_ui(self):
        """Initialize the user interface."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)

        # Title
        self.title = QLabel("Import/Export Manager")
        self.title.setStyleSheet("font-size: 24px; font-weight: bold; padding-bottom: 10px;")
        self.layout.addWidget(self.title)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        
        # Export Tab
        self.export_tab = self.create_export_tab()
        self.tab_widget.addTab(self.export_tab, "Export")
        
        # Import Tab
        self.import_tab = self.create_import_tab()
        self.tab_widget.addTab(self.import_tab, "Import")
        
        # Backup Tab
        self.backup_tab = self.create_backup_tab()
        self.tab_widget.addTab(self.backup_tab, "Backup")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)
    
    def create_export_tab(self):
        """Create the export tab."""
        # Create main widget
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scrollable content area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create content widget that will go inside the scroll area
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Set the content widget in the scroll area
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        
        # Export Configuration
        config_group = QGroupBox("Export Configuration")
        config_layout = QFormLayout(config_group)
        layout.addWidget(config_group)
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["JSON", "YAML", "XML"])
        config_layout.addRow("Format:", self.export_format_combo)
        
        # Export Selection
        selection_group = QGroupBox("Export Selection")
        selection_layout = QVBoxLayout(selection_group)
        layout.addWidget(selection_group)
        
        self.export_all_radio = QRadioButton("Export All Configurations")
        self.export_all_radio.setChecked(True)
        selection_layout.addWidget(self.export_all_radio)
        
        self.export_selected_radio = QRadioButton("Export Selected Types")
        selection_layout.addWidget(self.export_selected_radio)
        
        # Export type selection
        self.export_types_group = QGroupBox("Configuration Types")
        self.export_types_layout = QVBoxLayout(self.export_types_group)
        self.export_types_group.setEnabled(False)
        selection_layout.addWidget(self.export_types_group)
        
        # Create checkboxes for each type
        self.export_checkboxes = {}
        for config_type in ["LLM", "Database", "Vector", "Cache", "Storage", "Embedding"]:
            checkbox = QCheckBox(config_type)
            checkbox.setChecked(True)
            self.export_checkboxes[config_type] = checkbox
            self.export_types_layout.addWidget(checkbox)
        
        # Connect radio buttons
        self.export_selected_radio.toggled.connect(self.export_types_group.setEnabled)
        
        # Export Actions
        actions_layout = QHBoxLayout()
        layout.addLayout(actions_layout)
        
        self.export_preview_button = QPushButton("Preview Export")
        self.export_preview_button.clicked.connect(self.preview_export)
        actions_layout.addWidget(self.export_preview_button)
        
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_configurations)
        actions_layout.addWidget(self.export_button)
        
        actions_layout.addStretch()
        
        # Export Preview
        preview_group = QGroupBox("Export Preview")
        preview_layout = QVBoxLayout(preview_group)
        layout.addWidget(preview_group)
        
        self.export_preview_text = QTextEdit()
        self.export_preview_text.setReadOnly(True)
        self.export_preview_text.setMaximumHeight(200)
        preview_layout.addWidget(self.export_preview_text)
        
        # Add stretch to push everything to the top in the scroll area
        layout.addStretch()
        
        return widget
    
    def create_import_tab(self):
        """Create the import tab."""
        # Create main widget
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scrollable content area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create content widget that will go inside the scroll area
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Set the content widget in the scroll area
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        
        # File Selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        layout.addWidget(file_group)
        
        file_select_layout = QHBoxLayout()
        self.import_file_edit = QLineEdit()
        self.import_file_edit.setPlaceholderText("Select file to import...")
        self.import_file_edit.setReadOnly(True)
        file_select_layout.addWidget(self.import_file_edit)
        
        self.browse_file_button = QPushButton("Browse")
        self.browse_file_button.clicked.connect(self.browse_import_file)
        file_select_layout.addWidget(self.browse_file_button)
        
        file_layout.addLayout(file_select_layout)
        
        # Analyze button
        analyze_layout = QHBoxLayout()
        self.analyze_file_button = QPushButton("Analyze File")
        self.analyze_file_button.clicked.connect(self.analyze_import_file)
        self.analyze_file_button.setEnabled(False)
        analyze_layout.addWidget(self.analyze_file_button)
        analyze_layout.addStretch()
        file_layout.addLayout(analyze_layout)
        
        # File Analysis
        analysis_group = QGroupBox("File Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        layout.addWidget(analysis_group)
        
        self.file_analysis_text = QTextEdit()
        self.file_analysis_text.setReadOnly(True)
        self.file_analysis_text.setMaximumHeight(150)
        analysis_layout.addWidget(self.file_analysis_text)
        
        # Import Options
        options_group = QGroupBox("Import Options")
        options_layout = QFormLayout(options_group)
        layout.addWidget(options_group)
        
        self.import_format_combo = QComboBox()
        self.import_format_combo.addItems(["JSON", "YAML", "XML"])
        options_layout.addRow("Format:", self.import_format_combo)
        
        self.overwrite_existing_check = QCheckBox("Overwrite Existing Configurations")
        options_layout.addRow(self.overwrite_existing_check)
        
        self.validate_before_import_check = QCheckBox("Validate Before Import")
        self.validate_before_import_check.setChecked(True)
        options_layout.addRow(self.validate_before_import_check)
        
        # Import Actions
        import_actions_layout = QHBoxLayout()
        layout.addLayout(import_actions_layout)
        
        self.import_preview_button = QPushButton("Preview Import")
        self.import_preview_button.clicked.connect(self.preview_import)
        self.import_preview_button.setEnabled(False)
        import_actions_layout.addWidget(self.import_preview_button)
        
        self.import_button = QPushButton("Import")
        self.import_button.clicked.connect(self.import_configurations)
        self.import_button.setEnabled(False)
        import_actions_layout.addWidget(self.import_button)
        
        import_actions_layout.addStretch()
        
        # Import Preview
        import_preview_group = QGroupBox("Import Preview")
        import_preview_layout = QVBoxLayout(import_preview_group)
        layout.addWidget(import_preview_group)
        
        self.import_preview_text = QTextEdit()
        self.import_preview_text.setReadOnly(True)
        self.import_preview_text.setMaximumHeight(200)
        import_preview_layout.addWidget(self.import_preview_text)
        
        # Add stretch to push everything to the top in the scroll area
        layout.addStretch()
        
        return widget
    
    def create_backup_tab(self):
        """Create the backup tab."""
        # Create main widget
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scrollable content area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create content widget that will go inside the scroll area
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Set the content widget in the scroll area
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        
        # Backup Configuration
        config_group = QGroupBox("Backup Configuration")
        config_layout = QFormLayout(config_group)
        layout.addWidget(config_group)
        
        self.backup_location_edit = QLineEdit()
        self.backup_location_edit.setPlaceholderText("Default backup location will be used if empty")
        config_layout.addRow("Backup Location:", self.backup_location_edit)
        
        browse_layout = QHBoxLayout()
        self.browse_backup_button = QPushButton("Browse")
        self.browse_backup_button.clicked.connect(self.browse_backup_location)
        browse_layout.addWidget(self.browse_backup_button)
        browse_layout.addStretch()
        config_layout.addRow(browse_layout)
        
        # Backup Options
        options_group = QGroupBox("Backup Options")
        options_layout = QVBoxLayout(options_group)
        layout.addWidget(options_group)
        
        self.include_metadata_check = QCheckBox("Include Metadata")
        self.include_metadata_check.setChecked(True)
        options_layout.addWidget(self.include_metadata_check)
        
        self.compress_backup_check = QCheckBox("Compress Backup")
        self.compress_backup_check.setChecked(True)
        options_layout.addWidget(self.compress_backup_check)
        
        self.encrypt_backup_check = QCheckBox("Encrypt Backup")
        options_layout.addWidget(self.encrypt_backup_check)
        
        # Backup Actions
        backup_actions_layout = QHBoxLayout()
        layout.addLayout(backup_actions_layout)
        
        self.create_backup_button = QPushButton("Create Backup")
        self.create_backup_button.clicked.connect(self.create_backup)
        backup_actions_layout.addWidget(self.create_backup_button)
        
        backup_actions_layout.addStretch()
        
        # Backup History
        history_group = QGroupBox("Backup History")
        history_layout = QVBoxLayout(history_group)
        layout.addWidget(history_group)
        
        self.backup_history_list = QListWidget()
        history_layout.addWidget(self.backup_history_list)
        
        # Add stretch to push everything to the top in the scroll area
        layout.addStretch()
        
        return widget
    
    # UI Action Methods (Pure Presentation Layer)
    def preview_export(self):
        """Preview export operation."""
        export_options = self.get_export_options()
        if self.controller:
            self.controller.generate_export_preview(export_options)
    
    def export_configurations(self):
        """Export configurations."""
        export_options = self.get_export_options()
        
        # Get save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Configurations", 
            f"configurations_export.{export_options['format'].lower()}", 
            f"{export_options['format']} Files (*.{export_options['format'].lower()});;All Files (*)"
        )
        
        if file_path:
            export_options['output_path'] = file_path
            if self.controller:
                self.controller.export_configurations(export_options)
    
    def browse_import_file(self):
        """Browse for import file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Import File", "", 
            "JSON Files (*.json);;YAML Files (*.yaml *.yml);;XML Files (*.xml);;All Files (*)"
        )
        
        if file_path:
            self.selected_file = file_path
            self.import_file_edit.setText(file_path)
            self.analyze_file_button.setEnabled(True)
            self.import_preview_button.setEnabled(True)
            self.import_button.setEnabled(True)
    
    def analyze_import_file(self):
        """Analyze import file."""
        if self.selected_file and self.controller:
            self.controller.analyze_import_file(self.selected_file)
    
    def preview_import(self):
        """Preview import operation."""
        if not self.selected_file:
            QMessageBox.warning(self, "No File Selected", "Please select a file to import.")
            return
        
        import_options = self.get_import_options()
        if self.controller:
            self.controller.generate_import_preview(import_options)
    
    def import_configurations(self):
        """Import configurations."""
        if not self.selected_file:
            QMessageBox.warning(self, "No File Selected", "Please select a file to import.")
            return
        
        import_options = self.get_import_options()
        if self.controller:
            self.controller.import_configurations(import_options)
    
    def browse_backup_location(self):
        """Browse for backup location."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Backup Location")
        if folder_path:
            self.backup_location_edit.setText(folder_path)
    
    def create_backup(self):
        """Create repository backup."""
        backup_options = self.get_backup_options()
        if self.controller:
            self.controller.create_backup(backup_options)
    
    def get_export_options(self):
        """Get export options from UI."""
        export_types = []
        if self.export_selected_radio.isChecked():
            for config_type, checkbox in self.export_checkboxes.items():
                if checkbox.isChecked():
                    export_types.append(config_type)
        else:
            export_types = list(self.export_checkboxes.keys())
        
        return {
            "format": self.export_format_combo.currentText(),
            "export_all": self.export_all_radio.isChecked(),
            "export_types": export_types
        }
    
    def get_import_options(self):
        """Get import options from UI."""
        return {
            "file_path": self.selected_file,
            "format": self.import_format_combo.currentText(),
            "overwrite_existing": self.overwrite_existing_check.isChecked(),
            "validate_before_import": self.validate_before_import_check.isChecked()
        }
    
    def get_backup_options(self):
        """Get backup options from UI."""
        return {
            "backup_location": self.backup_location_edit.text().strip() or None,
            "include_metadata": self.include_metadata_check.isChecked(),
            "compress": self.compress_backup_check.isChecked(),
            "encrypt": self.encrypt_backup_check.isChecked()
        }
    
    # Controller Signal Handlers (Business Logic -> UI Updates)
    def on_export_completed(self, export_path):
        """Handle export completion."""
        QMessageBox.information(self, "Export Complete", f"Configurations exported to:\n{export_path}")
        self.export_preview_text.clear()
    
    def on_import_completed(self, import_result):
        """Handle import completion."""
        # Fail-fast approach
        if 'imported_count' not in import_result:
            raise ValueError("Import result missing required 'imported_count' field")
        if 'skipped_count' not in import_result:
            raise ValueError("Import result missing required 'skipped_count' field")
        if 'error_count' not in import_result:
            raise ValueError("Import result missing required 'error_count' field")
            
        imported_count = import_result['imported_count']
        skipped_count = import_result['skipped_count']
        error_count = import_result['error_count']
        
        message = f"Import completed!\n"
        message += f"Imported: {imported_count}\n"
        message += f"Skipped: {skipped_count}\n"
        message += f"Errors: {error_count}"
        
        QMessageBox.information(self, "Import Complete", message)
        self.import_preview_text.clear()
    
    def on_backup_completed(self, backup_path):
        """Handle backup completion."""
        QMessageBox.information(self, "Backup Complete", f"Backup created at:\n{backup_path}")
        
        # Add to backup history
        item = QListWidgetItem(f"{backup_path} - {Path(backup_path).stat().st_mtime}")
        self.backup_history_list.addItem(item)
    
    def on_file_analyzed(self, analysis_result):
        """Handle file analysis completion."""
        # Fail-fast approach
        if 'format' not in analysis_result:
            raise ValueError("Analysis result missing required 'format' field")
        if 'config_count' not in analysis_result:
            raise ValueError("Analysis result missing required 'config_count' field")
        if 'file_size' not in analysis_result:
            raise ValueError("Analysis result missing required 'file_size' field")
        if 'is_valid' not in analysis_result:
            raise ValueError("Analysis result missing required 'is_valid' field")
            
        analysis_text = f"File Format: {analysis_result['format']}\n"
        analysis_text += f"Configuration Count: {analysis_result['config_count']}\n"
        analysis_text += f"File Size: {analysis_result['file_size']}\n"
        analysis_text += f"Valid: {analysis_result['is_valid']}\n"
        
        if 'errors' in analysis_result:
            analysis_text += f"Errors: {len(analysis_result['errors'])}\n"
            for error in analysis_result['errors'][:5]:  # Show first 5 errors
                analysis_text += f"  - {error}\n"
        
        self.file_analysis_text.setText(analysis_text)
    
    def on_preview_generated(self, operation_type, preview_data):
        """Handle preview generation completion."""
        preview_text = json.dumps(preview_data, indent=2)
        
        if operation_type == "export":
            self.export_preview_text.setText(preview_text)
        elif operation_type == "import":
            self.import_preview_text.setText(preview_text)
    
    def on_operation_started(self, operation_name):
        """Handle operation start."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        logger.info(f"Operation started: {operation_name}")
    
    def on_operation_completed(self, operation_name, success, result):
        """Handle operation completion."""
        self.progress_bar.setVisible(False)
        logger.info(f"Operation completed: {operation_name} - Success: {success}")
    
    def on_operation_failed(self, operation_name, error_message):
        """Handle operation failure."""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Operation Failed", 
                           f"Operation '{operation_name}' failed:\n{error_message}")
        logger.error(f"Operation failed: {operation_name} - {error_message}")
    
    def on_progress_updated(self, operation_name, percentage):
        """Handle progress updates."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(percentage)
    
    def on_status_updated(self, status_message):
        """Handle status updates."""
        logger.info(f"Status: {status_message}")
    
    def closeEvent(self, event):
        """Handle page close event."""
        if self.controller:
            self.controller.cleanup()
        event.accept()