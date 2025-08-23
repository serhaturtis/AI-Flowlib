"""
Refactored Preset Manager Page following MVC pattern.

This demonstrates the new architecture with business logic controllers
and pure presentation layer.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                               QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
                               QLineEdit, QTextEdit, QTabWidget, QGroupBox, QGridLayout,
                               QMessageBox, QDialog, QDialogButtonBox, QFormLayout,
                               QCheckBox, QSpinBox, QDoubleSpinBox, QFileDialog, QProgressBar, QListWidget,
                               QListWidgetItem, QSplitter, QRadioButton, QButtonGroup,
                               QScrollArea, QFrame)
from PySide6.QtCore import Qt, Signal
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Union

from flowlib.gui.logic.config_manager.preset_controller import PresetController
from flowlib.gui.logic.services.service_factory import ServiceFactory
from flowlib.gui.logic.services.base_controller import ControllerManager

logger = logging.getLogger(__name__)


class PresetManagerPage(QWidget):
    """
    Preset Manager Page - Pure Presentation Layer
    
    Handles only UI concerns and delegates business logic to the controller.
    """
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing Preset Manager page")
        
        # Initialize MVC components
        self.service_factory = ServiceFactory()
        self.controller_manager = ControllerManager(self.service_factory)
        self.controller = self.controller_manager.get_controller_sync(PresetController)
        
        # Controller is initialized in constructor - no deferred initialization
        
        # UI state
        self.presets = []
        self.selected_preset = None
        self.current_variables = {}
        
        # Initialize UI
        self.init_ui()
        
        # Connect controller signals
        self.connect_controller_signals()
        
        # Data will be loaded when page becomes visible via page_visible() method
    
    def connect_controller_signals(self):
        """Connect controller signals to UI update methods."""
        if not self.controller:
            return
            
        # Connect business logic signals to UI updates
        self.controller.presets_loaded.connect(self.update_presets_list)
        self.controller.preset_created.connect(self.on_preset_created)
        self.controller.preset_applied.connect(self.on_preset_applied)
        self.controller.preset_deleted.connect(self.on_preset_deleted)
        self.controller.variables_extracted.connect(self.on_variables_extracted)
        
        # Connect common operation signals
        self.controller.operation_started.connect(self.on_operation_started)
        self.controller.operation_completed.connect(self.on_operation_completed)
        self.controller.operation_failed.connect(self.on_operation_failed)
        self.controller.progress_updated.connect(self.on_progress_updated)
        self.controller.status_updated.connect(self.on_status_updated)
    
    def get_title(self):
        """Get page title for navigation."""
        return "Preset Manager"
    
    def page_visible(self):
        """Called when page becomes visible - load data."""
        logger.debug("Preset Manager page became visible")
        if self.controller:
            logger.info("Loading presets on page visibility")
            self.refresh_presets()
        else:
            logger.warning("No controller available to load presets")
    
    def get_state(self):
        """Get current page state for persistence."""
        return {
            "current_tab": self.tab_widget.currentIndex() if hasattr(self, 'tab_widget') else 0,
            "selected_preset": self.selected_preset,
            "filter_type": self.filter_combo.currentText() if hasattr(self, 'filter_combo') else "All"
        }
    
    def set_state(self, state):
        """Set page state when loading."""
        if hasattr(self, 'tab_widget') and "current_tab" in state:
            self.tab_widget.setCurrentIndex(state["current_tab"])
        if hasattr(self, 'filter_combo') and "filter_type" in state:
            self.filter_combo.setCurrentText(state["filter_type"])
        if "selected_preset" in state and state["selected_preset"]:
            self.selected_preset = state["selected_preset"]
    
    def init_ui(self):
        """Initialize the user interface."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)

        # Title
        self.title = QLabel("Preset Manager")
        self.title.setStyleSheet("font-size: 24px; font-weight: bold; padding-bottom: 10px;")
        self.layout.addWidget(self.title)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        
        # Presets List Tab
        self.presets_tab = self.create_presets_tab()
        self.tab_widget.addTab(self.presets_tab, "Presets")
        
        # Variable Configuration Tab
        self.variables_tab = self.create_variables_tab()
        self.tab_widget.addTab(self.variables_tab, "Variables")
        
        # Apply Preset Tab
        self.apply_tab = self.create_apply_tab()
        self.tab_widget.addTab(self.apply_tab, "Apply Preset")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)
    
    def create_presets_tab(self):
        """Create the presets list tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Filter section
        filter_group = QGroupBox("Filters")
        filter_layout = QHBoxLayout(filter_group)
        layout.addWidget(filter_group)
        
        filter_layout.addWidget(QLabel("Type:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "LLM", "Database", "Vector", "Cache", "Storage"])
        self.filter_combo.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.filter_combo)
        
        # Search
        filter_layout.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search presets...")
        self.search_edit.textChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.search_edit)
        
        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_presets)
        filter_layout.addWidget(refresh_button)
        
        filter_layout.addStretch()
        
        # Action buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        
        self.create_preset_button = QPushButton("Create Preset")
        self.create_preset_button.clicked.connect(self.create_preset)
        button_layout.addWidget(self.create_preset_button)
        
        self.import_preset_button = QPushButton("Import Preset")
        self.import_preset_button.clicked.connect(self.import_preset)
        button_layout.addWidget(self.import_preset_button)
        
        self.export_preset_button = QPushButton("Export Preset")
        self.export_preset_button.clicked.connect(self.export_preset)
        self.export_preset_button.setEnabled(False)
        button_layout.addWidget(self.export_preset_button)
        
        self.delete_preset_button = QPushButton("Delete Preset")
        self.delete_preset_button.clicked.connect(self.delete_preset)
        self.delete_preset_button.setEnabled(False)
        button_layout.addWidget(self.delete_preset_button)
        
        button_layout.addStretch()
        
        # Presets table
        self.presets_table = QTableWidget()
        self.presets_table.setColumnCount(5)
        self.presets_table.setHorizontalHeaderLabels(["Name", "Type", "Description", "Variables", "Created"])
        self.presets_table.horizontalHeader().setStretchLastSection(True)
        self.presets_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.presets_table.selectionModel().selectionChanged.connect(self.on_preset_selection_changed)
        layout.addWidget(self.presets_table)
        
        return widget
    
    def create_variables_tab(self):
        """Create the variables configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Preset selection
        preset_group = QGroupBox("Preset Selection")
        preset_layout = QHBoxLayout(preset_group)
        layout.addWidget(preset_group)
        
        preset_layout.addWidget(QLabel("Select Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.currentTextChanged.connect(self.on_preset_combo_changed)
        preset_layout.addWidget(self.preset_combo)
        
        self.extract_variables_button = QPushButton("Extract Variables")
        self.extract_variables_button.clicked.connect(self.extract_variables)
        preset_layout.addWidget(self.extract_variables_button)
        
        preset_layout.addStretch()
        
        # Variables list
        variables_group = QGroupBox("Variables")
        variables_layout = QVBoxLayout(variables_group)
        layout.addWidget(variables_group)
        
        # Variables scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.variables_widget = QWidget()
        self.variables_layout = QVBoxLayout(self.variables_widget)
        scroll_area.setWidget(self.variables_widget)
        variables_layout.addWidget(scroll_area)
        
        return widget
    
    def create_apply_tab(self):
        """Create the apply preset tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Preset selection and preview
        apply_group = QGroupBox("Apply Preset")
        apply_layout = QVBoxLayout(apply_group)
        layout.addWidget(apply_group)
        
        # Preset selection
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Preset:"))
        self.apply_preset_combo = QComboBox()
        selection_layout.addWidget(self.apply_preset_combo)
        
        self.validate_preset_button = QPushButton("Validate")
        self.validate_preset_button.clicked.connect(self.validate_preset)
        selection_layout.addWidget(self.validate_preset_button)
        
        selection_layout.addStretch()
        apply_layout.addLayout(selection_layout)
        
        # Preview area
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(200)
        apply_layout.addWidget(self.preview_text)
        
        # Apply button
        apply_button_layout = QHBoxLayout()
        self.apply_preset_button = QPushButton("Apply Preset")
        self.apply_preset_button.clicked.connect(self.apply_preset)
        self.apply_preset_button.setEnabled(False)
        apply_button_layout.addWidget(self.apply_preset_button)
        apply_button_layout.addStretch()
        apply_layout.addLayout(apply_button_layout)
        
        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        layout.addWidget(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        return widget
    
    # UI Action Methods (Pure Presentation Layer)
    def refresh_presets(self):
        """Refresh presets list."""
        if self.controller:
            self.controller.refresh_presets()
    
    def create_preset(self):
        """Show create preset dialog."""
        dialog = PresetCreateDialog(self)
        if dialog.exec() == QDialog.Accepted:
            preset_data = dialog.get_preset_data()
            if self.controller:
                self.controller.create_preset(preset_data)
    
    def import_preset(self):
        """Import a preset from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Preset", "", "JSON Files (*.json);;All Files (*)"
        )
        if file_path and self.controller:
            self.controller.import_preset(file_path)
    
    def export_preset(self):
        """Export selected preset to file."""
        if not self.selected_preset:
            QMessageBox.warning(self, "No Selection", "Please select a preset to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Preset", f"{self.selected_preset}.json", "JSON Files (*.json);;All Files (*)"
        )
        if file_path and self.controller:
            self.controller.export_preset(self.selected_preset, file_path)
    
    def delete_preset(self):
        """Delete selected preset."""
        if not self.selected_preset:
            QMessageBox.warning(self, "No Selection", "Please select a preset to delete.")
            return
        
        reply = QMessageBox.question(
            self, "Delete Preset", 
            f"Are you sure you want to delete preset '{self.selected_preset}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes and self.controller and hasattr(self, 'preset_name_to_id'):
            preset_id = self.preset_name_to_id.get(self.selected_preset)
            if preset_id:
                self.controller.delete_preset(preset_id)
    
    def extract_variables(self):
        """Extract variables from selected preset."""
        preset_name = self.preset_combo.currentText()
        if preset_name and self.controller and hasattr(self, 'preset_name_to_id'):
            preset_id = self.preset_name_to_id.get(preset_name)
            if preset_id:
                self.controller.extract_variables(preset_id)
    
    def validate_preset(self):
        """Validate preset with current variables."""
        preset_name = self.apply_preset_combo.currentText()
        if preset_name and self.controller and hasattr(self, 'preset_name_to_id'):
            preset_id = self.preset_name_to_id.get(preset_name)
            if preset_id:
                # Check if we have variable widgets - if not, extract variables first
                if not hasattr(self, 'variable_widgets') or not self.variable_widgets:
                    # Auto-extract variables before validation
                    self.controller.extract_variables(preset_id)
                    # Note: Validation will be called again after extraction completes
                    return
                
                # Get fresh variable values from widgets  
                current_vars = self.get_current_variables()
                self.controller.validate_preset(preset_id, current_vars)
    
    def apply_preset(self):
        """Apply preset with current variables."""
        preset_name = self.apply_preset_combo.currentText()
        if preset_name and self.controller and hasattr(self, 'preset_name_to_id'):
            preset_id = self.preset_name_to_id.get(preset_name)
            if preset_id:
                # Get fresh variables just like validation
                current_vars = self.get_current_variables()
                self.controller.apply_preset(preset_id, current_vars)
    
    def apply_filters(self):
        """Apply filters to presets table."""
        filter_type = self.filter_combo.currentText()
        search_text = self.search_edit.text().lower()
        
        for row in range(self.presets_table.rowCount()):
            type_item = self.presets_table.item(row, 1)
            name_item = self.presets_table.item(row, 0)
            desc_item = self.presets_table.item(row, 2)
            
            show_row = True
            
            # Type filter
            if filter_type != "All" and type_item and type_item.text() != filter_type:
                show_row = False
            
            # Search filter
            if search_text:
                match_found = False
                for item in [name_item, desc_item]:
                    if item and search_text in item.text().lower():
                        match_found = True
                        break
                if not match_found:
                    show_row = False
            
            self.presets_table.setRowHidden(row, not show_row)
    
    def on_preset_selection_changed(self):
        """Handle preset selection changes."""
        selected_row = self.presets_table.currentRow()
        if selected_row >= 0:
            self.selected_preset = self.presets_table.item(selected_row, 0).text()
            self.export_preset_button.setEnabled(True)
            self.delete_preset_button.setEnabled(True)
            
            # Update details
            if self.controller and hasattr(self, 'preset_name_to_id'):
                preset_id = self.preset_name_to_id.get(self.selected_preset)
                if preset_id:
                    self.controller.get_preset_details(preset_id)
        else:
            self.selected_preset = None
            self.export_preset_button.setEnabled(False)
            self.delete_preset_button.setEnabled(False)
    
    def on_preset_combo_changed(self, preset_name):
        """Handle preset combo selection change."""
        if preset_name and self.controller and hasattr(self, 'preset_name_to_id'):
            preset_id = self.preset_name_to_id.get(preset_name)
            if preset_id:
                self.controller.get_preset_details(preset_id)
    
    def create_variable_widgets(self, variables):
        """Create widgets for variable configuration."""
        # Clear existing widgets
        for i in reversed(range(self.variables_layout.count())):
            self.variables_layout.itemAt(i).widget().setParent(None)
        
        self.variable_widgets = {}
        
        for var_name, var_info in variables.items():
            var_frame = QFrame()
            var_frame.setFrameStyle(QFrame.Box)
            var_layout = QFormLayout(var_frame)
            
            # Variable name label
            name_label = QLabel(f"<b>{var_name}</b>")
            var_layout.addRow(name_label)
            
            # Variable description
            if 'description' in var_info:
                desc_label = QLabel(var_info['description'])
                desc_label.setWordWrap(True)
                var_layout.addRow("Description:", desc_label)
            
            # Variable input - fail-fast approach
            if 'type' not in var_info:
                raise ValueError(f"Variable info missing required 'type' field for variable: {var_name}")
            var_type = var_info['type']
            
            if var_type == 'boolean':
                widget = QCheckBox()
                if 'default' not in var_info:
                    raise ValueError(f"Boolean variable missing required 'default' field: {var_name}")
                widget.setChecked(var_info['default'])
            elif var_type == 'integer':
                widget = QSpinBox()
                if 'default' not in var_info:
                    raise ValueError(f"Integer variable missing required 'default' field: {var_name}")
                widget.setValue(var_info['default'])
                if 'min' in var_info and var_info['min'] is not None:
                    widget.setMinimum(int(var_info['min']))
                if 'max' in var_info and var_info['max'] is not None:
                    widget.setMaximum(int(var_info['max']))
            elif var_type == 'number':
                widget = QDoubleSpinBox()
                if 'default' not in var_info:
                    raise ValueError(f"Number variable missing required 'default' field: {var_name}")
                widget.setValue(float(var_info['default']))
                if 'min' in var_info and var_info['min'] is not None:
                    widget.setMinimum(float(var_info['min']))
                if 'max' in var_info and var_info['max'] is not None:
                    widget.setMaximum(float(var_info['max']))
            elif var_type == 'choice':
                widget = QComboBox()
                if 'choices' not in var_info:
                    raise ValueError(f"Choice variable missing required 'choices' field: {var_name}")
                if 'default' not in var_info:
                    raise ValueError(f"Choice variable missing required 'default' field: {var_name}")
                choices = var_info['choices']
                widget.addItems(choices)
                default = var_info['default']
                if default in choices:
                    widget.setCurrentText(default)
            else:  # string
                widget = QLineEdit()
                if 'default' not in var_info:
                    raise ValueError(f"String variable missing required 'default' field: {var_name}")
                default_value = var_info['default']
                widget.setText(str(default_value))
            
            var_layout.addRow("Value:", widget)
            self.variable_widgets[var_name] = widget
            self.variables_layout.addWidget(var_frame)
        
        # Add stretch
        self.variables_layout.addStretch()
    
    def get_current_variables(self):
        """Get current variable values from widgets."""
        variables = {}
        for var_name, widget in self.variable_widgets.items():
            if isinstance(widget, QCheckBox):
                variables[var_name] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                variables[var_name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                variables[var_name] = widget.value()
            elif isinstance(widget, QComboBox):
                variables[var_name] = widget.currentText()
            else:  # QLineEdit
                variables[var_name] = widget.text()
        return variables
    
    # Controller Signal Handlers (Business Logic -> UI Updates)
    def update_presets_list(self, presets):
        """Update presets list with new data."""
        self.presets = presets
        self.presets_table.setRowCount(len(presets))
        
        # Create mapping from preset names to IDs - single source of truth
        self.preset_name_to_id = {}
        preset_names = []
        for preset in presets:
            if 'name' not in preset:
                raise ValueError("Preset missing required 'name' field")
            if 'id' not in preset:
                raise ValueError("Preset missing required 'id' field")
            self.preset_name_to_id[preset['name']] = preset['id']
            preset_names.append(preset['name'])
        
        self.preset_combo.clear()
        self.preset_combo.addItems(preset_names)
        self.apply_preset_combo.clear()
        self.apply_preset_combo.addItems(preset_names)
        
        # Update table - fail-fast approach
        for row, preset in enumerate(presets):
            if 'name' not in preset:
                raise ValueError("Preset missing required 'name' field")
            if 'type' not in preset:
                raise ValueError("Preset missing required 'type' field")
            if 'description' not in preset:
                raise ValueError("Preset missing required 'description' field")
                
            name = preset['name']
            preset_type = preset['type']
            description = preset['description']
            if 'variable_count' not in preset:
                raise ValueError("Preset missing required 'variable_count' field")
            var_count = str(preset['variable_count'])
            if 'created' not in preset:
                raise ValueError("Preset missing required 'created' field")
            created = preset['created']
            
            self.presets_table.setItem(row, 0, QTableWidgetItem(name))
            self.presets_table.setItem(row, 1, QTableWidgetItem(preset_type))
            self.presets_table.setItem(row, 2, QTableWidgetItem(description))
            self.presets_table.setItem(row, 3, QTableWidgetItem(var_count))
            self.presets_table.setItem(row, 4, QTableWidgetItem(created))
        
        # Apply current filters
        self.apply_filters()
    
    def on_preset_created(self, preset):
        """Handle preset creation completion."""
        QMessageBox.information(self, "Success", "Preset created successfully!")
        self.refresh_presets()
    
    def on_preset_applied(self, preset_name, result):
        """Handle preset application completion."""
        self.results_text.setText(f"Preset '{preset_name}' applied successfully.\n\nResults:\n{json.dumps(result, indent=2)}")
        self.apply_preset_button.setEnabled(False)
    
    def on_preset_deleted(self, preset_name):
        """Handle preset deletion completion."""
        QMessageBox.information(self, "Success", f"Preset '{preset_name}' deleted successfully!")
        self.refresh_presets()
    
    def on_variables_extracted(self, preset_name, variables):
        """Handle variables extraction completion."""
        variables_dict = {}
        for var in variables:
            if 'name' not in var:
                raise ValueError("Variable missing required 'name' field")
            variables_dict[var['name']] = var
        
        self.create_variable_widgets(variables_dict)
        self.current_variables = self.get_current_variables()
    
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


class PresetCreateDialog(QDialog):
    """Dialog for creating new presets."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Preset")
        self.setModal(True)
        self.resize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # Form layout
        form_layout = QFormLayout()
        
        self.name_edit = QLineEdit()
        form_layout.addRow("Preset Name:", self.name_edit)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["LLM", "Database", "Vector", "Cache", "Storage"])
        form_layout.addRow("Type:", self.type_combo)
        
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(100)
        form_layout.addRow("Description:", self.description_edit)
        
        layout.addLayout(form_layout)
        
        # Template content
        template_group = QGroupBox("Template Content")
        template_layout = QVBoxLayout(template_group)
        
        self.template_edit = QTextEdit()
        self.template_edit.setPlaceholderText("Enter template content with {{variable}} placeholders...")
        template_layout.addWidget(self.template_edit)
        
        layout.addWidget(template_group)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_preset_data(self):
        """Get preset data from dialog."""
        return {
            "name": self.name_edit.text(),
            "type": self.type_combo.currentText(),
            "description": self.description_edit.toPlainText(),
            "template": self.template_edit.toPlainText()
        }