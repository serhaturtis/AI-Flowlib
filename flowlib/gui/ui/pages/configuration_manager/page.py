"""
Configuration Manager Page

Enhanced configuration manager page that integrates form-based
configuration creation and editing with text-based system.
"""

import logging
from typing import Optional, Dict, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QButtonGroup,
    QMessageBox, QMenu, QFrame, QLabel, QGroupBox, QTableWidget, 
    QTableWidgetItem, QHeaderView, QComboBox, QLineEdit, QTextEdit, 
    QTabWidget, QGridLayout, QDialog, QDialogButtonBox, QFormLayout,
    QCheckBox, QSpinBox, QFileDialog, QProgressBar, QInputDialog,
    QSplitter
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QIcon

from flowlib.gui.logic.config_manager.form_configuration_controller import FormConfigurationController
from flowlib.gui.logic.services.service_factory import ServiceFactory
from flowlib.gui.logic.services.base_controller import ControllerManager
from flowlib.gui.ui.dialogs.provider_form_factory import (
    ProviderFormFactory, 
    show_provider_form_wizard
)
from flowlib.gui.ui.dialogs.configuration_editor_dialog import ConfigurationEditorDialog
from flowlib.gui.ui.dialogs.file_browser_dialog import FileBrowserDialog
from flowlib.gui.ui.widgets.drag_drop_table import DragDropConfigurationTable

logger = logging.getLogger(__name__)


class ConfigurationManagerPage(QWidget):
    """
    Configuration Manager Page with form-based UI support.
    
    Integrates form-based configuration creation with text-based
    configuration management while maintaining all existing functionality.
    """
    
    # Additional signals for form-based operations
    form_configuration_created = Signal(str, str)  # config_name, provider_type
    form_dialog_opened = Signal(str)               # provider_type
    
    def __init__(self):
        super().__init__()
        
        # Initialize MVC components
        self.service_factory = ServiceFactory()
        self.controller_manager = ControllerManager(self.service_factory)
        self.controller = self.controller_manager.get_controller_sync(FormConfigurationController)
        
        # UI state
        self.configurations = []
        self._pending_export_config = None
        self._pending_edit_config = None
        self._ui_initialized = False
        
        # Initialize UI
        self.init_ui()
        
        # Connect controller signals
        self.connect_controller_signals()
        
        # Add form-based UI components
        self._add_form_ui_components()
        
        # Connect form-specific signals
        self._connect_form_signals()
        
        # Data will be loaded when page becomes visible via page_visible() method
        
        logger.info("Configuration Manager page initialized")
    
    def connect_controller_signals(self):
        """Connect controller signals to UI update methods."""
        if not self.controller:
            return
            
        # Connect business logic signals to UI updates
        self.controller.configurations_loaded.connect(self.update_configurations_list)
        self.controller.configuration_created.connect(self.on_configuration_created)
        self.controller.configuration_deleted.connect(self.on_configuration_deleted)
        self.controller.configuration_validated.connect(self.on_configuration_validated)
        
        # Connect common operation signals
        self.controller.operation_started.connect(self.on_operation_started)
        self.controller.operation_completed.connect(self.on_operation_completed)
        self.controller.operation_failed.connect(self.on_operation_failed)
        self.controller.progress_updated.connect(self.on_progress_updated)
        self.controller.status_updated.connect(self.on_status_updated)

    def get_title(self):
        """Get page title for navigation."""
        return "Configuration Manager"
    
    def page_visible(self):
        """Called when page becomes visible - load data."""
        logger.debug("Configuration Manager page became visible")
        if self.controller:
            logger.info("Loading configurations on page visibility")
            self.refresh_configurations()
        else:
            logger.warning("No controller available to load configurations")
    
    def get_state(self):
        """Get current page state for persistence."""
        return {
            "current_tab": self.tab_widget.currentIndex() if hasattr(self, 'tab_widget') else 0,
            "filter_type": self.filter_combo.currentText() if hasattr(self, 'filter_combo') else "All",
            "form_enhanced": True,
            "form_ui_initialized": hasattr(self, 'create_form_button'),
            "supported_providers": ProviderFormFactory.get_supported_provider_types() if ProviderFormFactory else []
        }
    
    def set_state(self, state):
        """Set page state when loading."""
        if hasattr(self, 'tab_widget') and "current_tab" in state:
            self.tab_widget.setCurrentIndex(state["current_tab"])
        if hasattr(self, 'filter_combo') and "filter_type" in state:
            self.filter_combo.setCurrentText(state["filter_type"])
        # Restore form-specific state if needed
        if (state.get("form_enhanced", False) and not state.get("form_ui_initialized", False)):
            # Re-initialize form UI if it wasn't properly initialized
            self._add_form_ui_components()

    def init_ui(self):
        """Initialize the user interface."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        # Title
        self.title = QLabel("Configuration Manager")
        self.title.setStyleSheet("font-size: 24px; font-weight: bold; padding-bottom: 10px;")
        self.main_layout.addWidget(self.title)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # Configurations List Tab
        self.configs_tab = self.create_configurations_tab()
        self.tab_widget.addTab(self.configs_tab, "Configurations")
        
        # Configuration Details Tab
        self.details_tab = self.create_details_tab()
        self.tab_widget.addTab(self.details_tab, "Details")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)
        
        self._ui_initialized = True

    def create_configurations_tab(self):
        """Create the configurations management tab."""
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
        self.search_edit.setPlaceholderText("Search configurations...")
        self.search_edit.textChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.search_edit)
        
        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_configurations)
        filter_layout.addWidget(refresh_button)
        
        filter_layout.addStretch()
        
        # Action buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        
        self.create_button = QPushButton("Create Configuration")
        self.create_button.clicked.connect(self.create_configuration_with_form_or_text)
        button_layout.addWidget(self.create_button)
        
        self.edit_button = QPushButton("Edit Configuration")
        self.edit_button.clicked.connect(self.edit_configuration)
        self.edit_button.setEnabled(False)
        button_layout.addWidget(self.edit_button)
        
        self.delete_button = QPushButton("Delete Configuration")
        self.delete_button.clicked.connect(self.delete_configuration)
        self.delete_button.setEnabled(False)
        button_layout.addWidget(self.delete_button)
        
        self.export_button = QPushButton("Export Configuration")
        self.export_button.clicked.connect(self.export_configuration)
        self.export_button.setEnabled(False)
        button_layout.addWidget(self.export_button)
        
        button_layout.addStretch()
        
        # Configurations table
        self.configs_table = DragDropConfigurationTable()
        self.configs_table.setColumnCount(4)
        self.configs_table.setHorizontalHeaderLabels(["Name", "Type", "Provider", "Status"])
        self.configs_table.horizontalHeader().setStretchLastSection(True)
        self.configs_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.configs_table.selectionModel().selectionChanged.connect(self.on_configuration_selection_changed)
        layout.addWidget(self.configs_table)
        
        return widget

    def create_details_tab(self):
        """Create the configuration details tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Configuration selection
        selection_group = QGroupBox("Configuration Selection")
        selection_layout = QHBoxLayout(selection_group)
        layout.addWidget(selection_group)
        
        selection_layout.addWidget(QLabel("Select Configuration:"))
        self.details_config_combo = QComboBox()
        self.details_config_combo.currentTextChanged.connect(self.on_details_config_changed)
        selection_layout.addWidget(self.details_config_combo)
        
        selection_layout.addStretch()
        
        # Configuration details
        details_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(details_splitter)
        
        # Content
        content_group = QGroupBox("Configuration Content")
        content_layout = QVBoxLayout(content_group)
        
        self.content_text = QTextEdit()
        self.content_text.setReadOnly(True)
        content_layout.addWidget(self.content_text)
        
        details_splitter.addWidget(content_group)
        
        # Validation
        validation_group = QGroupBox("Validation")
        validation_layout = QVBoxLayout(validation_group)
        
        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setMaximumHeight(150)
        validation_layout.addWidget(self.validation_text)
        
        # Validation button
        validate_button = QPushButton("Validate Configuration")
        validate_button.clicked.connect(self.validate_selected_configuration)
        validation_layout.addWidget(validate_button)
        
        details_splitter.addWidget(validation_group)
        
        return widget

    def _add_form_ui_components(self):
        """Add form-based UI components to the existing interface."""
        try:
            # Enhance the create button tooltip
            if hasattr(self, 'create_button'):
                self.create_button.setToolTip("Create configuration using guided form interface (or code editor for advanced users)")
            
            # Add form creation info panel
            self._add_form_info_panel()
            
        except Exception as e:
            logger.error(f"Failed to add form UI components: {e}")
    
    def _add_form_info_panel(self):
        """Add informational panel about form-based configuration."""
        try:
            # Create info panel
            info_frame = QFrame()
            info_frame.setFrameStyle(QFrame.Shape.StyledPanel)
            info_layout = QVBoxLayout(info_frame)
            
            # Info label
            info_label = QLabel(
                "💡 <b>Configuration Options:</b> "
                "Use the form interface for guided configuration creation with validation, "
                "or create directly with code for advanced customization."
            )
            info_label.setWordWrap(True)
            info_label.setStyleSheet("color: #666; padding: 8px; font-size: 11px;")
            info_layout.addWidget(info_label)
            
            # Insert near the top of the main layout
            if hasattr(self, 'main_layout') and self.main_layout:
                self.main_layout.insertWidget(1, info_frame)  # After title
            
        except Exception as e:
            logger.error(f"Failed to add form info panel: {e}")
    
    def _connect_form_signals(self):
        """Connect form-specific signals."""
        try:
            if hasattr(self.controller, 'form_configuration_created'):
                self.controller.form_configuration_created.connect(self.on_form_configuration_created)
            
            if hasattr(self.controller, 'form_dialog_requested'):
                self.controller.form_dialog_requested.connect(self.on_form_dialog_requested)
            
        except Exception as e:
            logger.error(f"Failed to connect form signals: {e}")

    # UI Action Methods
    def refresh_configurations(self):
        """Refresh configurations list."""
        if self.controller:
            self.controller.refresh_configurations()

    def create_configuration_with_form_or_text(self):
        """Create configuration - show form dialog with fallback to text editor."""
        try:
            from PySide6.QtWidgets import QMessageBox, QPushButton
            
            # Create custom message box with proper button labels
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Create Configuration")
            msg_box.setText("How would you like to create the configuration?")
            msg_box.setInformativeText(
                "• Form: Guided interface with validation (recommended)\n"
                "• Code: Direct code editing (advanced users)"
            )
            
            # Add custom buttons with proper labels
            form_button = msg_box.addButton("Use Form", QMessageBox.ButtonRole.AcceptRole)
            code_button = msg_box.addButton("Write Code", QMessageBox.ButtonRole.RejectRole) 
            cancel_button = msg_box.addButton("Cancel", QMessageBox.ButtonRole.DestructiveRole)
            
            msg_box.setDefaultButton(form_button)
            
            # Show dialog and handle response
            result = msg_box.exec()
            clicked_button = msg_box.clickedButton()
            
            if clicked_button == form_button:
                # User chose form-based
                self.create_configuration_with_form()
            elif clicked_button == code_button:
                # User chose text-based
                self.create_configuration_with_text()
            # Cancel does nothing
            
        except Exception as e:
            logger.error(f"Failed to show configuration creation dialog: {e}")
            # Fallback to text-based creation
            self.create_configuration_with_text()

    def create_configuration_with_form(self):
        """Create configuration using universal form-based interface."""
        try:
            logger.info("Opening universal form-based configuration creation")
            
            # Use the universal wizard
            result = show_provider_form_wizard(self)
            if result:
                # Validate required fields are present
                self._validate_configuration_data(result)
                # Extract configuration data
                config_name = result['config_name']
                # Fail-fast approach - no fallbacks
                if 'config_type' not in result:
                    raise ValueError("Required field 'config_type' missing from form result")
                config_type = result['config_type']
                provider_type = result['provider_type']
                
                # Remove metadata fields from form data
                form_data = {k: v for k, v in result.items() 
                           if k not in ['config_name', 'config_type', 'provider_type']}
                
                # Parse provider type to get specific type
                if '/' in provider_type:
                    category, specific_type = provider_type.split('/', 1)
                    provider_specific_type = specific_type
                else:
                    provider_specific_type = provider_type
                    # Use registry to discover category - no fallbacks
                    category = self._discover_provider_category(provider_type)
                
                # Import the configuration controller and data model
                from flowlib.gui.logic.config_manager.form_configuration_controller import (
                    FormConfigurationController, FormConfigurationData
                )
                
                # Create form configuration data
                form_config = FormConfigurationData(
                    config_name=config_name,
                    provider_type=category,
                    provider_specific_type=provider_specific_type,
                    config_type=config_type,
                    form_data=form_data,
                    description=f"Auto-generated {config_type} configuration for {provider_type}"
                )
                
                # Generate configuration code
                config_code = FormConfigurationController.generate_configuration_code(
                    form_config, config_type
                )
                
                # Save the configuration using the configuration service
                # Use the existing service infrastructure which handles async properly
                self._save_configuration_async(config_name, config_code, config_type)
            
        except Exception as e:
            logger.error(f"Failed to create configuration with form: {e}")
            QMessageBox.critical(
                self,
                "Form Creation Error",
                f"Failed to create configuration:\n\n{str(e)}"
            )

    def create_configuration_with_text(self):
        """Create configuration using text-based interface."""
        try:
            logger.info("Opening text-based configuration creation")
            dialog = ConfigurationEditorDialog(self)
            if dialog.exec() == QDialog.Accepted:
                config_data = dialog.get_configuration_data()
                if self.controller:
                    self.controller.create_configuration(config_data)
            
        except Exception as e:
            logger.error(f"Failed to create configuration with text: {e}")
            QMessageBox.critical(
                self,
                "Text Creation Error",
                f"Failed to open text-based configuration creator:\n\n{str(e)}"
            )

    def edit_configuration(self):
        """Edit selected configuration."""
        selected_row = self.configs_table.currentRow()
        if selected_row >= 0:
            config_name = self.configs_table.item(selected_row, 0).text()
            if self.controller:
                self.controller.edit_configuration(config_name, self)

    def delete_configuration(self):
        """Delete selected configuration."""
        selected_row = self.configs_table.currentRow()
        if selected_row >= 0:
            config_name = self.configs_table.item(selected_row, 0).text()
            
            reply = QMessageBox.question(
                self, "Delete Configuration", 
                f"Are you sure you want to delete configuration '{config_name}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes and self.controller:
                self.controller.delete_configuration(config_name)

    def export_configuration(self):
        """Export selected configuration."""
        selected_row = self.configs_table.currentRow()
        if selected_row >= 0:
            config_name = self.configs_table.item(selected_row, 0).text()
            if self.controller:
                self.controller.export_configuration(config_name, self)

    def validate_selected_configuration(self):
        """Validate the selected configuration."""
        config_name = self.details_config_combo.currentText()
        if config_name and self.controller:
            self.controller.validate_configuration(config_name)

    def apply_filters(self):
        """Apply filters to configurations table."""
        filter_type = self.filter_combo.currentText()
        search_text = self.search_edit.text().lower()
        
        for row in range(self.configs_table.rowCount()):
            type_item = self.configs_table.item(row, 1)
            name_item = self.configs_table.item(row, 0)
            provider_item = self.configs_table.item(row, 2)
            
            show_row = True
            
            # Type filter
            if filter_type != "All" and type_item and type_item.text() != filter_type:
                show_row = False
            
            # Search filter
            if search_text:
                match_found = False
                for item in [name_item, provider_item]:
                    if item and search_text in item.text().lower():
                        match_found = True
                        break
                if not match_found:
                    show_row = False
            
            self.configs_table.setRowHidden(row, not show_row)

    def on_configuration_selection_changed(self):
        """Handle configuration selection changes."""
        selected_row = self.configs_table.currentRow()
        has_selection = selected_row >= 0
        
        self.edit_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)
        self.export_button.setEnabled(has_selection)
        
        if has_selection:
            config_name = self.configs_table.item(selected_row, 0).text()
            # Update details combo
            self.details_config_combo.setCurrentText(config_name)

    def on_details_config_changed(self, config_name):
        """Handle details configuration selection change."""
        if config_name and self.controller:
            self.controller.get_configuration_details(config_name)

    # Helper methods
    def _discover_provider_category(self, provider_type: str) -> str:
        """Discover provider category from registry - no fallbacks."""
        from flowlib.gui.logic.settings_discovery import SettingsDiscovery
        
        try:
            # Get available provider types from registry
            discovery = SettingsDiscovery()
            available_types = discovery.get_available_provider_types()
            
            # Search for the provider in each category
            for category, providers in available_types.items():
                if provider_type in providers:
                    return category
            
            # No fallbacks - if not found, raise error with helpful message
            available_providers = []
            for category, providers in available_types.items():
                for provider in providers:
                    available_providers.append(f"{category}/{provider}")
            
            raise ValueError(
                f"Provider type '{provider_type}' not found in registry. "
                f"Available providers: {', '.join(sorted(available_providers))}"
            )
            
        except Exception as e:
            # If registry discovery fails, provide clear error
            raise ValueError(
                f"Failed to discover category for provider '{provider_type}': {str(e)}"
            )

    def _validate_configuration_data(self, result: Dict[str, Any]) -> None:
        """Validate configuration data has required fields."""
        required_fields = ['config_name', 'provider_type']
        missing_fields = []
        
        for field in required_fields:
            if field not in result or not result[field]:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(
                f"Missing required configuration fields: {', '.join(missing_fields)}"
            )
        
        # Validate config name format
        config_name = result['config_name']
        if not config_name.replace('-', '').replace('_', '').replace(' ', '').isalnum():
            raise ValueError(
                f"Configuration name '{config_name}' contains invalid characters. "
                "Use only letters, numbers, hyphens, underscores, and spaces."
            )
        
        # Validate provider type format
        provider_type = result['provider_type']
        if not provider_type or not isinstance(provider_type, str):
            raise ValueError("Provider type must be a non-empty string")

    def _save_configuration_async(self, config_name: str, config_code: str, config_type: str):
        """Save configuration using existing service infrastructure."""
        # Use the controller's start_operation method instead of self.start_operation
        operation_name = f"save_configuration_{config_name}"
        
        # Start the save operation using the controller
        self.controller.start_operation(
            operation_name,
            self.controller.config_manager.save_configuration,
            name=config_name,
            code=config_code,
            config_type=config_type
        )
        
        # Connect to the completion signal to handle the result
        def handle_save_result(operation: str, success: bool, result=None):
            if operation == operation_name:
                if success:
                    QMessageBox.information(
                        self,
                        "Configuration Created",
                        f"Configuration '{config_name}' created and saved successfully!"
                    )
                    # Refresh the configurations list
                    self.refresh_configurations()
                else:
                    # Success status is the single source of truth - result contains only data
                    QMessageBox.critical(
                        self,
                        "Save Error", 
                        f"Failed to save configuration '{config_name}'"
                    )
                # Disconnect this handler after use
                try:
                    self.controller.operation_completed.disconnect(handle_save_result)
                except RuntimeError:
                    # Signal was already disconnected or connection doesn't exist
                    pass
        
        # Connect the result handler
        self.controller.operation_completed.connect(handle_save_result)

    # Controller Signal Handlers
    def update_configurations_list(self, configurations):
        """Update configurations list with new data."""
        self.configurations = configurations
        self.configs_table.setRowCount(len(configurations))
        
        # Update combo boxes - handle both dict and object access
        config_names = []
        for config in configurations:
            if isinstance(config, dict):
                if 'name' in config:
                    config_names.append(config['name'])
            elif hasattr(config, 'name'):
                config_names.append(config.name)
        
        self.details_config_combo.clear()
        self.details_config_combo.addItems(config_names)
        
        # Update table - handle both dict and object access
        for row, config in enumerate(configurations):
            # Check if it's a dictionary or Pydantic model object
            if isinstance(config, dict):
                # Use fail-fast approach - all required fields must be present
                if 'name' not in config:
                    raise ValueError("Configuration missing required 'name' field")
                if 'type' not in config:
                    raise ValueError("Configuration missing required 'type' field")
                if 'provider' not in config:
                    raise ValueError("Configuration missing required 'provider' field")
                if 'status' not in config:
                    raise ValueError("Configuration missing required 'status' field")
                    
                name = config['name']
                config_type = config['type']
                provider = config['provider']
                status = config['status']
            else:
                # Handle Pydantic model objects directly
                if not hasattr(config, 'name'):
                    raise ValueError("Configuration object missing required 'name' attribute")
                if not hasattr(config, 'type'):
                    raise ValueError("Configuration object missing required 'type' attribute")
                if not hasattr(config, 'provider'):
                    raise ValueError("Configuration object missing required 'provider' attribute")
                if not hasattr(config, 'status'):
                    raise ValueError("Configuration object missing required 'status' attribute")
                    
                name = config.name
                config_type = str(config.type)  # Convert enum to string
                provider = config.provider
                status = str(config.status)  # Convert enum to string
            
            self.configs_table.setItem(row, 0, QTableWidgetItem(name))
            self.configs_table.setItem(row, 1, QTableWidgetItem(config_type))
            self.configs_table.setItem(row, 2, QTableWidgetItem(provider))
            self.configs_table.setItem(row, 3, QTableWidgetItem(status))
        
        # Apply current filters
        self.apply_filters()

    def on_configuration_created(self, config_name):
        """Handle configuration creation."""
        QMessageBox.information(self, "Success", f"Configuration '{config_name}' created successfully!")
        self.refresh_configurations()

    def on_configuration_deleted(self, config_name):
        """Handle configuration deletion."""
        QMessageBox.information(self, "Success", f"Configuration '{config_name}' deleted successfully!")
        self.refresh_configurations()

    def on_configuration_validated(self, config_name, validation_result):
        """Handle configuration validation."""
        # Fail-fast approach
        if 'is_valid' not in validation_result:
            raise ValueError("Validation result missing required 'is_valid' field")
        if 'errors' not in validation_result:
            raise ValueError("Validation result missing required 'errors' field")
        if 'warnings' not in validation_result:
            raise ValueError("Validation result missing required 'warnings' field")
            
        is_valid = validation_result['is_valid']
        errors = validation_result['errors']
        warnings = validation_result['warnings']
        
        result_text = f"Validation Result for '{config_name}':\n\n"
        result_text += f"Status: {'✅ Valid' if is_valid else '❌ Invalid'}\n\n"
        
        if errors:
            result_text += "Errors:\n"
            for error in errors:
                result_text += f"  • {error}\n"
            result_text += "\n"
        
        if warnings:
            result_text += "Warnings:\n"
            for warning in warnings:
                result_text += f"  • {warning}\n"
        
        if not errors and not warnings:
            result_text += "No issues found."
        
        self.validation_text.setText(result_text)

    def on_form_configuration_created(self, config_name: str, provider_type: str):
        """Handle form-based configuration creation."""
        try:
            logger.info(f"Form configuration created: {config_name} ({provider_type})")
            
            # Emit our own signal
            self.form_configuration_created.emit(config_name, provider_type)
            
            # Show success message with provider type info
            QMessageBox.information(
                self,
                "Configuration Created",
                f"Configuration '{config_name}' created successfully!\n\n"
                f"Provider Type: {provider_type.replace('_', ' ').title()}\n"
                f"Created using form-based interface."
            )
            
            # Refresh configurations list
            self.refresh_configurations()
            
        except Exception as e:
            logger.error(f"Failed to handle form configuration creation: {e}")

    def on_form_dialog_requested(self, provider_type: str):
        """Handle form dialog request."""
        try:
            logger.info(f"Form dialog requested for {provider_type}")
            self.form_dialog_opened.emit(provider_type)
            
        except Exception as e:
            logger.error(f"Failed to handle form dialog request: {e}")

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


