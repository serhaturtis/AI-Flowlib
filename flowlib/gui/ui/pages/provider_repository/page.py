"""
Refactored Provider Repository Page following MVC pattern.

This demonstrates the new architecture with business logic controllers
and pure presentation layer.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QLabel, QPushButton, 
                               QGroupBox, QHBoxLayout, QComboBox, QTableWidget, QTableWidgetItem,
                               QHeaderView, QMessageBox, QTabWidget, QListWidget, QListWidgetItem,
                               QSplitter, QTextEdit, QProgressBar, QDialog, QFormLayout, QLineEdit,
                               QDialogButtonBox)
from PySide6.QtCore import Qt, Signal
import logging

from flowlib.gui.logic.config_manager.provider_repository_controller import ProviderRepositoryController
from flowlib.gui.logic.services.service_factory import ServiceFactory
from flowlib.gui.logic.services.base_controller import ControllerManager

logger = logging.getLogger(__name__)


class ProviderRepositoryPage(QWidget):
    """
    Provider Repository Page - Pure Presentation Layer
    
    Handles only UI concerns and delegates business logic to the controller.
    """
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing Provider Repository page")
        
        # Initialize MVC components
        self.service_factory = ServiceFactory()
        self.controller_manager = ControllerManager(self.service_factory)
        self.controller = self.controller_manager.get_controller_sync(ProviderRepositoryController)
        
        # UI state
        self.repository_overview = {}
        self.available_environments = []
        self.current_environment = ""
        
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
        self.controller.overview_loaded.connect(self.update_repository_overview)
        self.controller.environment_switched.connect(self.on_environment_switched)
        self.controller.role_assigned.connect(self.on_role_assigned)
        self.controller.role_unassigned.connect(self.on_role_unassigned)
        self.controller.backup_created.connect(self.on_backup_created)
        
        # Connect common operation signals
        self.controller.operation_started.connect(self.on_operation_started)
        self.controller.operation_completed.connect(self.on_operation_completed)
        self.controller.operation_failed.connect(self.on_operation_failed)
        self.controller.progress_updated.connect(self.on_progress_updated)
        self.controller.status_updated.connect(self.on_status_updated)
    
    def get_title(self):
        """Get page title for navigation."""
        return "Provider Repository"
    
    def page_visible(self):
        """Called when page becomes visible - load data."""
        logger.debug("Provider Repository page became visible")
        if self.controller:
            logger.info("Loading repository data on page visibility")
            self.refresh_data()
        else:
            logger.warning("No controller available to load repository data")
    
    def get_state(self):
        """Get page state for saving."""
        return {
            "current_tab": self.tab_widget.currentIndex() if hasattr(self, 'tab_widget') else 0,
            "selected_environment": self.env_combo.currentText() if hasattr(self, 'env_combo') else ""
        }
    
    def set_state(self, state):
        """Set page state when loading."""
        if hasattr(self, 'tab_widget') and "current_tab" in state:
            self.tab_widget.setCurrentIndex(state["current_tab"])
        if hasattr(self, 'env_combo') and "selected_environment" in state:
            env_text = state["selected_environment"]
            index = self.env_combo.findText(env_text)
            if index >= 0:
                self.env_combo.setCurrentIndex(index)
    
    def init_ui(self):
        """Initialize the user interface."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)

        # Title
        self.title = QLabel("Provider Repository Management")
        self.title.setStyleSheet("font-size: 24px; font-weight: bold; padding-bottom: 10px;")
        self.layout.addWidget(self.title)

        # Create tab widget for different sections
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        
        # Overview Tab
        self.overview_tab = self.create_overview_tab()
        self.tab_widget.addTab(self.overview_tab, "Overview")
        
        # Environment Management Tab
        self.env_tab = self.create_environment_tab()
        self.tab_widget.addTab(self.env_tab, "Environments")
        
        # Role Management Tab
        self.roles_tab = self.create_roles_tab()
        self.tab_widget.addTab(self.roles_tab, "Role Assignments")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)
    
    def create_overview_tab(self):
        """Create the overview tab with status information."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Status Section
        status_group = QGroupBox("Repository Status")
        status_layout = QGridLayout(status_group)
        layout.addWidget(status_group)

        self.current_environment_label = QLabel("Current Environment:")
        self.current_environment_value = QLabel("Loading...")
        status_layout.addWidget(self.current_environment_label, 0, 0)
        status_layout.addWidget(self.current_environment_value, 0, 1)

        self.total_configs_label = QLabel("Total Configurations:")
        self.total_configs_value = QLabel("Loading...")
        status_layout.addWidget(self.total_configs_label, 1, 0)
        status_layout.addWidget(self.total_configs_value, 1, 1)

        self.health_status_label = QLabel("Health Status:")
        self.health_status_value = QLabel("Loading...")
        status_layout.addWidget(self.health_status_label, 2, 0)
        status_layout.addWidget(self.health_status_value, 2, 1)

        # Role Assignments Section
        roles_group = QGroupBox("Role Assignment Summary")
        roles_layout = QGridLayout(roles_group)
        layout.addWidget(roles_group)

        self.assigned_roles_label = QLabel("Assigned Roles:")
        self.assigned_roles_value = QLabel("Loading...")
        roles_layout.addWidget(self.assigned_roles_label, 0, 0)
        roles_layout.addWidget(self.assigned_roles_value, 0, 1)

        self.missing_roles_label = QLabel("Missing Roles:")
        self.missing_roles_value = QLabel("Loading...")
        roles_layout.addWidget(self.missing_roles_label, 1, 0)
        roles_layout.addWidget(self.missing_roles_value, 1, 1)

        # Actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QHBoxLayout(actions_group)
        layout.addWidget(actions_group)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_data)
        actions_layout.addWidget(self.refresh_button)

        self.backup_button = QPushButton("Create Backup")
        self.backup_button.clicked.connect(self.create_backup)
        actions_layout.addWidget(self.backup_button)

        layout.addStretch()
        return widget
    
    def create_environment_tab(self):
        """Create the environment management tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Environment Selection
        env_group = QGroupBox("Environment Management")
        env_layout = QVBoxLayout(env_group)
        layout.addWidget(env_group)
        
        # Current environment switcher
        switch_layout = QHBoxLayout()
        switch_layout.addWidget(QLabel("Switch to Environment:"))
        
        self.env_combo = QComboBox()
        self.env_combo.addItems(["development", "staging", "production"])
        switch_layout.addWidget(self.env_combo)
        
        self.switch_button = QPushButton("Switch")
        self.switch_button.clicked.connect(self.switch_environment)
        switch_layout.addWidget(self.switch_button)
        
        switch_layout.addStretch()
        env_layout.addLayout(switch_layout)
        
        # Environment details
        self.env_details = QTextEdit()
        self.env_details.setReadOnly(True)
        self.env_details.setMaximumHeight(200)
        env_layout.addWidget(self.env_details)
        
        layout.addStretch()
        return widget
    
    def create_roles_tab(self):
        """Create the role assignments tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Role assignments table
        roles_group = QGroupBox("Role Assignments")
        roles_layout = QVBoxLayout(roles_group)
        layout.addWidget(roles_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.assign_role_button = QPushButton("Assign Role")
        self.assign_role_button.clicked.connect(self.assign_role)
        button_layout.addWidget(self.assign_role_button)
        
        self.unassign_role_button = QPushButton("Unassign Role")
        self.unassign_role_button.clicked.connect(self.unassign_role)
        self.unassign_role_button.setEnabled(False)
        button_layout.addWidget(self.unassign_role_button)
        
        button_layout.addStretch()
        roles_layout.addLayout(button_layout)
        
        # Roles table
        self.roles_table = QTableWidget()
        self.roles_table.setColumnCount(3)
        self.roles_table.setHorizontalHeaderLabels(["Role", "Assigned Configuration", "Status"])
        self.roles_table.horizontalHeader().setStretchLastSection(True)
        self.roles_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.roles_table.selectionModel().selectionChanged.connect(self.on_role_selection_changed)
        roles_layout.addWidget(self.roles_table)
        
        layout.addStretch()
        return widget
    
    # UI Action Methods (Pure Presentation Layer)
    def refresh_data(self):
        """Refresh all repository data."""
        if self.controller:
            self.controller.get_repository_overview()
            self.controller.get_available_environments()
    
    def switch_environment(self):
        """Switch to selected environment."""
        environment = self.env_combo.currentText()
        if self.controller:
            self.controller.switch_environment(environment)
    
    def create_backup(self):
        """Create a repository backup."""
        if self.controller:
            self.controller.create_backup()
    
    def assign_role(self):
        """Assign a role to a configuration."""
        dialog = RoleAssignmentDialog(self)
        if dialog.exec() == QDialog.Accepted:
            role_data = dialog.get_role_data()
            
            if self.controller:
                self.controller.assign_role(role_data['role'], role_data['config'])
                
                # Update the roles table
                self.add_role_to_table(role_data['role'], role_data['config'])
                
                QMessageBox.information(
                    self, "Role Assigned", 
                    f"Role '{role_data['role']}' assigned to configuration '{role_data['config']}'"
                )
    
    def unassign_role(self):
        """Unassign selected role."""
        selected_row = self.roles_table.currentRow()
        if selected_row >= 0:
            role_name = self.roles_table.item(selected_row, 0).text()
            
            reply = QMessageBox.question(
                self, "Unassign Role", 
                f"Are you sure you want to unassign role '{role_name}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes and self.controller:
                self.controller.unassign_role(role_name)
    
    def on_role_selection_changed(self):
        """Handle role selection changes."""
        has_selection = self.roles_table.currentRow() >= 0
        self.unassign_role_button.setEnabled(has_selection)
    
    # Controller Signal Handlers (Business Logic -> UI Updates)
    def update_repository_overview(self, overview):
        """Update repository overview display."""
        self.repository_overview = overview
        
        # Update status labels - fail-fast approach (overview is dictionary from OperationData)
        if 'active_environment' not in overview or overview['active_environment'] is None:
            raise ValueError("Repository overview missing required 'active_environment' field")
        if 'total_configurations' not in overview or overview['total_configurations'] is None:
            raise ValueError("Repository overview missing required 'total_configurations' field")
            
        self.current_environment_value.setText(overview['active_environment'])
        self.total_configs_value.setText(str(overview['total_configurations']))
        # Note: health_status not in OperationData model - will add if needed
        self.health_status_value.setText('Healthy')  # Default until we add health_status field
        
        # Update role assignments - using available data (roles not implemented in current data model)
        assigned_roles = overview.get('assigned_roles', [])
        missing_roles = overview.get('missing_roles', [])
        
        self.assigned_roles_value.setText(f"{len(assigned_roles)} roles assigned")
        self.missing_roles_value.setText(f"{len(missing_roles)} roles missing")
        
        # Update roles table
        self.update_roles_table(assigned_roles, missing_roles)
        
        # Update environment details using available data  
        details_text = f"Environment: {overview['active_environment']}\n"
        details_text += f"Status: Active\n"
        details_text += f"Last Updated: {overview['last_updated']}\n"
        details_text += f"Configurations: {overview['total_configurations']}\n"
        details_text += f"Plugins: {overview['total_plugins']}\n"
        details_text += f"Backups: {overview['total_backups']}\n"
        details_text += f"Size: {overview['total_size_mb']} MB"
        self.env_details.setText(details_text)
    
    def update_roles_table(self, assigned_roles, missing_roles):
        """Update the roles table with current assignments."""
        all_roles = []
        
        # Add assigned roles - fail-fast approach
        for role in assigned_roles:
            if 'name' not in role:
                raise ValueError("Assigned role missing required 'name' field")
            if 'provider' not in role:
                raise ValueError("Assigned role missing required 'provider' field")
            all_roles.append({
                'role': role['name'],
                'provider': role['provider'],
                'status': 'Assigned'
            })
        
        # Add missing roles - fail-fast approach
        for role in missing_roles:
            if 'name' not in role:
                raise ValueError("Missing role missing required 'name' field")
            all_roles.append({
                'role': role['name'],
                'provider': 'None',
                'status': 'Missing'
            })
        
        self.roles_table.setRowCount(len(all_roles))
        
        for row, role_data in enumerate(all_roles):
            self.roles_table.setItem(row, 0, QTableWidgetItem(role_data['role']))
            self.roles_table.setItem(row, 1, QTableWidgetItem(role_data['provider']))
            self.roles_table.setItem(row, 2, QTableWidgetItem(role_data['status']))
    
    def add_role_to_table(self, role_name: str, config_name: str):
        """Add a new role assignment to the table."""
        row_count = self.roles_table.rowCount()
        self.roles_table.insertRow(row_count)
        
        self.roles_table.setItem(row_count, 0, QTableWidgetItem(role_name))
        self.roles_table.setItem(row_count, 1, QTableWidgetItem(config_name))
        self.roles_table.setItem(row_count, 2, QTableWidgetItem('Assigned'))
    
    def on_environment_switched(self, environment_name):
        """Handle environment switch completion."""
        self.current_environment = environment_name
        self.current_environment_value.setText(environment_name)
        QMessageBox.information(self, "Success", f"Switched to environment: {environment_name}")
        self.refresh_data()
    
    def on_role_assigned(self, role_name, config_name):
        """Handle role assignment completion."""
        QMessageBox.information(self, "Success", f"Role '{role_name}' assigned to '{config_name}'")
        self.refresh_data()
    
    def on_role_unassigned(self, role_name):
        """Handle role unassignment completion."""
        QMessageBox.information(self, "Success", f"Role '{role_name}' unassigned")
        self.refresh_data()
    
    def on_backup_created(self, backup_path):
        """Handle backup creation completion."""
        QMessageBox.information(self, "Success", f"Backup created at: {backup_path}")
    
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


class RoleAssignmentDialog(QDialog):
    """Dialog for assigning roles to providers."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Role to Provider")
        self.setModal(True)
        self.resize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Form layout
        form_layout = QFormLayout()
        
        # Role selection (semantic role names that flows actually use)
        self.role_combo = QComboBox()
        self.role_combo.setEditable(True)
        self.role_combo.addItems([
            "knowledge-extraction", "default-llm", "default-vector-db", "default-graph-db",
            "default-embedding", "conversation-llm", "backup-llm", "primary-vector-db",
            "memory-manager", "analysis-llm", "generation-llm", "classification-llm"
        ])
        form_layout.addRow("Semantic Role Name:", self.role_combo)
        
        # Configuration selection (canonical names of existing configurations)
        self.config_combo = QComboBox()
        self.config_combo.setEditable(False)  # Only existing configs
        self._populate_existing_configurations()
        form_layout.addRow("Assign to Configuration:", self.config_combo)
        
        # Description
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(100)
        self.description_edit.setPlaceholderText("Optional description for this role assignment...")
        form_layout.addRow("Description:", self.description_edit)
        
        layout.addLayout(form_layout)
        
        # Priority selection
        priority_group = QGroupBox("Priority")
        priority_layout = QHBoxLayout(priority_group)
        
        self.priority_combo = QComboBox()
        self.priority_combo.addItems(["High", "Medium", "Low"])
        self.priority_combo.setCurrentText("Medium")
        priority_layout.addWidget(self.priority_combo)
        
        layout.addWidget(priority_group)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Validation
        button_box.button(QDialogButtonBox.Ok).setEnabled(False)
        self.role_combo.currentTextChanged.connect(self.validate_input)
        self.config_combo.currentTextChanged.connect(self.validate_input)
        
        # Initial validation
        self.validate_input()
    
    def validate_input(self):
        """Validate input and enable/disable OK button."""
        role_text = self.role_combo.currentText().strip()
        config_text = self.config_combo.currentText().strip()
        
        is_valid = bool(role_text and config_text)
        
        ok_button = self.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok)
        if ok_button:
            ok_button.setEnabled(is_valid)
    
    def get_role_data(self):
        """Get role assignment data from dialog."""
        return {
            "role": self.role_combo.currentText().strip(),
            "config": self.config_combo.currentText().strip(),
            "description": self.description_edit.toPlainText().strip(),
            "priority": self.priority_combo.currentText()
        }
    
    def _populate_existing_configurations(self):
        """Populate config combo with existing canonical configuration names."""
        try:
            from flowlib.resources.registry.registry import resource_registry
            
            # Get all existing configurations from resource registry
            config_names = []
            
            # Get all resource types and their configurations
            for resource_type in resource_registry.list_types():
                configs_of_type = resource_registry.get_by_type(resource_type)
                for config_name in configs_of_type.keys():
                    config_names.append(config_name)
            
            # Sort and add to combo box
            config_names.sort()
            self.config_combo.addItems(config_names)
            
            if not config_names:
                self.config_combo.addItem("No configurations available")
                self.config_combo.setEnabled(False)
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to populate configurations: {e}")
            self.config_combo.addItem("Error loading configurations")
            self.config_combo.setEnabled(False)