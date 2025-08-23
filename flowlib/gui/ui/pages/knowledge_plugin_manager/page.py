"""
Refactored Knowledge Plugin Manager Page following MVC pattern.

This demonstrates the new architecture with business logic controllers
and pure presentation layer.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                               QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
                               QLineEdit, QTextEdit, QTabWidget, QGroupBox, QGridLayout,
                               QMessageBox, QDialog, QDialogButtonBox, QFormLayout,
                               QCheckBox, QSpinBox, QFileDialog, QProgressBar, QListWidget,
                               QListWidgetItem, QSplitter, QTreeWidget, QTreeWidgetItem,
                               QScrollArea, QFrame)
from PySide6.QtCore import Qt, Signal
import logging
import json
from pathlib import Path
from typing import List, Union

from flowlib.gui.logic.config_manager.knowledge_plugin_controller import KnowledgePluginController
from flowlib.gui.logic.services.service_factory import ServiceFactory
from flowlib.gui.logic.services.base_controller import ControllerManager

logger = logging.getLogger(__name__)


class KnowledgePluginManagerPage(QWidget):
    """
    Knowledge Plugin Manager Page - Pure Presentation Layer
    
    Handles only UI concerns and delegates business logic to the controller.
    """
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing Knowledge Plugin Manager page")
        
        # Initialize MVC components
        self.service_factory = ServiceFactory()
        self.controller_manager = ControllerManager(self.service_factory)
        self.controller = self.controller_manager.get_controller_sync(KnowledgePluginController)
        
        # Controller is initialized in constructor - no deferred initialization
        
        # UI state
        self.plugins = []
        self.selected_plugin = None
        self.available_extractors = []
        
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
        self.controller.plugins_loaded.connect(self.update_plugins_list)
        self.controller.plugin_generated.connect(self.on_plugin_generated)
        self.controller.plugin_deleted.connect(self.on_plugin_deleted)
        self.controller.plugin_tested.connect(self.on_plugin_tested)
        self.controller.generation_progress.connect(self.on_generation_progress)
        self.controller.domain_strategies_loaded.connect(self.update_domain_strategies)
        
        # Connect common operation signals
        self.controller.operation_started.connect(self.on_operation_started)
        self.controller.operation_completed.connect(self.on_operation_completed)
        self.controller.operation_failed.connect(self.on_operation_failed)
        self.controller.progress_updated.connect(self.on_progress_updated)
        self.controller.status_updated.connect(self.on_status_updated)
    
    def get_title(self):
        """Get page title for navigation."""
        return "Knowledge Plugin Manager"
    
    def page_visible(self):
        """Called when page becomes visible - load data."""
        logger.debug("Knowledge Plugin Manager page became visible")
        if self.controller:
            logger.info("Loading knowledge plugins on page visibility")
            self.refresh_plugins()
            self.controller.get_available_domain_strategies()
        else:
            logger.warning("No controller available to load knowledge plugins")
    
    def get_state(self):
        """Get current page state for persistence."""
        return {
            "current_tab": self.tab_widget.currentIndex() if hasattr(self, 'tab_widget') else 0,
            "selected_plugin": self.selected_plugin,
            "generation_settings": self.get_generation_settings()
        }
    
    def set_state(self, state):
        """Set page state when loading."""
        if hasattr(self, 'tab_widget') and "current_tab" in state:
            self.tab_widget.setCurrentIndex(state["current_tab"])
        if "selected_plugin" in state:
            self.selected_plugin = state["selected_plugin"]
        if "generation_settings" in state:
            self.set_generation_settings(state["generation_settings"])
    
    def init_ui(self):
        """Initialize the user interface."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)

        # Title
        self.title = QLabel("Knowledge Plugin Manager")
        self.title.setStyleSheet("font-size: 24px; font-weight: bold; padding-bottom: 10px;")
        self.layout.addWidget(self.title)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        
        # Plugin List Tab
        self.plugins_tab = self.create_plugins_tab()
        self.tab_widget.addTab(self.plugins_tab, "Plugins")
        
        # Generation Tab
        self.generation_tab = self.create_generation_tab()
        self.tab_widget.addTab(self.generation_tab, "Generate Plugin")
        
        # Plugin Details Tab
        self.details_tab = self.create_details_tab()
        self.tab_widget.addTab(self.details_tab, "Plugin Details")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)
    
    def create_plugins_tab(self):
        """Create the plugins list tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Filter section
        filter_group = QGroupBox("Filters")
        filter_layout = QHBoxLayout(filter_group)
        layout.addWidget(filter_group)
        
        filter_layout.addWidget(QLabel("Status:"))
        self.status_filter = QComboBox()
        self.status_filter.addItems(["All", "Active", "Inactive", "Error"])
        self.status_filter.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.status_filter)
        
        # Search
        filter_layout.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search plugins...")
        self.search_edit.textChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.search_edit)
        
        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_plugins)
        filter_layout.addWidget(refresh_button)
        
        filter_layout.addStretch()
        
        # Action buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        
        self.validate_plugin_button = QPushButton("Validate Plugin")
        self.validate_plugin_button.clicked.connect(self.validate_plugin)
        self.validate_plugin_button.setEnabled(False)
        button_layout.addWidget(self.validate_plugin_button)
        
        self.test_plugin_button = QPushButton("Test Plugin")
        self.test_plugin_button.clicked.connect(self.test_plugin)
        self.test_plugin_button.setEnabled(False)
        button_layout.addWidget(self.test_plugin_button)
        
        self.delete_plugin_button = QPushButton("Delete Plugin")
        self.delete_plugin_button.clicked.connect(self.delete_plugin)
        self.delete_plugin_button.setEnabled(False)
        button_layout.addWidget(self.delete_plugin_button)
        
        button_layout.addStretch()
        
        # Plugins table
        self.plugins_table = QTableWidget()
        self.plugins_table.setColumnCount(6)
        self.plugins_table.setHorizontalHeaderLabels(["Name", "Version", "Status", "Domain", "Documents", "Created"])
        self.plugins_table.horizontalHeader().setStretchLastSection(True)
        self.plugins_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.plugins_table.selectionModel().selectionChanged.connect(self.on_plugin_selection_changed)
        layout.addWidget(self.plugins_table)
        
        return widget
    
    def create_generation_tab(self):
        """Create the plugin generation tab."""
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
        
        # Plugin Configuration
        config_group = QGroupBox("Plugin Configuration")
        config_layout = QFormLayout(config_group)
        layout.addWidget(config_group)
        
        self.plugin_name_edit = QLineEdit()
        config_layout.addRow("Plugin Name:", self.plugin_name_edit)
        
        self.plugin_domain_combo = QComboBox()
        self.plugin_domain_combo.setEditable(True)
        self.plugin_domain_combo.setToolTip("Select or enter the plugin domain")
        # Will be populated with domain strategies from flowlib registry
        config_layout.addRow("Domain:", self.plugin_domain_combo)
        
        self.plugin_description_edit = QTextEdit()
        self.plugin_description_edit.setMaximumHeight(100)
        config_layout.addRow("Description:", self.plugin_description_edit)
        
        # Document Sources
        sources_group = QGroupBox("Document Sources")
        sources_layout = QVBoxLayout(sources_group)
        layout.addWidget(sources_group)
        
        # Document list
        docs_button_layout = QHBoxLayout()
        self.add_document_button = QPushButton("Add Document")
        self.add_document_button.clicked.connect(self.add_document)
        docs_button_layout.addWidget(self.add_document_button)
        
        self.add_folder_button = QPushButton("Add Folder")
        self.add_folder_button.clicked.connect(self.add_folder)
        docs_button_layout.addWidget(self.add_folder_button)
        
        self.remove_document_button = QPushButton("Remove")
        self.remove_document_button.clicked.connect(self.remove_document)
        self.remove_document_button.setEnabled(False)
        docs_button_layout.addWidget(self.remove_document_button)
        
        docs_button_layout.addStretch()
        sources_layout.addLayout(docs_button_layout)
        
        self.documents_list = QListWidget()
        self.documents_list.selectionModel().selectionChanged.connect(self.on_document_selection_changed)
        sources_layout.addWidget(self.documents_list)
        
        # Extraction Settings
        extraction_group = QGroupBox("Extraction Settings")
        extraction_layout = QFormLayout(extraction_group)
        layout.addWidget(extraction_group)
        
        # Domain Strategy Selection
        self.domain_strategy_combo = QComboBox()
        self.domain_strategy_combo.setToolTip("Select the domain-specific extraction strategy")
        extraction_layout.addRow("Domain Strategy:", self.domain_strategy_combo)
        
        self.extractor_combo = QComboBox()
        extraction_layout.addRow("Extractor:", self.extractor_combo)
        
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(100, 10000)
        self.chunk_size_spin.setValue(1000)
        extraction_layout.addRow("Chunk Size:", self.chunk_size_spin)
        
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 500)
        self.overlap_spin.setValue(200)
        extraction_layout.addRow("Overlap:", self.overlap_spin)
        
        # Generation Actions
        actions_layout = QHBoxLayout()
        layout.addLayout(actions_layout)
        
        self.preview_button = QPushButton("Preview Generation")
        self.preview_button.clicked.connect(self.preview_generation)
        actions_layout.addWidget(self.preview_button)
        
        self.generate_button = QPushButton("Generate Plugin")
        self.generate_button.clicked.connect(self.generate_plugin)
        actions_layout.addWidget(self.generate_button)
        
        actions_layout.addStretch()
        
        # Add stretch to push everything to the top in the scroll area
        layout.addStretch()
        
        # Generation Progress and Status (outside scroll area so always visible)
        self.generation_progress = QProgressBar()
        self.generation_progress.setVisible(False)
        main_layout.addWidget(self.generation_progress)
        
        self.generation_status = QLabel()
        self.generation_status.setVisible(False)
        main_layout.addWidget(self.generation_status)
        
        return widget
    
    def create_details_tab(self):
        """Create the plugin details tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Plugin selection
        selection_group = QGroupBox("Plugin Selection")
        selection_layout = QHBoxLayout(selection_group)
        layout.addWidget(selection_group)
        
        selection_layout.addWidget(QLabel("Select Plugin:"))
        self.details_plugin_combo = QComboBox()
        self.details_plugin_combo.currentTextChanged.connect(self.on_details_plugin_changed)
        selection_layout.addWidget(self.details_plugin_combo)
        
        selection_layout.addStretch()
        
        # Plugin details
        details_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(details_splitter)
        
        # Manifest
        manifest_group = QGroupBox("Manifest")
        manifest_layout = QVBoxLayout(manifest_group)
        
        self.manifest_text = QTextEdit()
        self.manifest_text.setReadOnly(True)
        manifest_layout.addWidget(self.manifest_text)
        
        details_splitter.addWidget(manifest_group)
        
        # Files
        files_group = QGroupBox("Plugin Files")
        files_layout = QVBoxLayout(files_group)
        
        self.files_tree = QTreeWidget()
        self.files_tree.setHeaderLabel("Files")
        files_layout.addWidget(self.files_tree)
        
        details_splitter.addWidget(files_group)
        
        return widget
    
    # UI Action Methods (Pure Presentation Layer)
    def refresh_plugins(self):
        """Refresh plugins list."""
        if self.controller:
            self.controller.refresh_plugins()
    
    def validate_plugin(self):
        """Validate selected plugin."""
        if not self.selected_plugin:
            QMessageBox.warning(self, "No Selection", "Please select a plugin to validate.")
            return
        
        if self.controller:
            self.controller.validate_plugin(self.selected_plugin)
    
    def test_plugin(self):
        """Test selected plugin."""
        if not self.selected_plugin:
            QMessageBox.warning(self, "No Selection", "Please select a plugin to test.")
            return
        
        if self.controller:
            self.controller.test_plugin(self.selected_plugin)
    
    def delete_plugin(self):
        """Delete selected plugin."""
        if not self.selected_plugin:
            QMessageBox.warning(self, "No Selection", "Please select a plugin to delete.")
            return
        
        reply = QMessageBox.question(
            self, "Delete Plugin", 
            f"Are you sure you want to delete plugin '{self.selected_plugin}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes and self.controller:
            self.controller.delete_plugin(self.selected_plugin)
    
    def add_document(self):
        """Add a document to the generation sources."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Add Document", "", "All Files (*)"
        )
        if file_path:
            self.documents_list.addItem(file_path)
    
    def add_folder(self):
        """Add a folder to the generation sources."""
        folder_path = QFileDialog.getExistingDirectory(self, "Add Folder")
        if folder_path:
            self.documents_list.addItem(f"[FOLDER] {folder_path}")
    
    def remove_document(self):
        """Remove selected document from sources."""
        current_item = self.documents_list.currentItem()
        if current_item:
            self.documents_list.takeItem(self.documents_list.row(current_item))
    
    def preview_generation(self):
        """Preview plugin generation."""
        plugin_config = self.get_generation_config()
        if not plugin_config:
            return
        
        if self.controller:
            self.controller.preview_plugin_generation(plugin_config)
    
    def generate_plugin(self):
        """Generate a new plugin."""
        plugin_config = self.get_generation_config()
        if not plugin_config:
            return
        
        if self.controller:
            self.controller.generate_plugin(plugin_config)
    
    def get_generation_config(self):
        """Get plugin generation configuration from UI."""
        plugin_name = self.plugin_name_edit.text().strip()
        if not plugin_name:
            QMessageBox.warning(self, "Invalid Input", "Please enter a plugin name.")
            return None
        
        documents = []
        for i in range(self.documents_list.count()):
            documents.append(self.documents_list.item(i).text())
        
        if not documents:
            QMessageBox.warning(self, "Invalid Input", "Please add at least one document source.")
            return None
        
        return {
            "name": plugin_name,
            "domain": self.plugin_domain_combo.currentText().strip(),
            "description": self.plugin_description_edit.toPlainText().strip(),
            "documents": documents,
            "extractor": self.extractor_combo.currentText(),
            "domain_strategy": self.get_selected_domain_strategy(),
            "chunk_size": self.chunk_size_spin.value(),
            "overlap": self.overlap_spin.value()
        }
    
    def get_selected_domain_strategy(self):
        """Get the currently selected domain strategy ID."""
        current_data = self.domain_strategy_combo.currentData()
        if current_data:
            return current_data  # This will be the strategy ID (e.g., "generic", "software_engineering")
        return "generic"  # Default fallback
    
    def get_generation_settings(self):
        """Get current generation settings for state persistence."""
        return {
            "plugin_name": self.plugin_name_edit.text(),
            "plugin_domain": self.plugin_domain_combo.currentText(),
            "plugin_description": self.plugin_description_edit.toPlainText(),
            "extractor": self.extractor_combo.currentText(),
            "domain_strategy": self.get_selected_domain_strategy(),
            "chunk_size": self.chunk_size_spin.value(),
            "overlap": self.overlap_spin.value()
        }
    
    def set_generation_settings(self, settings):
        """Set generation settings from state."""
        # Fail-fast approach - require all settings
        if "plugin_name" not in settings:
            raise ValueError("Generation settings missing required 'plugin_name' field")
        if "plugin_domain" not in settings:
            raise ValueError("Generation settings missing required 'plugin_domain' field")
        if "plugin_description" not in settings:
            raise ValueError("Generation settings missing required 'plugin_description' field")
        if "extractor" not in settings:
            raise ValueError("Generation settings missing required 'extractor' field")
        if "domain_strategy" not in settings:
            raise ValueError("Generation settings missing required 'domain_strategy' field")
        if "chunk_size" not in settings:
            raise ValueError("Generation settings missing required 'chunk_size' field")
        if "overlap" not in settings:
            raise ValueError("Generation settings missing required 'overlap' field")
            
        self.plugin_name_edit.setText(settings["plugin_name"])
        self.plugin_domain_combo.setCurrentText(settings["plugin_domain"])
        self.plugin_description_edit.setPlainText(settings["plugin_description"])
        self.extractor_combo.setCurrentText(settings["extractor"])
        self.set_domain_strategy_selection(settings["domain_strategy"])
        self.chunk_size_spin.setValue(settings["chunk_size"])
        self.overlap_spin.setValue(settings["overlap"])
    
    def set_domain_strategy_selection(self, strategy_id):
        """Set the domain strategy selection by strategy ID."""
        for i in range(self.domain_strategy_combo.count()):
            if self.domain_strategy_combo.itemData(i) == strategy_id:
                self.domain_strategy_combo.setCurrentIndex(i)
                break
    
    def apply_filters(self):
        """Apply filters to plugins table."""
        status_filter = self.status_filter.currentText()
        search_text = self.search_edit.text().lower()
        
        for row in range(self.plugins_table.rowCount()):
            status_item = self.plugins_table.item(row, 2)
            name_item = self.plugins_table.item(row, 0)
            domain_item = self.plugins_table.item(row, 3)
            
            show_row = True
            
            # Status filter
            if status_filter != "All" and status_item and status_item.text() != status_filter:
                show_row = False
            
            # Search filter
            if search_text:
                match_found = False
                for item in [name_item, domain_item]:
                    if item and search_text in item.text().lower():
                        match_found = True
                        break
                if not match_found:
                    show_row = False
            
            self.plugins_table.setRowHidden(row, not show_row)
    
    def on_plugin_selection_changed(self):
        """Handle plugin selection changes."""
        selected_row = self.plugins_table.currentRow()
        if selected_row >= 0:
            self.selected_plugin = self.plugins_table.item(selected_row, 0).text()
            self.validate_plugin_button.setEnabled(True)
            self.test_plugin_button.setEnabled(True)
            self.delete_plugin_button.setEnabled(True)
            
            # Update details
            if self.controller:
                self.controller.get_plugin_details(self.selected_plugin)
        else:
            self.selected_plugin = None
            self.validate_plugin_button.setEnabled(False)
            self.test_plugin_button.setEnabled(False)
            self.delete_plugin_button.setEnabled(False)
    
    def on_document_selection_changed(self):
        """Handle document selection changes."""
        has_selection = self.documents_list.currentItem() is not None
        self.remove_document_button.setEnabled(has_selection)
    
    def on_details_plugin_changed(self, plugin_name):
        """Handle plugin details combo selection change."""
        if plugin_name and self.controller:
            self.controller.get_plugin_details(plugin_name)
            self.controller.get_plugin_manifest(plugin_name)
    
    # Controller Signal Handlers (Business Logic -> UI Updates)
    def update_plugins_list(self, plugins):
        """Update plugins list with new data."""
        self.plugins = plugins
        self.plugins_table.setRowCount(len(plugins))
        
        # Update combo boxes - fail-fast approach
        plugin_names = []
        for plugin in plugins:
            if 'name' not in plugin:
                raise ValueError("Plugin missing required 'name' field")
            plugin_names.append(plugin['name'])
        self.details_plugin_combo.clear()
        self.details_plugin_combo.addItems(plugin_names)
        
        # Update table - fail-fast approach
        for row, plugin in enumerate(plugins):
            if 'name' not in plugin:
                raise ValueError("Plugin missing required 'name' field")
            if 'version' not in plugin:
                raise ValueError("Plugin missing required 'version' field")
            if 'status' not in plugin:
                raise ValueError("Plugin missing required 'status' field")
            if 'domain' not in plugin:
                raise ValueError("Plugin missing required 'domain' field")
            if 'documents' not in plugin:
                raise ValueError("Plugin missing required 'documents' field")
            if 'created' not in plugin:
                raise ValueError("Plugin missing required 'created' field")
                
            name = plugin['name']
            version = plugin['version']
            status = plugin['status']
            domain = plugin['domain']
            doc_count = str(len(plugin['documents']))
            created = plugin['created']
            
            self.plugins_table.setItem(row, 0, QTableWidgetItem(name))
            self.plugins_table.setItem(row, 1, QTableWidgetItem(version))
            self.plugins_table.setItem(row, 2, QTableWidgetItem(status))
            self.plugins_table.setItem(row, 3, QTableWidgetItem(domain))
            self.plugins_table.setItem(row, 4, QTableWidgetItem(doc_count))
            self.plugins_table.setItem(row, 5, QTableWidgetItem(created))
        
        # Update extractors combo
        if hasattr(self, 'available_extractors'):
            self.extractor_combo.clear()
            self.extractor_combo.addItems(self.available_extractors)
        
        # Apply current filters
        self.apply_filters()
    
    def update_domain_strategies(self, strategies_list):
        """Update domain strategies combobox with data from flowlib registry."""
        self.domain_strategy_combo.clear()
        self.plugin_domain_combo.clear()
        
        for strategy in strategies_list:
            strategy_id = strategy['id']  # e.g., "generic", "software_engineering"
            strategy_name = strategy['name']  # e.g., "Generic", "Software Engineering"
            strategy_desc = strategy['description']  # Full description
            
            # Add item with user-friendly name and store ID as data to strategy combo
            self.domain_strategy_combo.addItem(strategy_name, strategy_id)
            
            # Set tooltip with description
            index = self.domain_strategy_combo.count() - 1
            self.domain_strategy_combo.setItemData(index, strategy_desc, Qt.ToolTipRole)
            
            # Also add to plugin domain combo with ID for easy typing
            self.plugin_domain_combo.addItem(strategy_id)
        
        # Set default to Generic if available
        self.set_domain_strategy_selection("generic")
        
        logger.info(f"Loaded {len(strategies_list)} domain strategies from flowlib registry")
    
    def on_plugin_generated(self, plugin_details):
        """Handle plugin generation completion."""
        self.generation_progress.setVisible(False)
        self.generation_status.setVisible(False)
        
        # Fail-fast approach
        if 'name' not in plugin_details:
            raise ValueError("Plugin details missing required 'name' field")
        plugin_name = plugin_details['name']
        QMessageBox.information(self, "Success", f"Plugin '{plugin_name}' generated successfully!")
        
        # Clear generation form
        self.plugin_name_edit.clear()
        self.plugin_domain_edit.clear()
        self.plugin_description_edit.clear()
        self.documents_list.clear()
        
        # Refresh plugins list
        self.refresh_plugins()
    
    def on_plugin_deleted(self, plugin_name):
        """Handle plugin deletion completion."""
        QMessageBox.information(self, "Success", f"Plugin '{plugin_name}' deleted successfully!")
        self.refresh_plugins()
    
    def on_plugin_tested(self, plugin_name, test_results):
        """Handle plugin testing completion."""
        results_text = json.dumps(test_results, indent=2)
        QMessageBox.information(self, "Test Results", f"Plugin '{plugin_name}' test results:\n{results_text}")
    
    def on_generation_progress(self, status_message, percentage):
        """Handle generation progress updates."""
        self.generation_progress.setVisible(True)
        self.generation_status.setVisible(True)
        self.generation_status.setText(status_message)
        self.generation_progress.setValue(percentage)
    
    def on_operation_started(self, operation_name):
        """Handle operation start."""
        if operation_name != "generate_plugin":  # Generation has its own progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
        logger.info(f"Operation started: {operation_name}")
    
    def on_operation_completed(self, operation_name, success, result):
        """Handle operation completion."""
        if operation_name != "generate_plugin":
            self.progress_bar.setVisible(False)
        logger.info(f"Operation completed: {operation_name} - Success: {success}")
    
    def on_operation_failed(self, operation_name, error_message):
        """Handle operation failure."""
        self.progress_bar.setVisible(False)
        self.generation_progress.setVisible(False)
        self.generation_status.setVisible(False)
        
        QMessageBox.critical(self, "Operation Failed", 
                           f"Operation '{operation_name}' failed:\n{error_message}")
        logger.error(f"Operation failed: {operation_name} - {error_message}")
    
    def on_progress_updated(self, operation_name, percentage):
        """Handle progress updates."""
        if operation_name != "generate_plugin":
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