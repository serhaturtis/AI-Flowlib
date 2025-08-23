"""
Refactored Testing Tools Page following MVC pattern.

This demonstrates the new architecture with business logic controllers
and pure presentation layer.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                               QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
                               QLineEdit, QTextEdit, QTabWidget, QGroupBox, QGridLayout,
                               QMessageBox, QDialog, QDialogButtonBox, QFormLayout,
                               QCheckBox, QSpinBox, QFileDialog, QProgressBar, QListWidget,
                               QListWidgetItem, QSplitter, QTreeWidget, QTreeWidgetItem,
                               QScrollArea)
from PySide6.QtCore import Qt, Signal, QDateTime
import logging
import json
from datetime import datetime
from typing import List, Union

from flowlib.gui.logic.config_manager.testing_controller import TestingController
from flowlib.gui.logic.services.service_factory import ServiceFactory
from flowlib.gui.logic.services.base_controller import ControllerManager

logger = logging.getLogger(__name__)


class TestingToolsPage(QWidget):
    """
    Testing Tools Page - Pure Presentation Layer
    
    Handles only UI concerns and delegates business logic to the controller.
    """
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing Testing Tools page")
        
        # Initialize MVC components
        self.service_factory = ServiceFactory()
        self.controller_manager = ControllerManager(self.service_factory)
        self.controller = self.controller_manager.get_controller_sync(TestingController)
        
        # Controller is initialized in constructor - no deferred initialization
        
        # UI state
        self.test_history = []
        self.available_tests = []
        self.selected_test = None
        
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
        self.controller.test_completed.connect(self.on_test_completed)
        self.controller.test_history_loaded.connect(self.on_test_history_loaded)
        self.controller.test_suite_executed.connect(self.on_test_suite_executed)
        self.controller.provider_tested.connect(self.on_provider_tested)
        self.controller.flow_tested.connect(self.on_flow_tested)
        
        # Connect common operation signals
        self.controller.operation_started.connect(self.on_operation_started)
        self.controller.operation_completed.connect(self.on_operation_completed)
        self.controller.operation_failed.connect(self.on_operation_failed)
        self.controller.progress_updated.connect(self.on_progress_updated)
        self.controller.status_updated.connect(self.on_status_updated)
    
    def get_title(self):
        """Get page title for navigation."""
        return "Testing Tools"
    
    def page_visible(self):
        """Called when page becomes visible - load data."""
        logger.debug("Testing Tools page became visible")
        if self.controller:
            logger.info("Loading test data on page visibility")
            self.refresh_test_data()
        else:
            logger.warning("No controller available to load test data")
    
    def refresh_test_data(self):
        """Refresh test data when page becomes visible."""
        logger.debug("Refreshing test data...")
        if self.controller:
            self.controller.get_test_history()
            self.controller.get_available_tests()
            logger.debug("Test data refresh initiated")
        else:
            logger.warning("No controller available to refresh test data")
    
    def get_state(self):
        """Get current page state for persistence."""
        return {
            "current_tab": self.tab_widget.currentIndex() if hasattr(self, 'tab_widget') else 0,
            "test_type": self.test_type_combo.currentText() if hasattr(self, 'test_type_combo') else "Provider",
            "selected_test": self.selected_test
        }
    
    def set_state(self, state):
        """Set page state when loading."""
        if hasattr(self, 'tab_widget') and "current_tab" in state:
            self.tab_widget.setCurrentIndex(state["current_tab"])
        if hasattr(self, 'test_type_combo') and "test_type" in state:
            self.test_type_combo.setCurrentText(state["test_type"])
        if "selected_test" in state:
            self.selected_test = state["selected_test"]
    
    def init_ui(self):
        """Initialize the user interface."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)

        # Title
        self.title = QLabel("Testing Tools")
        self.title.setStyleSheet("font-size: 24px; font-weight: bold; padding-bottom: 10px;")
        self.layout.addWidget(self.title)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        
        # Run Tests Tab
        self.run_tests_tab = self.create_run_tests_tab()
        self.tab_widget.addTab(self.run_tests_tab, "Run Tests")
        
        # Test History Tab
        self.history_tab = self.create_history_tab()
        self.tab_widget.addTab(self.history_tab, "Test History")
        
        # Test Suites Tab
        self.suites_tab = self.create_suites_tab()
        self.tab_widget.addTab(self.suites_tab, "Test Suites")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)
    
    def create_run_tests_tab(self):
        """Create the run tests tab."""
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
        
        # Test Configuration
        config_group = QGroupBox("Test Configuration")
        config_layout = QFormLayout(config_group)
        layout.addWidget(config_group)
        
        self.test_type_combo = QComboBox()
        self.test_type_combo.addItems(["Provider", "Flow", "Configuration"])
        self.test_type_combo.currentTextChanged.connect(self.on_test_type_changed)
        config_layout.addRow("Test Type:", self.test_type_combo)
        
        self.test_target_combo = QComboBox()
        config_layout.addRow("Test Target:", self.test_target_combo)
        
        # Test Options
        options_group = QGroupBox("Test Options")
        options_layout = QFormLayout(options_group)
        layout.addWidget(options_group)
        
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(1, 300)
        self.timeout_spin.setValue(30)
        self.timeout_spin.setSuffix(" seconds")
        options_layout.addRow("Timeout:", self.timeout_spin)
        
        self.detailed_output_check = QCheckBox("Detailed Output")
        self.detailed_output_check.setChecked(True)
        options_layout.addRow(self.detailed_output_check)
        
        self.stop_on_error_check = QCheckBox("Stop on First Error")
        options_layout.addRow(self.stop_on_error_check)
        
        # Test Actions
        actions_layout = QHBoxLayout()
        layout.addLayout(actions_layout)
        
        self.run_test_button = QPushButton("Run Test")
        self.run_test_button.clicked.connect(self.run_test)
        actions_layout.addWidget(self.run_test_button)
        
        self.validate_environment_button = QPushButton("Validate Environment")
        self.validate_environment_button.clicked.connect(self.validate_environment)
        actions_layout.addWidget(self.validate_environment_button)
        
        actions_layout.addStretch()
        
        # Test Results
        results_group = QGroupBox("Test Results")
        results_layout = QVBoxLayout(results_group)
        layout.addWidget(results_group)
        
        self.test_results_text = QTextEdit()
        self.test_results_text.setReadOnly(True)
        results_layout.addWidget(self.test_results_text)
        
        # Results actions
        results_actions_layout = QHBoxLayout()
        self.export_results_button = QPushButton("Export Results")
        self.export_results_button.clicked.connect(self.export_results)
        self.export_results_button.setEnabled(False)
        results_actions_layout.addWidget(self.export_results_button)
        
        self.clear_results_button = QPushButton("Clear Results")
        self.clear_results_button.clicked.connect(self.clear_results)
        results_actions_layout.addWidget(self.clear_results_button)
        
        results_actions_layout.addStretch()
        results_layout.addLayout(results_actions_layout)
        
        # Add stretch to push everything to the top in the scroll area
        layout.addStretch()
        
        return widget
    
    def create_history_tab(self):
        """Create the test history tab."""
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
        
        # History Controls
        controls_group = QGroupBox("History Controls")
        controls_layout = QHBoxLayout(controls_group)
        layout.addWidget(controls_group)
        
        self.refresh_history_button = QPushButton("Refresh History")
        self.refresh_history_button.clicked.connect(self.refresh_history)
        controls_layout.addWidget(self.refresh_history_button)
        
        self.clear_history_button = QPushButton("Clear History")
        self.clear_history_button.clicked.connect(self.clear_history)
        controls_layout.addWidget(self.clear_history_button)
        
        controls_layout.addStretch()
        
        # History Filter
        controls_layout.addWidget(QLabel("Filter:"))
        self.history_filter_combo = QComboBox()
        self.history_filter_combo.addItems(["All", "Passed", "Failed", "Error"])
        self.history_filter_combo.currentTextChanged.connect(self.apply_history_filter)
        controls_layout.addWidget(self.history_filter_combo)
        
        # History Table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(["Test Name", "Type", "Target", "Status", "Duration", "Timestamp"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.selectionModel().selectionChanged.connect(self.on_history_selection_changed)
        layout.addWidget(self.history_table)
        
        # History Details
        details_group = QGroupBox("Test Details")
        details_layout = QVBoxLayout(details_group)
        layout.addWidget(details_group)
        
        self.history_details_text = QTextEdit()
        self.history_details_text.setReadOnly(True)
        self.history_details_text.setMaximumHeight(200)
        details_layout.addWidget(self.history_details_text)
        
        # Add stretch to push everything to the top in the scroll area
        layout.addStretch()
        
        return widget
    
    def create_suites_tab(self):
        """Create the test suites tab."""
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
        
        # Suite Selection
        suite_group = QGroupBox("Test Suites")
        suite_layout = QVBoxLayout(suite_group)
        layout.addWidget(suite_group)
        
        # Suite controls
        suite_controls_layout = QHBoxLayout()
        self.create_suite_button = QPushButton("Create Suite")
        self.create_suite_button.clicked.connect(self.create_test_suite)
        suite_controls_layout.addWidget(self.create_suite_button)
        
        self.edit_suite_button = QPushButton("Edit Suite")
        self.edit_suite_button.clicked.connect(self.edit_test_suite)
        self.edit_suite_button.setEnabled(False)
        suite_controls_layout.addWidget(self.edit_suite_button)
        
        self.delete_suite_button = QPushButton("Delete Suite")
        self.delete_suite_button.clicked.connect(self.delete_test_suite)
        self.delete_suite_button.setEnabled(False)
        suite_controls_layout.addWidget(self.delete_suite_button)
        
        suite_controls_layout.addStretch()
        suite_layout.addLayout(suite_controls_layout)
        
        # Suite list
        self.suites_list = QListWidget()
        self.suites_list.selectionModel().selectionChanged.connect(self.on_suite_selection_changed)
        suite_layout.addWidget(self.suites_list)
        
        # Suite execution
        execution_group = QGroupBox("Suite Execution")
        execution_layout = QVBoxLayout(execution_group)
        layout.addWidget(execution_group)
        
        # Execution options
        exec_options_layout = QFormLayout()
        
        self.parallel_execution_check = QCheckBox("Parallel Execution")
        exec_options_layout.addRow(self.parallel_execution_check)
        
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(1, 10)
        self.max_workers_spin.setValue(4)
        self.max_workers_spin.setEnabled(False)
        exec_options_layout.addRow("Max Workers:", self.max_workers_spin)
        
        self.parallel_execution_check.toggled.connect(self.max_workers_spin.setEnabled)
        
        execution_layout.addLayout(exec_options_layout)
        
        # Execute button
        self.execute_suite_button = QPushButton("Execute Suite")
        self.execute_suite_button.clicked.connect(self.execute_test_suite)
        self.execute_suite_button.setEnabled(False)
        execution_layout.addWidget(self.execute_suite_button)
        
        # Suite results
        self.suite_results_text = QTextEdit()
        self.suite_results_text.setReadOnly(True)
        execution_layout.addWidget(self.suite_results_text)
        
        # Add stretch to push everything to the top in the scroll area
        layout.addStretch()
        
        return widget
    
    # UI Action Methods (Pure Presentation Layer)
    def on_test_type_changed(self, test_type):
        """Handle test type selection change."""
        self.test_target_combo.clear()
        
        if test_type == "Provider":
            self.test_target_combo.addItems(["llamacpp", "sqlite", "chromadb", "memory", "local"])
        elif test_type == "Flow":
            self.test_target_combo.addItems(["classification", "conversation", "memory", "planning"])
        elif test_type == "Configuration":
            self.test_target_combo.addItems(["default-llm", "default-db", "default-vector"])
    
    def run_test(self):
        """Run the configured test."""
        test_type = self.test_type_combo.currentText()
        test_target = self.test_target_combo.currentText()
        
        if not test_target:
            QMessageBox.warning(self, "Invalid Configuration", "Please select a test target.")
            return
        
        test_config = {
            "timeout": self.timeout_spin.value(),
            "detailed_output": self.detailed_output_check.isChecked(),
            "stop_on_error": self.stop_on_error_check.isChecked()
        }
        
        if self.controller:
            if test_type == "Provider":
                self.controller.run_provider_test(test_target, test_config)
            elif test_type == "Flow":
                self.controller.run_flow_test(test_target, test_config)
            elif test_type == "Configuration":
                self.controller.run_configuration_test(test_target, test_config)
    
    def validate_environment(self):
        """Validate the testing environment."""
        if self.controller:
            self.controller.validate_test_environment()
    
    def export_results(self):
        """Export test results to file."""
        if not self.test_results_text.toPlainText():
            QMessageBox.warning(self, "No Results", "No test results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Test Results", "test_results.txt", "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.test_results_text.toPlainText())
                QMessageBox.information(self, "Export Complete", f"Results exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")
    
    def clear_results(self):
        """Clear test results."""
        self.test_results_text.clear()
        self.export_results_button.setEnabled(False)
    
    def refresh_history(self):
        """Refresh test history."""
        if self.controller:
            self.controller.get_test_history()
    
    def clear_history(self):
        """Clear test history."""
        reply = QMessageBox.question(
            self, "Clear History", 
            "Are you sure you want to clear all test history?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes and self.controller:
            self.controller.clear_test_history()
    
    def apply_history_filter(self):
        """Apply filter to history table."""
        filter_text = self.history_filter_combo.currentText()
        
        for row in range(self.history_table.rowCount()):
            status_item = self.history_table.item(row, 3)
            if filter_text == "All" or (status_item and status_item.text() == filter_text):
                self.history_table.setRowHidden(row, False)
            else:
                self.history_table.setRowHidden(row, True)
    
    def on_history_selection_changed(self):
        """Handle history selection changes."""
        selected_row = self.history_table.currentRow()
        if selected_row >= 0:
            # Get test details for selected row
            test_name = self.history_table.item(selected_row, 0).text()
            # This would normally fetch detailed results
            self.history_details_text.setText(f"Details for test: {test_name}")
    
    def create_test_suite(self):
        """Create a new test suite."""
        dialog = TestSuiteDialog(self)
        if dialog.exec() == QDialog.Accepted:
            suite_data = dialog.get_suite_data()
            # This would create the suite via controller
            QMessageBox.information(self, "Suite Created", f"Test suite '{suite_data['name']}' created.")
    
    def edit_test_suite(self):
        """Edit selected test suite."""
        selected_item = self.suites_list.currentItem()
        if selected_item:
            suite_name = selected_item.text()
            
            # Get current suite data (in a real implementation, this would load from storage)
            current_suite_data = {
                "name": suite_name,
                "description": f"Description for {suite_name}",
                "tests": ["Provider: llamacpp", "Configuration: default-llm"]  # Example tests
            }
            
            # Open edit dialog with current data
            dialog = TestSuiteDialog(self, edit_mode=True, suite_data=current_suite_data)
            dialog.setWindowTitle(f"Edit Test Suite - {suite_name}")
            
            if dialog.exec() == QDialog.Accepted:
                updated_suite_data = dialog.get_suite_data()
                
                # Update the suite (in a real implementation, this would save to storage)
                if self.controller:
                    # Controller method would handle the update
                    pass
                
                # Update the UI
                if updated_suite_data["name"] != suite_name:
                    selected_item.setText(updated_suite_data["name"])
                
                QMessageBox.information(
                    self, "Suite Updated", 
                    f"Test suite '{updated_suite_data['name']}' updated successfully."
                )
    
    def delete_test_suite(self):
        """Delete selected test suite."""
        selected_item = self.suites_list.currentItem()
        if selected_item:
            suite_name = selected_item.text()
            reply = QMessageBox.question(
                self, "Delete Suite", 
                f"Are you sure you want to delete test suite '{suite_name}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.suites_list.takeItem(self.suites_list.row(selected_item))
                QMessageBox.information(self, "Suite Deleted", f"Test suite '{suite_name}' deleted.")
    
    def execute_test_suite(self):
        """Execute selected test suite."""
        selected_item = self.suites_list.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "No Selection", "Please select a test suite to execute.")
            return
        
        suite_name = selected_item.text()
        suite_options = {
            "parallel": self.parallel_execution_check.isChecked(),
            "max_workers": self.max_workers_spin.value()
        }
        
        if self.controller:
            self.controller.run_test_suite(suite_name, suite_options)
    
    def on_suite_selection_changed(self):
        """Handle suite selection changes."""
        has_selection = self.suites_list.currentItem() is not None
        self.edit_suite_button.setEnabled(has_selection)
        self.delete_suite_button.setEnabled(has_selection)
        self.execute_suite_button.setEnabled(has_selection)
    
    # Controller Signal Handlers (Business Logic -> UI Updates)
    def on_test_completed(self, test_name, test_results):
        """Handle test completion."""
        results_text = f"Test: {test_name}\n"
        # Fail-fast approach
        if 'status' not in test_results:
            raise ValueError("Test results missing required 'status' field")
        if 'duration' not in test_results:
            raise ValueError("Test results missing required 'duration' field")
            
        results_text += f"Status: {test_results['status']}\n"
        results_text += f"Duration: {test_results['duration']}\n"
        results_text += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if 'details' in test_results:
            results_text += "Details:\n"
            results_text += json.dumps(test_results['details'], indent=2)
        
        self.test_results_text.setText(results_text)
        self.export_results_button.setEnabled(True)
    
    def on_test_history_loaded(self, history):
        """Handle test history loading."""
        self.test_history = history
        self.history_table.setRowCount(len(history))
        
        for row, test_entry in enumerate(history):
            # Fail-fast approach - all fields required
            if 'name' not in test_entry:
                raise ValueError("Test entry missing required 'name' field")
            if 'type' not in test_entry:
                raise ValueError("Test entry missing required 'type' field")
            if 'target' not in test_entry:
                raise ValueError("Test entry missing required 'target' field")
            if 'status' not in test_entry:
                raise ValueError("Test entry missing required 'status' field")
            if 'duration' not in test_entry:
                raise ValueError("Test entry missing required 'duration' field")
            if 'timestamp' not in test_entry:
                raise ValueError("Test entry missing required 'timestamp' field")
                
            self.history_table.setItem(row, 0, QTableWidgetItem(test_entry['name']))
            self.history_table.setItem(row, 1, QTableWidgetItem(test_entry['type']))
            self.history_table.setItem(row, 2, QTableWidgetItem(test_entry['target']))
            self.history_table.setItem(row, 3, QTableWidgetItem(test_entry['status']))
            self.history_table.setItem(row, 4, QTableWidgetItem(test_entry['duration']))
            self.history_table.setItem(row, 5, QTableWidgetItem(test_entry['timestamp']))
        
        self.apply_history_filter()
    
    def on_test_suite_executed(self, suite_name, results):
        """Handle test suite execution completion."""
        results_text = f"Suite: {suite_name}\n"
        # Fail-fast approach
        if 'total_tests' not in results:
            raise ValueError("Suite results missing required 'total_tests' field")
        if 'passed' not in results:
            raise ValueError("Suite results missing required 'passed' field")
        if 'failed' not in results:
            raise ValueError("Suite results missing required 'failed' field")
        if 'duration' not in results:
            raise ValueError("Suite results missing required 'duration' field")
            
        results_text += f"Total Tests: {results['total_tests']}\n"
        results_text += f"Passed: {results['passed']}\n"
        results_text += f"Failed: {results['failed']}\n"
        results_text += f"Duration: {results['duration']}\n\n"
        
        if 'test_results' in results:
            results_text += "Individual Test Results:\n"
            for test_result in results['test_results']:
                # Each test result must have required fields
                if 'name' not in test_result:
                    raise ValueError("Test result missing required 'name' field")
                if 'status' not in test_result:
                    raise ValueError("Test result missing required 'status' field")
                results_text += f"  {test_result['name']}: {test_result['status']}\n"
        
        self.suite_results_text.setText(results_text)
    
    def on_provider_tested(self, provider_name, success, results):
        """Handle provider test completion."""
        status = "PASSED" if success else "FAILED"
        self.on_test_completed(f"Provider: {provider_name}", {
            'status': status,
            'duration': results['duration'] if 'duration' in results else 'Unknown',
            'details': results
        })
    
    def on_flow_tested(self, flow_name, success, results):
        """Handle flow test completion."""
        status = "PASSED" if success else "FAILED"
        self.on_test_completed(f"Flow: {flow_name}", {
            'status': status,
            'duration': results['duration'] if 'duration' in results else 'Unknown',
            'details': results
        })
    
    def on_operation_started(self, operation_name):
        """Handle operation start."""
        logger.info(f"GUI received operation_started signal: {operation_name}")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        logger.info(f"Progress bar set to visible for operation: {operation_name}")
    
    def on_operation_completed(self, operation_name, success, result):
        """Handle operation completion."""
        logger.info(f"GUI received operation_completed signal: {operation_name} - Success: {success}, result_type: {type(result)}")
        self.progress_bar.setVisible(False)
        logger.info(f"Progress bar hidden for completed operation: {operation_name}")
        
        # Handle validate_environment specific results
        if operation_name == "validate_environment" and hasattr(result, 'message'):
            self.test_results_text.setText(f"Environment Validation Results:\n{result.message}\n\nDetails: {result.data}")
            self.export_results_button.setEnabled(True)
    
    def on_operation_failed(self, operation_name, error_message):
        """Handle operation failure."""
        logger.info(f"GUI received operation_failed signal: {operation_name} - {error_message}")
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


class TestSuiteDialog(QDialog):
    """Dialog for creating/editing test suites."""
    
    def __init__(self, parent=None, edit_mode=False, suite_data=None):
        super().__init__(parent)
        self.edit_mode = edit_mode
        self.suite_data = suite_data or {}
        
        self.setWindowTitle("Edit Test Suite" if edit_mode else "Create Test Suite")
        self.setModal(True)
        self.resize(500, 400)
        
        layout = QVBoxLayout(self)
        
        # Form layout
        form_layout = QFormLayout()
        
        self.name_edit = QLineEdit()
        form_layout.addRow("Suite Name:", self.name_edit)
        
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(100)
        form_layout.addRow("Description:", self.description_edit)
        
        layout.addLayout(form_layout)
        
        # Test Selection
        tests_group = QGroupBox("Tests")
        tests_layout = QVBoxLayout(tests_group)
        
        self.tests_list = QListWidget()
        self.tests_list.setSelectionMode(QListWidget.MultiSelection)
        
        # Available tests (in real implementation, this would come from a service)
        available_tests = [
            "Provider: llamacpp", "Provider: sqlite", "Provider: chromadb",
            "Flow: classification", "Flow: memory", "Flow: planning",
            "Configuration: default-llm", "Configuration: vector-db", "Configuration: cache"
        ]
        
        for test in available_tests:
            item = QListWidgetItem(test)
            self.tests_list.addItem(item)
        
        tests_layout.addWidget(self.tests_list)
        
        layout.addWidget(tests_group)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Populate fields if editing
        if edit_mode and suite_data:
            self.populate_fields()
    
    def populate_fields(self):
        """Populate dialog fields with existing suite data."""
        if not self.suite_data:
            return
        
        # Set name and description
        self.name_edit.setText(self.suite_data["name"] if "name" in self.suite_data else "")
        self.description_edit.setPlainText(self.suite_data["description"] if "description" in self.suite_data else "")
        
        # Select the tests that are part of this suite
        suite_tests = self.suite_data["tests"] if "tests" in self.suite_data else []
        for i in range(self.tests_list.count()):
            item = self.tests_list.item(i)
            if item.text() in suite_tests:
                item.setSelected(True)
    
    def get_suite_data(self):
        """Get suite data from dialog."""
        selected_tests = []
        for i in range(self.tests_list.count()):
            item = self.tests_list.item(i)
            if item.isSelected():
                selected_tests.append(item.text())
        
        return {
            "name": self.name_edit.text(),
            "description": self.description_edit.toPlainText(),
            "tests": selected_tests
        }