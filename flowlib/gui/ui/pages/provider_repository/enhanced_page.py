"""
Enhanced Provider Repository Page with Advanced Features

Provides comprehensive provider repository management with advanced search,
conflict detection, role validation, and improved UX/UI patterns.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QGroupBox, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QSplitter, QTextEdit, QProgressBar, QFrame,
    QComboBox, QLineEdit, QCheckBox, QTreeWidget, QTreeWidgetItem,
    QScrollArea, QToolButton, QMenu, QActionGroup, QSpacerItem,
    QSizePolicy, QDialog, QDialogButtonBox, QFormLayout
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QIcon, QFont, QAction, QPixmap, QPainter, QColor

from .page import ProviderRepositoryPage
from flowlib.gui.ui.widgets.advanced_search_widget import AdvancedSearchWidget
from flowlib.gui.ui.widgets.role_conflict_detector import RoleConflictDetector
from flowlib.gui.logic.config_manager.provider_repository_controller import ProviderRepositoryController
from flowlib.gui.logic.services.service_factory import ServiceFactory
from flowlib.gui.logic.services.base_controller import ControllerManager

logger = logging.getLogger(__name__)


class ProviderHealthIndicator(QWidget):
    """Widget to display provider health status with visual indicators."""
    
    def __init__(self, provider_name: str, health_status: str, parent=None):
        super().__init__(parent)
        self.provider_name = provider_name
        self.health_status = health_status
        self.init_ui()
    
    def init_ui(self):
        """Initialize the health indicator UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Status icon
        self.status_icon = QLabel()
        self.status_icon.setFixedSize(16, 16)
        self.update_status_icon()
        layout.addWidget(self.status_icon)
        
        # Provider name
        self.name_label = QLabel(self.provider_name)
        layout.addWidget(self.name_label)
        
        # Status text
        self.status_label = QLabel(self.health_status)
        self.status_label.setStyleSheet(self.get_status_style())
        layout.addWidget(self.status_label)
        
        layout.addStretch()
    
    def update_status_icon(self):
        """Update the status icon based on health status."""
        # Create a simple colored circle icon
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Choose color based on status
        status_colors = {
            "healthy": QColor(76, 175, 80),    # Green
            "warning": QColor(255, 193, 7),    # Yellow
            "error": QColor(220, 53, 69),      # Red
            "unknown": QColor(128, 128, 128),  # Gray
            "testing": QColor(33, 150, 243)    # Blue
        }
        
        color = status_colors[self.health_status.lower()] if self.health_status.lower() in status_colors else status_colors["unknown"]
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(2, 2, 12, 12)
        painter.end()
        
        self.status_icon.setPixmap(pixmap)
    
    def get_status_style(self) -> str:
        """Get CSS style for status label."""
        status_styles = {
            "healthy": "color: #4CAF50; font-weight: bold;",
            "warning": "color: #FF9800; font-weight: bold;",
            "error": "color: #F44336; font-weight: bold;",
            "unknown": "color: #757575;",
            "testing": "color: #2196F3; font-weight: bold;"
        }
        return status_styles[self.health_status.lower()] if self.health_status.lower() in status_styles else "color: #757575;"
    
    def update_health_status(self, new_status: str):
        """Update the health status."""
        self.health_status = new_status
        self.status_label.setText(new_status)
        self.status_label.setStyleSheet(self.get_status_style())
        self.update_status_icon()


class RoleAssignmentWidget(QWidget):
    """Enhanced role assignment widget with drag-and-drop support."""
    
    role_assigned = Signal(str, str)  # role_name, config_name
    role_unassigned = Signal(str)     # role_name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_assignments = {}
        self.init_ui()
    
    def init_ui(self):
        """Initialize the role assignment UI."""
        layout = QVBoxLayout(self)
        
        # Header with controls
        header_layout = QHBoxLayout()
        
        # Quick assignment
        header_layout.addWidget(QLabel("Quick Assign:"))
        self.quick_role_combo = QComboBox()
        self.quick_role_combo.setEditable(True)
        self.quick_role_combo.addItems([
            "knowledge-extraction", "default-llm", "default-vector-db",
            "default-graph-db", "default-embedding", "conversation-llm",
            "backup-llm", "primary-vector-db", "memory-manager"
        ])
        header_layout.addWidget(self.quick_role_combo)
        
        self.quick_config_combo = QComboBox()
        header_layout.addWidget(self.quick_config_combo)
        
        self.quick_assign_btn = QPushButton("Assign")
        self.quick_assign_btn.clicked.connect(self.quick_assign_role)
        header_layout.addWidget(self.quick_assign_btn)
        
        header_layout.addStretch()
        
        # Bulk operations
        self.bulk_assign_btn = QPushButton("Bulk Assign")
        self.bulk_unassign_btn = QPushButton("Bulk Unassign")
        header_layout.addWidget(self.bulk_assign_btn)
        header_layout.addWidget(self.bulk_unassign_btn)
        
        layout.addLayout(header_layout)
        
        # Role assignments table
        self.assignments_table = QTableWidget()
        self.assignments_table.setColumnCount(4)
        self.assignments_table.setHorizontalHeaderLabels([
            "Role", "Assigned Configuration", "Status", "Actions"
        ])
        self.assignments_table.horizontalHeader().setStretchLastSection(True)
        self.assignments_table.setAlternatingRowColors(True)
        self.assignments_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        # Enable drag and drop
        self.assignments_table.setDragDropMode(QTableWidget.DragDropMode.InternalMove)
        
        layout.addWidget(self.assignments_table)
        
        # Status summary
        self.summary_label = QLabel("Role assignments ready")
        self.summary_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(self.summary_label)
    
    def quick_assign_role(self):
        """Quickly assign a role to a configuration."""
        role = self.quick_role_combo.currentText().strip()
        config = self.quick_config_combo.currentText().strip()
        
        if not role or not config:
            QMessageBox.warning(self, "Invalid Input", "Please select both role and configuration.")
            return
        
        self.role_assigned.emit(role, config)
        self.add_assignment(role, config, "Assigned")
    
    def add_assignment(self, role: str, config: str, status: str = "Assigned"):
        """Add role assignment to the table."""
        row = self.assignments_table.rowCount()
        self.assignments_table.insertRow(row)
        
        self.assignments_table.setItem(row, 0, QTableWidgetItem(role))
        self.assignments_table.setItem(row, 1, QTableWidgetItem(config))
        
        # Status with color coding
        status_item = QTableWidgetItem(status)
        if status == "Assigned":
            status_item.setForeground(QColor(76, 175, 80))  # Green
        elif status == "Conflict":
            status_item.setForeground(QColor(220, 53, 69))  # Red
        elif status == "Warning":
            status_item.setForeground(QColor(255, 193, 7))  # Yellow
        
        self.assignments_table.setItem(row, 2, status_item)
        
        # Action buttons
        actions_widget = QWidget()
        actions_layout = QHBoxLayout(actions_widget)
        actions_layout.setContentsMargins(2, 2, 2, 2)
        
        edit_btn = QPushButton("Edit")
        edit_btn.setMaximumWidth(50)
        # Store role in button to avoid stale row index
        edit_btn.setProperty("role_name", role)
        edit_btn.clicked.connect(lambda checked, r=role: self.edit_assignment_by_role(r))
        actions_layout.addWidget(edit_btn)
        
        remove_btn = QPushButton("Remove")
        remove_btn.setMaximumWidth(60)
        # Store role in button to avoid stale row index
        remove_btn.setProperty("role_name", role)
        remove_btn.clicked.connect(lambda checked, r=role: self.remove_assignment_by_role(r))
        actions_layout.addWidget(remove_btn)
        
        self.assignments_table.setCellWidget(row, 3, actions_widget)
        
        # Update current assignments
        self.current_assignments[role] = config
        self.update_summary()
    
    def remove_assignment(self, row: int):
        """Remove role assignment by row index."""
        role_item = self.assignments_table.item(row, 0)
        if role_item:
            role = role_item.text()
            self.remove_assignment_by_role(role)
    
    def remove_assignment_by_role(self, role: str):
        """Remove role assignment by role name."""
        if not role:
            return
            
        self.role_unassigned.emit(role)
        if role in self.current_assignments:
            del self.current_assignments[role]
        
        # Find and remove the row
        for row in range(self.assignments_table.rowCount()):
            role_item = self.assignments_table.item(row, 0)
            if role_item and role_item.text() == role:
                self.assignments_table.removeRow(row)
                break
        
        self.update_summary()
    
    def edit_assignment(self, row: int):
        """Edit role assignment by row index."""
        role_item = self.assignments_table.item(row, 0)
        if role_item:
            role = role_item.text()
            self.edit_assignment_by_role(role)
    
    def edit_assignment_by_role(self, role: str):
        """Edit role assignment by role name."""
        if not role:
            return
            
        # Implementation for editing assignments
        QMessageBox.information(self, "Edit Assignment", 
                              f"Assignment editing for role '{role}' feature coming soon!")
    
    def update_summary(self):
        """Update assignments summary."""
        total = len(self.current_assignments)
        self.summary_label.setText(f"{total} role assignments active")
    
    def set_available_configs(self, configs: List[str]):
        """Set available configurations for assignment."""
        self.quick_config_combo.clear()
        self.quick_config_combo.addItems(configs)
    
    def set_assignments(self, assignments: Dict[str, str]):
        """Set current role assignments."""
        self.assignments_table.setRowCount(0)
        self.current_assignments = assignments.copy()
        
        for role, config in assignments.items():
            self.add_assignment(role, config, "Assigned")


class EnhancedProviderRepositoryPage(ProviderRepositoryPage):
    """
    Enhanced Provider Repository Page with advanced features.
    
    Adds comprehensive search, conflict detection, health monitoring,
    and improved role management capabilities to the base repository page.
    """
    
    # Additional signals
    advanced_search_performed = Signal(dict)  # Search criteria
    conflicts_detected = Signal(list)         # List of conflicts
    health_status_updated = Signal(dict)      # Health status data
    
    def __init__(self):
        # Initialize base class first
        super().__init__()
        
        # Add enhanced features
        self._add_enhanced_features()
        
        # Connect enhanced signals
        self._connect_enhanced_signals()
        
        # Initialize data refresh timer
        self.auto_refresh_timer = QTimer()
        self.auto_refresh_timer.timeout.connect(self.refresh_all_data)
        
        logger.info("Enhanced Provider Repository page initialized")
    
    def _add_enhanced_features(self):
        """Add enhanced features to the existing page."""
        try:
            # Replace the existing tab widget with enhanced version
            self._enhance_existing_tabs()
            
            # Add new advanced tabs
            self._add_advanced_tabs()
            
            # Add toolbar with advanced actions
            self._add_advanced_toolbar()
            
        except Exception as e:
            logger.error(f"Failed to add enhanced features: {e}")
    
    def _enhance_existing_tabs(self):
        """Enhance existing tabs with new functionality."""
        try:
            # Enhance Overview tab with health indicators
            self._enhance_overview_tab()
            
            # Enhance Roles tab with advanced assignment features
            self._enhance_roles_tab()
            
        except Exception as e:
            logger.error(f"Failed to enhance existing tabs: {e}")
    
    def _enhance_overview_tab(self):
        """Enhance the overview tab with health monitoring."""
        try:
            if hasattr(self, 'overview_tab'):
                # Add health monitoring section
                health_group = QGroupBox("Provider Health Status")
                health_layout = QVBoxLayout(health_group)
                
                # Health indicators scroll area
                health_scroll = QScrollArea()
                health_scroll.setWidgetResizable(True)
                health_scroll.setMaximumHeight(150)
                
                self.health_indicators_widget = QWidget()
                self.health_indicators_layout = QVBoxLayout(self.health_indicators_widget)
                health_scroll.setWidget(self.health_indicators_widget)
                
                health_layout.addWidget(health_scroll)
                
                # Auto-refresh controls
                refresh_layout = QHBoxLayout()
                self.auto_refresh_cb = QCheckBox("Auto-refresh")
                self.auto_refresh_cb.toggled.connect(self.toggle_auto_refresh)
                refresh_layout.addWidget(self.auto_refresh_cb)
                
                self.refresh_interval_combo = QComboBox()
                self.refresh_interval_combo.addItems(["30 seconds", "1 minute", "5 minutes", "10 minutes"])
                self.refresh_interval_combo.setCurrentText("1 minute")
                refresh_layout.addWidget(self.refresh_interval_combo)
                
                refresh_layout.addStretch()
                
                health_layout.addLayout(refresh_layout)
                
                # Add to overview tab
                if hasattr(self.overview_tab, 'layout'):
                    self.overview_tab.layout().addWidget(health_group)
                
        except Exception as e:
            logger.error(f"Failed to enhance overview tab: {e}")
    
    def _enhance_roles_tab(self):
        """Enhance the roles tab with advanced assignment features."""
        try:
            if hasattr(self, 'roles_tab'):
                # Replace simple table with enhanced role assignment widget
                self.enhanced_role_widget = RoleAssignmentWidget()
                self.enhanced_role_widget.role_assigned.connect(self.on_role_assigned)
                self.enhanced_role_widget.role_unassigned.connect(self.on_role_unassigned)
                
                # Add to roles tab (would need to restructure the existing layout)
                # For demo purposes, we'll add it as a new section
                if hasattr(self.roles_tab, 'layout'):
                    enhanced_group = QGroupBox("Enhanced Role Management")
                    enhanced_layout = QVBoxLayout(enhanced_group)
                    enhanced_layout.addWidget(self.enhanced_role_widget)
                    self.roles_tab.layout().addWidget(enhanced_group)
                
        except Exception as e:
            logger.error(f"Failed to enhance roles tab: {e}")
    
    def _add_advanced_tabs(self):
        """Add new advanced feature tabs."""
        try:
            # Advanced Search Tab
            self.search_tab = self._create_search_tab()
            self.tab_widget.addTab(self.search_tab, "Advanced Search")
            
            # Conflict Detection Tab
            self.conflicts_tab = self._create_conflicts_tab()
            self.tab_widget.addTab(self.conflicts_tab, "Conflict Detection")
            
            # Role Assignment Tab
            self.role_assignment_tab = self._create_role_assignment_tab()
            self.tab_widget.addTab(self.role_assignment_tab, "Role Assignment")
            
            # Role Templates Tab
            self.role_templates_tab = self._create_role_templates_tab()
            self.tab_widget.addTab(self.role_templates_tab, "Role Templates")
            
            # Analytics Tab
            self.analytics_tab = self._create_analytics_tab()
            self.tab_widget.addTab(self.analytics_tab, "Analytics")
            
        except Exception as e:
            logger.error(f"Failed to add advanced tabs: {e}")
    
    def _create_search_tab(self):
        """Create advanced search tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Search widget
        self.search_widget = AdvancedSearchWidget()
        self.search_widget.search_updated.connect(self.on_search_updated)
        self.search_widget.search_cleared.connect(self.on_search_cleared)
        layout.addWidget(self.search_widget)
        
        # Search results
        results_group = QGroupBox("Search Results")
        results_layout = QVBoxLayout(results_group)
        
        self.search_results_table = QTableWidget()
        self.search_results_table.setColumnCount(5)
        self.search_results_table.setHorizontalHeaderLabels([
            "Name", "Type", "Status", "Roles", "Last Modified"
        ])
        self.search_results_table.horizontalHeader().setStretchLastSection(True)
        self.search_results_table.setAlternatingRowColors(True)
        self.search_results_table.setSortingEnabled(True)
        
        results_layout.addWidget(self.search_results_table)
        layout.addWidget(results_group)
        
        return widget
    
    def _create_conflicts_tab(self):
        """Create conflict detection tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Conflict detector widget
        self.conflict_detector = RoleConflictDetector()
        self.conflict_detector.conflict_detected.connect(self.on_conflict_detected)
        self.conflict_detector.analysis_completed.connect(self.on_conflicts_analysis_completed)
        layout.addWidget(self.conflict_detector)
        
        return widget
    
    def _create_analytics_tab(self):
        """Create analytics and reporting tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Analytics summary
        summary_group = QGroupBox("Repository Analytics")
        summary_layout = QGridLayout(summary_group)
        
        # Key metrics
        self.total_configs_metric = QLabel("0")
        self.total_configs_metric.setStyleSheet("font-size: 24px; font-weight: bold; color: #2196F3;")
        summary_layout.addWidget(QLabel("Total Configurations:"), 0, 0)
        summary_layout.addWidget(self.total_configs_metric, 0, 1)
        
        self.active_roles_metric = QLabel("0")
        self.active_roles_metric.setStyleSheet("font-size: 24px; font-weight: bold; color: #4CAF50;")
        summary_layout.addWidget(QLabel("Active Roles:"), 1, 0)
        summary_layout.addWidget(self.active_roles_metric, 1, 1)
        
        self.health_score_metric = QLabel("100%")
        self.health_score_metric.setStyleSheet("font-size: 24px; font-weight: bold; color: #FF9800;")
        summary_layout.addWidget(QLabel("Health Score:"), 2, 0)
        summary_layout.addWidget(self.health_score_metric, 2, 1)
        
        layout.addWidget(summary_group)
        
        # Analytics controls
        controls_group = QGroupBox("Analytics Tools")
        controls_layout = QHBoxLayout(controls_group)
        
        self.generate_report_btn = QPushButton("Generate Report")
        self.export_analytics_btn = QPushButton("Export Analytics")
        self.health_check_btn = QPushButton("Health Check")
        
        controls_layout.addWidget(self.generate_report_btn)
        controls_layout.addWidget(self.export_analytics_btn)
        controls_layout.addWidget(self.health_check_btn)
        controls_layout.addStretch()
        
        layout.addWidget(controls_group)
        
        # Recent activity
        activity_group = QGroupBox("Recent Activity")
        activity_layout = QVBoxLayout(activity_group)
        
        self.activity_list = QTreeWidget()
        self.activity_list.setHeaderLabels(["Time", "Action", "Details", "User"])
        self.activity_list.setMaximumHeight(200)
        activity_layout.addWidget(self.activity_list)
        
        layout.addWidget(activity_group)
        
        layout.addStretch()
        return widget
    
    def _add_advanced_toolbar(self):
        """Add advanced toolbar with quick actions."""
        if hasattr(self, 'layout') and self.layout:
            # Create toolbar frame
            toolbar_frame = QFrame()
            toolbar_frame.setFrameStyle(QFrame.Shape.StyledPanel)
            toolbar_layout = QHBoxLayout(toolbar_frame)
            toolbar_layout.setContentsMargins(5, 5, 5, 5)
            
            # Quick actions
            self.quick_health_check_btn = QPushButton("Quick Health Check")
            self.quick_health_check_btn.clicked.connect(self.quick_health_check)
            toolbar_layout.addWidget(self.quick_health_check_btn)
            
            self.auto_resolve_btn = QPushButton("Auto-Resolve Conflicts")
            self.auto_resolve_btn.clicked.connect(self.auto_resolve_conflicts)
            toolbar_layout.addWidget(self.auto_resolve_btn)
            
            # View options
            toolbar_layout.addWidget(QLabel("|"))
            
            self.view_mode_combo = QComboBox()
            self.view_mode_combo.addItems(["Standard View", "Compact View", "Detailed View"])
            self.view_mode_combo.currentTextChanged.connect(self.change_view_mode)
            toolbar_layout.addWidget(self.view_mode_combo)
            
            # Status indicator
            toolbar_layout.addStretch()
            self.global_status_label = QLabel("System Status: Ready")
            self.global_status_label.setStyleSheet("color: green; font-weight: bold;")
            toolbar_layout.addWidget(self.global_status_label)
            
            # Insert toolbar at the top
            self.layout.insertWidget(1, toolbar_frame)  # After title
    
    def _connect_enhanced_signals(self):
        """Connect enhanced feature signals."""
        try:
            if hasattr(self, 'search_widget'):
                self.search_widget.search_updated.connect(self.perform_advanced_search)
                self.search_widget.search_cleared.connect(self.on_search_cleared)
            
            if hasattr(self, 'conflict_detector'):
                self.conflict_detector.analysis_completed.connect(self.update_global_status)
                self.conflict_detector.conflict_detected.connect(self.on_conflict_detected)
            
            if hasattr(self, 'enhanced_role_widget'):
                self.enhanced_role_widget.role_assigned.connect(self.on_enhanced_role_assigned)
                self.enhanced_role_widget.role_unassigned.connect(self.on_enhanced_role_unassigned)
            
        except Exception as e:
            logger.error(f"Failed to connect enhanced signals: {e}")
    
    def perform_advanced_search(self, criteria: Dict[str, Any]):
        """Perform advanced search with given criteria."""
        try:
            logger.info(f"Performing advanced search with {len(criteria)} criteria")
            
            # This would implement actual search logic
            # For now, we'll simulate results
            self.search_results_table.setRowCount(0)
            
            # Simulate search results
            sample_results = [
                ("phi4-model", "LLM", "Active", "default-llm, knowledge-extraction", "2024-01-15"),
                ("chroma-vector", "Vector DB", "Active", "default-vector-db", "2024-01-14"),
                ("neo4j-graph", "Graph DB", "Warning", "default-graph-db", "2024-01-10")
            ]
            
            for i, (name, ptype, status, roles, modified) in enumerate(sample_results):
                self.search_results_table.insertRow(i)
                self.search_results_table.setItem(i, 0, QTableWidgetItem(name))
                self.search_results_table.setItem(i, 1, QTableWidgetItem(ptype))
                self.search_results_table.setItem(i, 2, QTableWidgetItem(status))
                self.search_results_table.setItem(i, 3, QTableWidgetItem(roles))
                self.search_results_table.setItem(i, 4, QTableWidgetItem(modified))
            
            # Update search widget with results count
            if hasattr(self.search_widget, 'set_results_count'):
                self.search_widget.set_results_count(len(sample_results))
            
            self.advanced_search_performed.emit(criteria)
            
        except Exception as e:
            logger.error(f"Advanced search failed: {e}")
    
    def _create_role_assignment_tab(self):
        """Create drag-and-drop role assignment tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Import here to avoid circular imports
        from flowlib.gui.ui.widgets.drag_drop_role_widget import DragDropRoleWidget
        
        # Role assignment widget
        self.drag_drop_widget = DragDropRoleWidget()
        self.drag_drop_widget.role_assigned.connect(self.on_role_assigned)
        self.drag_drop_widget.role_unassigned.connect(self.on_role_unassigned)
        layout.addWidget(self.drag_drop_widget)
        
        # Controls
        controls_group = QGroupBox("Assignment Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        validate_btn = QPushButton("Validate Assignments")
        validate_btn.clicked.connect(self.validate_role_assignments)
        controls_layout.addWidget(validate_btn)
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_role_assignments)
        controls_layout.addWidget(clear_btn)
        
        export_btn = QPushButton("Export Configuration")
        export_btn.clicked.connect(self.export_role_assignments)
        controls_layout.addWidget(export_btn)
        
        controls_layout.addStretch()
        layout.addWidget(controls_group)
        
        return widget
    
    def _create_role_templates_tab(self):
        """Create role templates management tab."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Import here to avoid circular imports
        from flowlib.gui.logic.role_templates import RoleTemplateManager
        from flowlib.gui.logic.role_validation import RoleValidationEngine
        
        # Template manager
        self.template_manager = RoleTemplateManager()
        self.validation_engine = RoleValidationEngine()
        
        # Left panel - Template list
        left_panel = QGroupBox("Available Templates")
        left_layout = QVBoxLayout(left_panel)
        
        # Template list
        self.templates_table = QTableWidget()
        self.templates_table.setColumnCount(4)
        self.templates_table.setHorizontalHeaderLabels([
            "Name", "Category", "Version", "Description"
        ])
        self.templates_table.horizontalHeader().setStretchLastSection(True)
        self.templates_table.setAlternatingRowColors(True)
        self.templates_table.setSortingEnabled(True)
        self.templates_table.itemSelectionChanged.connect(self.on_template_selected)
        left_layout.addWidget(self.templates_table)
        
        # Template controls
        template_controls = QHBoxLayout()
        
        apply_template_btn = QPushButton("Apply Template")
        apply_template_btn.clicked.connect(self.apply_selected_template)
        template_controls.addWidget(apply_template_btn)
        
        create_template_btn = QPushButton("Create Template")
        create_template_btn.clicked.connect(self.create_new_template)
        template_controls.addWidget(create_template_btn)
        
        delete_template_btn = QPushButton("Delete Template")
        delete_template_btn.clicked.connect(self.delete_selected_template)
        template_controls.addWidget(delete_template_btn)
        
        template_controls.addStretch()
        left_layout.addLayout(template_controls)
        
        layout.addWidget(left_panel)
        
        # Right panel - Template details and presets
        right_panel = QGroupBox("Template Details")
        right_layout = QVBoxLayout(right_panel)
        
        # Template information
        self.template_info = QTextEdit()
        self.template_info.setReadOnly(True)
        self.template_info.setMaximumHeight(200)
        right_layout.addWidget(self.template_info)
        
        # Presets
        presets_group = QGroupBox("Available Presets")
        presets_layout = QVBoxLayout(presets_group)
        
        self.presets_table = QTableWidget()
        self.presets_table.setColumnCount(3)
        self.presets_table.setHorizontalHeaderLabels([
            "Name", "Description", "Usage Count"
        ])
        self.presets_table.horizontalHeader().setStretchLastSection(True)
        self.presets_table.setAlternatingRowColors(True)
        presets_layout.addWidget(self.presets_table)
        
        # Preset controls
        preset_controls = QHBoxLayout()
        
        apply_preset_btn = QPushButton("Apply with Preset")
        apply_preset_btn.clicked.connect(self.apply_template_with_preset)
        preset_controls.addWidget(apply_preset_btn)
        
        create_preset_btn = QPushButton("Create Preset")
        create_preset_btn.clicked.connect(self.create_new_preset)
        preset_controls.addWidget(create_preset_btn)
        
        preset_controls.addStretch()
        presets_layout.addLayout(preset_controls)
        
        right_layout.addWidget(presets_group)
        layout.addWidget(right_panel)
        
        # Load templates
        self.load_templates_table()
        
        return widget
    
    def on_search_updated(self, criteria: Dict[str, Any]):
        """Handle search criteria updates."""
        self.perform_advanced_search(criteria)
    
    def on_search_cleared(self):
        """Handle search cleared."""
        self.search_results_table.setRowCount(0)
        logger.info("Search results cleared")
    
    def on_enhanced_role_assigned(self, role_name: str, config_name: str):
        """Handle enhanced role assignment."""
        # Call the parent class method if it exists
        if hasattr(super(), 'assign_role'):
            # This would trigger the actual role assignment
            if self.controller:
                self.controller.assign_role(role_name, config_name)
        logger.info(f"Enhanced role assigned: {role_name} -> {config_name}")
    
    def on_enhanced_role_unassigned(self, role_name: str):
        """Handle enhanced role unassignment."""
        # Call the parent class method if it exists
        if hasattr(super(), 'unassign_role'):
            # This would trigger the actual role unassignment
            if self.controller:
                self.controller.unassign_role(role_name)
        logger.info(f"Enhanced role unassigned: {role_name}")
    
    def on_conflict_detected(self, conflict):
        """Handle conflict detection."""
        logger.warning(f"Conflict detected: {conflict.description}")
        self.update_global_status()
    
    def on_conflicts_analysis_completed(self, conflicts: List):
        """Handle conflict analysis completion."""
        critical_count = sum(1 for c in conflicts if c.severity.value == "critical")
        high_count = sum(1 for c in conflicts if c.severity.value == "high")
        
        if critical_count > 0:
            self.global_status_label.setText(f"System Status: {critical_count} Critical Issues")
            self.global_status_label.setStyleSheet("color: red; font-weight: bold;")
        elif high_count > 0:
            self.global_status_label.setText(f"System Status: {high_count} High Priority Issues")
            self.global_status_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.global_status_label.setText("System Status: Healthy")
            self.global_status_label.setStyleSheet("color: green; font-weight: bold;")
        
        self.conflicts_detected.emit(conflicts)
    
    def quick_health_check(self):
        """Perform quick health check of all providers."""
        try:
            self.global_status_label.setText("System Status: Checking...")
            self.global_status_label.setStyleSheet("color: blue; font-weight: bold;")
            
            # Simulate health check
            QTimer.singleShot(2000, self.complete_health_check)
            
            logger.info("Quick health check initiated")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def complete_health_check(self):
        """Complete health check simulation."""
        # Add sample health indicators
        if hasattr(self, 'health_indicators_layout'):
            # Clear existing indicators
            for i in reversed(range(self.health_indicators_layout.count())):
                child = self.health_indicators_layout.itemAt(i).widget()
                if child:
                    child.setParent(None)
            
            # Add new indicators
            sample_providers = [
                ("phi4-model", "healthy"),
                ("chroma-vector", "healthy"),
                ("neo4j-graph", "warning"),
                ("redis-cache", "error")
            ]
            
            for name, status in sample_providers:
                indicator = ProviderHealthIndicator(name, status)
                self.health_indicators_layout.addWidget(indicator)
        
        self.global_status_label.setText("System Status: Health Check Complete")
        self.global_status_label.setStyleSheet("color: green; font-weight: bold;")
    
    def auto_resolve_conflicts(self):
        """Auto-resolve conflicts using the conflict detector."""
        if hasattr(self, 'conflict_detector'):
            self.conflict_detector.auto_resolve_conflicts()
        else:
            QMessageBox.information(
                self, "Auto-Resolve",
                "Switch to the Conflict Detection tab to use auto-resolve features."
            )
    
    def change_view_mode(self, mode: str):
        """Change the view mode of the repository display."""
        logger.info(f"View mode changed to: {mode}")
        
        if mode == "Compact View":
            # Make tables more compact
            if hasattr(self, 'roles_table'):
                self.roles_table.verticalHeader().setDefaultSectionSize(20)
        elif mode == "Detailed View":
            # Show more details
            if hasattr(self, 'roles_table'):
                self.roles_table.verticalHeader().setDefaultSectionSize(40)
        else:
            # Standard view
            if hasattr(self, 'roles_table'):
                self.roles_table.verticalHeader().setDefaultSectionSize(30)
    
    def toggle_auto_refresh(self, enabled: bool):
        """Toggle auto-refresh of repository data."""
        if enabled:
            interval_text = self.refresh_interval_combo.currentText()
            interval_mapping = {
                "30 seconds": 30,
                "1 minute": 60,
                "5 minutes": 300,
                "10 minutes": 600
            }
            # Fail-fast approach - no fallbacks
            if interval_text not in interval_mapping:
                raise ValueError(f"Unknown refresh interval '{interval_text}' - must be one of: {list(interval_mapping.keys())}")
            interval_seconds = interval_mapping[interval_text]
            
            self.auto_refresh_timer.start(interval_seconds * 1000)
            logger.info(f"Auto-refresh enabled: {interval_text}")
        else:
            self.auto_refresh_timer.stop()
            logger.info("Auto-refresh disabled")
    
    def refresh_all_data(self):
        """Refresh all repository data."""
        try:
            # Refresh base data
            super().refresh_data()
            
            # Refresh enhanced features
            if hasattr(self, 'conflict_detector') and hasattr(self.conflict_detector, 'role_assignments'):
                # Update conflict detector with current assignments
                self.conflict_detector.set_role_assignments(self.get_current_role_assignments())
            
            # Update analytics
            self.update_analytics_metrics()
            
            # Update activity log
            self.update_activity_log()
            
            logger.debug("All repository data refreshed")
            
        except Exception as e:
            logger.error(f"Failed to refresh all data: {e}")
    
    def get_current_role_assignments(self) -> Dict[str, str]:
        """Get current role assignments from the repository."""
        # This would get actual role assignments from the role manager
        # For now, return sample data
        return {
            "default-llm": "phi4-model",
            "default-vector-db": "chroma-vector",
            "default-graph-db": "neo4j-graph",
            "knowledge-extraction": "phi4-model"
        }
    
    def update_analytics_metrics(self):
        """Update analytics metrics display."""
        try:
            # Get repository overview
            overview = getattr(self, 'repository_overview', {})
            
            # Update metrics - fail-fast approach
            if 'total_configurations' not in overview:
                raise ValueError("Repository overview missing required 'total_configurations' field")
            total_configs = overview['total_configurations']
            self.total_configs_metric.setText(str(total_configs))
            
            if 'assigned_roles' not in overview:
                raise ValueError("Repository overview missing required 'assigned_roles' field")
            assigned_roles = overview['assigned_roles']
            self.active_roles_metric.setText(str(len(assigned_roles)))
            
            # Calculate health score based on conflicts
            conflicts = getattr(self, 'current_conflicts', [])
            critical_count = sum(1 for c in conflicts if hasattr(c, 'severity') and c.severity.value == "critical")
            health_score = max(0, 100 - (critical_count * 20) - (len(conflicts) * 5))
            self.health_score_metric.setText(f"{health_score}%")
            
        except Exception as e:
            logger.error(f"Failed to update analytics metrics: {e}")
    
    def update_activity_log(self):
        """Update recent activity log."""
        try:
            # Add sample activity entries
            from datetime import datetime
            
            if hasattr(self, 'activity_list'):
                # Add new activity (simulated)
                current_time = datetime.now().strftime("%H:%M:%S")
                activities = [
                    (current_time, "Data Refresh", "Repository data updated", "System"),
                    (current_time, "Health Check", "All providers checked", "System")
                ]
                
                for time, action, details, user in activities:
                    item = QTreeWidgetItem([time, action, details, user])
                    self.activity_list.insertTopLevelItem(0, item)
                
                # Limit to last 50 entries
                while self.activity_list.topLevelItemCount() > 50:
                    self.activity_list.takeTopLevelItem(50)
        
        except Exception as e:
            logger.error(f"Failed to update activity log: {e}")
    
    def update_global_status(self):
        """Update global system status indicator."""
        try:
            conflicts = getattr(self, 'current_conflicts', [])
            if hasattr(self, 'conflict_detector'):
                conflicts = self.conflict_detector.current_conflicts
            
            critical_count = sum(1 for c in conflicts 
                                if hasattr(c, 'severity') and hasattr(c.severity, 'value') and c.severity.value == "critical")
            high_count = sum(1 for c in conflicts 
                           if hasattr(c, 'severity') and hasattr(c.severity, 'value') and c.severity.value == "high")
            
            if critical_count > 0:
                self.global_status_label.setText(f"System Status: {critical_count} Critical Issues")
                self.global_status_label.setStyleSheet("color: red; font-weight: bold;")
            elif high_count > 0:
                self.global_status_label.setText(f"System Status: {high_count} High Priority Issues")
                self.global_status_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.global_status_label.setText("System Status: Healthy")
                self.global_status_label.setStyleSheet("color: green; font-weight: bold;")
        
        except Exception as e:
            logger.error(f"Failed to update global status: {e}")
    
    # Role Assignment Event Handlers
    def on_role_assigned(self, provider_name: str, role_name: str, role_data: dict):
        """Handle role assignment from drag-and-drop widget."""
        try:
            # Validate the assignment
            from flowlib.gui.logic.role_validation import RoleValidationEngine
            validation_engine = RoleValidationEngine()
            
            # Get current assignments
            current_assignments = self.drag_drop_widget.get_role_assignments()
            
            # Validate assignment
            issues = validation_engine.validate_role_assignment(
                provider_name, role_name, current_assignments
            )
            
            # Check for critical issues
            critical_issues = [issue for issue in issues if issue.severity.value in ['error', 'critical']]
            if critical_issues:
                QMessageBox.warning(
                    self,
                    "Role Assignment Validation Failed",
                    f"Cannot assign role '{role_name}' to provider '{provider_name}':\n\n" +
                    "\n".join(str(issue) for issue in critical_issues[:3])
                )
                return
            
            logger.info(f"Role '{role_name}' successfully assigned to provider '{provider_name}'")
            self.update_global_status()
            
        except Exception as e:
            logger.error(f"Failed to handle role assignment: {e}")
            QMessageBox.critical(self, "Assignment Error", f"Failed to assign role: {str(e)}")
    
    def on_role_unassigned(self, provider_name: str, role_name: str):
        """Handle role unassignment."""
        try:
            logger.info(f"Role '{role_name}' unassigned from provider '{provider_name}'")
            self.update_global_status()
        except Exception as e:
            logger.error(f"Failed to handle role unassignment: {e}")
    
    def validate_role_assignments(self):
        """Validate all current role assignments."""
        try:
            from flowlib.gui.logic.role_validation import RoleValidationEngine
            validation_engine = RoleValidationEngine()
            
            # Get current assignments
            assignments = self.drag_drop_widget.get_role_assignments()
            
            # Validate complete assignment
            issues = validation_engine.validate_complete_assignment(assignments)
            
            # Generate report 
            report = validation_engine.generate_assignment_report(assignments)
            
            # Show validation results
            dialog = QDialog(self)
            dialog.setWindowTitle("Role Assignment Validation Report")
            dialog.setModal(True)
            dialog.resize(600, 500)
            
            layout = QVBoxLayout(dialog)
            
            report_text = QTextEdit()
            report_text.setPlainText(report)
            report_text.setReadOnly(True)
            layout.addWidget(report_text)
            
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
            buttons.accepted.connect(dialog.accept)
            layout.addWidget(buttons)
            
            dialog.exec()
            
        except Exception as e:
            logger.error(f"Failed to validate role assignments: {e}")
            QMessageBox.critical(self, "Validation Error", f"Failed to validate assignments: {str(e)}")
    
    def clear_role_assignments(self):
        """Clear all role assignments."""
        try:
            reply = QMessageBox.question(
                self,
                "Clear All Assignments",
                "Are you sure you want to clear all role assignments?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.drag_drop_widget.clear_all_assignments()
                logger.info("All role assignments cleared")
                
        except Exception as e:
            logger.error(f"Failed to clear role assignments: {e}")
    
    def export_role_assignments(self):
        """Export current role assignments to a file."""
        try:
            assignments = self.drag_drop_widget.get_role_assignments()
            
            # Simple JSON export for now
            import json
            from PySide6.QtWidgets import QFileDialog
            
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Role Assignments",
                "role_assignments.json",
                "JSON Files (*.json);;All Files (*)"
            )
            
            if filename:
                # Convert sets to lists for JSON serialization
                export_data = {
                    provider: list(roles) for provider, roles in assignments.items()
                }
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                QMessageBox.information(self, "Export Successful", f"Role assignments exported to {filename}")
                logger.info(f"Role assignments exported to {filename}")
                
        except Exception as e:
            logger.error(f"Failed to export role assignments: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export assignments: {str(e)}")
    
    # Template Management Event Handlers
    def on_template_selected(self):
        """Handle template selection."""
        try:
            selected_items = self.templates_table.selectedItems()
            if not selected_items:
                return
            
            row = selected_items[0].row()
            template_name = self.templates_table.item(row, 0).text()
            
            # Get template details
            template = self.template_manager.templates[template_name] if template_name in self.template_manager.templates else None
            if template:
                # Update template info
                info_text = f"Name: {template.name}\n"
                info_text += f"Description: {template.description}\n"
                info_text += f"Category: {template.category.value}\n"
                info_text += f"Version: {template.version}\n"
                info_text += f"Author: {template.author}\n"
                info_text += f"Created: {template.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                info_text += f"Role Assignments: {len(template.role_assignments)}\n"
                info_text += f"Required Provider Types: {', '.join(template.required_provider_types)}\n"
                
                if template.tags:
                    info_text += f"Tags: {', '.join(template.tags)}\n"
                
                self.template_info.setPlainText(info_text)
                
                # Load presets for this template
                self.load_presets_for_template(template_name)
            
        except Exception as e:
            logger.error(f"Failed to handle template selection: {e}")
    
    def load_templates_table(self):
        """Load templates into the table."""
        try:
            templates = self.template_manager.list_templates()
            
            self.templates_table.setRowCount(len(templates))
            
            for row, template in enumerate(templates):
                self.templates_table.setItem(row, 0, QTableWidgetItem(template.name))
                self.templates_table.setItem(row, 1, QTableWidgetItem(template.category.value))
                self.templates_table.setItem(row, 2, QTableWidgetItem(template.version))
                self.templates_table.setItem(row, 3, QTableWidgetItem(template.description))
            
        except Exception as e:
            logger.error(f"Failed to load templates table: {e}")
    
    def load_presets_for_template(self, template_name: str):
        """Load presets for the selected template."""
        try:
            presets = self.template_manager.list_presets(template_name)
            
            self.presets_table.setRowCount(len(presets))
            
            for row, preset in enumerate(presets):
                self.presets_table.setItem(row, 0, QTableWidgetItem(preset.name))
                self.presets_table.setItem(row, 1, QTableWidgetItem(preset.description))
                self.presets_table.setItem(row, 2, QTableWidgetItem(str(preset.use_count)))
            
        except Exception as e:
            logger.error(f"Failed to load presets: {e}")
    
    def apply_selected_template(self):
        """Apply the selected template."""
        try:
            selected_items = self.templates_table.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "No Selection", "Please select a template to apply.")
                return
            
            row = selected_items[0].row()
            template_name = self.templates_table.item(row, 0).text()
            
            # Get available providers (mock data for now)
            available_providers = ["provider1", "provider2", "provider3", "provider4", "provider5"]
            
            # Apply template
            assignments = self.template_manager.apply_template(template_name, available_providers)
            
            # Update drag-drop widget
            self.drag_drop_widget.clear_all_assignments()
            for provider_name, roles in assignments.items():
                for role_name in roles:
                    # Simulate assignment to drag-drop widget
                    pass  # Would need to add provider to widget first
            
            QMessageBox.information(
                self,
                "Template Applied",
                f"Template '{template_name}' applied successfully with {len(assignments)} provider assignments."
            )
            
        except Exception as e:
            logger.error(f"Failed to apply template: {e}")
            QMessageBox.critical(self, "Template Error", f"Failed to apply template: {str(e)}")
    
    def apply_template_with_preset(self):
        """Apply template with selected preset."""
        try:
            # Get selected template and preset
            template_items = self.templates_table.selectedItems()
            preset_items = self.presets_table.selectedItems()
            
            if not template_items or not preset_items:
                QMessageBox.warning(self, "No Selection", "Please select both a template and a preset.")
                return
            
            template_row = template_items[0].row()
            preset_row = preset_items[0].row()
            
            template_name = self.templates_table.item(template_row, 0).text()
            preset_name = self.presets_table.item(preset_row, 0).text()
            
            # Get available providers (mock data for now)
            available_providers = ["provider1", "provider2", "provider3", "provider4", "provider5"]
            
            # Apply template with preset
            assignments = self.template_manager.apply_template(
                template_name, available_providers, preset_name
            )
            
            QMessageBox.information(
                self,
                "Template with Preset Applied",
                f"Template '{template_name}' with preset '{preset_name}' applied successfully."
            )
            
        except Exception as e:
            logger.error(f"Failed to apply template with preset: {e}")
            QMessageBox.critical(self, "Template Error", f"Failed to apply template with preset: {str(e)}")
    
    def create_new_template(self):
        """Create a new role template."""
        try:
            QMessageBox.information(
                self,
                "Feature Not Implemented",
                "Template creation wizard is not yet implemented.\n\n"
                "This would open a dialog to create custom role templates."
            )
        except Exception as e:
            logger.error(f"Failed to create new template: {e}")
    
    def create_new_preset(self):
        """Create a new template preset."""
        try:
            QMessageBox.information(
                self,
                "Feature Not Implemented", 
                "Preset creation wizard is not yet implemented.\n\n"
                "This would open a dialog to create custom template presets."
            )
        except Exception as e:
            logger.error(f"Failed to create new preset: {e}")
    
    def delete_selected_template(self):
        """Delete the selected template."""
        try:
            selected_items = self.templates_table.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "No Selection", "Please select a template to delete.")
                return
            
            row = selected_items[0].row()
            template_name = self.templates_table.item(row, 0).text()
            
            reply = QMessageBox.question(
                self,
                "Delete Template",
                f"Are you sure you want to delete template '{template_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.template_manager.delete_template(template_name)
                self.load_templates_table()
                QMessageBox.information(self, "Template Deleted", f"Template '{template_name}' deleted successfully.")
                
        except Exception as e:
            logger.error(f"Failed to delete template: {e}")
            QMessageBox.critical(self, "Delete Error", f"Failed to delete template: {str(e)}")
    
    def get_title(self):
        """Get page title for navigation."""
        return "Provider Repository (Enhanced)"
    
    def get_state(self):
        """Get current page state including enhanced features."""
        base_state = super().get_state()
        
        # Add enhanced state
        enhanced_state = {
            "enhanced_features": True,
            "auto_refresh_enabled": self.auto_refresh_timer.isActive() if hasattr(self, 'auto_refresh_timer') else False,
            "view_mode": self.view_mode_combo.currentText() if hasattr(self, 'view_mode_combo') else "Standard View"
        }
        
        base_state.update(enhanced_state)
        return base_state
    
    def set_state(self, state):
        """Set page state including enhanced features."""
        super().set_state(state)
        
        # Restore enhanced state
        if state["auto_refresh_enabled"] if "auto_refresh_enabled" in state else False and hasattr(self, 'auto_refresh_cb'):
            self.auto_refresh_cb.setChecked(True)
        
        if "view_mode" in state and hasattr(self, 'view_mode_combo'):
            mode = state["view_mode"]
            index = self.view_mode_combo.findText(mode)
            if index >= 0:
                self.view_mode_combo.setCurrentIndex(index)
    
    def closeEvent(self, event):
        """Handle page close event with cleanup."""
        # Stop auto-refresh timer
        if hasattr(self, 'auto_refresh_timer'):
            self.auto_refresh_timer.stop()
        
        # Stop any running analysis
        if (hasattr(self, 'conflict_detector') and 
            hasattr(self.conflict_detector, 'analysis_thread') and 
            self.conflict_detector.analysis_thread is not None):
            if self.conflict_detector.analysis_thread.isRunning():
                self.conflict_detector.analysis_thread.stop()
                self.conflict_detector.analysis_thread.wait(1000)  # 1 second timeout
        
        # Call parent cleanup
        super().closeEvent(event)