"""
Advanced Search Widget for Provider Repository

Provides comprehensive search and filtering capabilities for configurations,
roles, and provider health status with real-time filtering and advanced options.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox,
    QCheckBox, QGroupBox, QLabel, QPushButton, QFrame,
    QButtonGroup, QScrollArea, QGridLayout, QDateEdit,
    QSpinBox, QSlider, QTabWidget, QTreeWidget, QTreeWidgetItem
)
from PySide6.QtCore import Qt, Signal, QTimer, QDate
from PySide6.QtGui import QIcon, QFont

logger = logging.getLogger(__name__)


class SearchScope(Enum):
    """Search scope options."""
    ALL = "all"
    CONFIGURATIONS = "configurations"
    ROLES = "roles"
    PROVIDERS = "providers"
    ENVIRONMENTS = "environments"


class FilterCriteria(Enum):
    """Filter criteria options."""
    PROVIDER_TYPE = "provider_type"
    STATUS = "status"
    CREATED_DATE = "created_date"
    LAST_MODIFIED = "last_modified"
    HAS_ROLES = "has_roles"
    HEALTH_STATUS = "health_status"
    ENVIRONMENT = "environment"


class AdvancedSearchWidget(QWidget):
    """
    Advanced search widget with comprehensive filtering capabilities.
    
    Features:
    - Real-time search with debouncing
    - Multiple filter criteria
    - Saved search profiles
    - Export/import filters
    - Search history
    """
    
    # Signals
    search_updated = Signal(dict)  # Filter criteria dict
    filter_applied = Signal(str, dict)  # Filter name, criteria
    search_cleared = Signal()
    search_saved = Signal(str, dict)  # Profile name, criteria
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Search state
        self.current_criteria = {}
        self.saved_profiles = {}
        self.search_history = []
        
        # Debounce timer for real-time search
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self._execute_search)
        
        self.init_ui()
        self.connect_signals()
        
        logger.info("Advanced search widget initialized")
    
    def init_ui(self):
        """Initialize the user interface."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(8)
        
        # Create tabbed interface for different search modes
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        
        # Quick Search Tab
        self.quick_tab = self.create_quick_search_tab()
        self.tab_widget.addTab(self.quick_tab, "Quick Search")
        
        # Advanced Filters Tab
        self.advanced_tab = self.create_advanced_filters_tab()
        self.tab_widget.addTab(self.advanced_tab, "Advanced Filters")
        
        # Saved Profiles Tab
        self.profiles_tab = self.create_profiles_tab()
        self.tab_widget.addTab(self.profiles_tab, "Saved Profiles")
        
        # Action buttons
        self.create_action_buttons()
    
    def create_quick_search_tab(self):
        """Create quick search interface."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Search text input
        search_group = QGroupBox("Quick Search")
        search_layout = QVBoxLayout(search_group)
        
        # Main search box
        search_input_layout = QHBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search configurations, roles, providers...")
        search_input_layout.addWidget(self.search_input)
        
        # Search scope selector
        self.scope_combo = QComboBox()
        self.scope_combo.addItems([
            "All Items", "Configurations", "Roles", "Providers", "Environments"
        ])
        self.scope_combo.setMaximumWidth(150)
        search_input_layout.addWidget(self.scope_combo)
        
        search_layout.addLayout(search_input_layout)
        
        # Quick filter buttons
        quick_filters_layout = QHBoxLayout()
        
        self.active_only_cb = QCheckBox("Active Only")
        self.with_issues_cb = QCheckBox("With Issues")
        self.recently_modified_cb = QCheckBox("Recent (7 days)")
        self.unassigned_roles_cb = QCheckBox("Unassigned Roles")
        
        quick_filters_layout.addWidget(self.active_only_cb)
        quick_filters_layout.addWidget(self.with_issues_cb)
        quick_filters_layout.addWidget(self.recently_modified_cb)
        quick_filters_layout.addWidget(self.unassigned_roles_cb)
        quick_filters_layout.addStretch()
        
        search_layout.addLayout(quick_filters_layout)
        layout.addWidget(search_group)
        
        # Search results summary
        self.results_label = QLabel("Ready to search...")
        self.results_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.results_label)
        
        layout.addStretch()
        return widget
    
    def create_advanced_filters_tab(self):
        """Create advanced filtering interface."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Scrollable area for filters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        filter_widget = QWidget()
        filter_layout = QVBoxLayout(filter_widget)
        
        # Provider Type Filters
        provider_group = QGroupBox("Provider Types")
        provider_layout = QGridLayout(provider_group)
        
        self.provider_checkboxes = {}
        provider_types = [
            "LLM", "Vector DB", "Graph DB", "Embedding", 
            "Cache", "Storage", "Message Queue", "Database"
        ]
        
        for i, ptype in enumerate(provider_types):
            cb = QCheckBox(ptype)
            self.provider_checkboxes[ptype.lower()] = cb
            provider_layout.addWidget(cb, i // 4, i % 4)
        
        filter_layout.addWidget(provider_group)
        
        # Status Filters
        status_group = QGroupBox("Status Filters")
        status_layout = QGridLayout(status_group)
        
        self.status_checkboxes = {}
        statuses = [
            "Active", "Inactive", "Error", "Warning",
            "Testing", "Production", "Development", "Deprecated"
        ]
        
        for i, status in enumerate(statuses):
            cb = QCheckBox(status)
            self.status_checkboxes[status.lower()] = cb
            status_layout.addWidget(cb, i // 4, i % 4)
        
        filter_layout.addWidget(status_group)
        
        # Date Range Filters
        date_group = QGroupBox("Date Filters")
        date_layout = QGridLayout(date_group)
        
        date_layout.addWidget(QLabel("Created After:"), 0, 0)
        self.created_after_date = QDateEdit()
        self.created_after_date.setDate(QDate.currentDate().addDays(-30))
        self.created_after_date.setCalendarPopup(True)
        date_layout.addWidget(self.created_after_date, 0, 1)
        
        date_layout.addWidget(QLabel("Modified After:"), 1, 0)
        self.modified_after_date = QDateEdit()
        self.modified_after_date.setDate(QDate.currentDate().addDays(-7))
        self.modified_after_date.setCalendarPopup(True)
        date_layout.addWidget(self.modified_after_date, 1, 1)
        
        filter_layout.addWidget(date_group)
        
        # Role Assignment Filters
        role_group = QGroupBox("Role Assignment")
        role_layout = QVBoxLayout(role_group)
        
        role_options_layout = QHBoxLayout()
        self.has_roles_cb = QCheckBox("Has assigned roles")
        self.missing_roles_cb = QCheckBox("Missing required roles")
        self.conflicting_roles_cb = QCheckBox("Role conflicts")
        
        role_options_layout.addWidget(self.has_roles_cb)
        role_options_layout.addWidget(self.missing_roles_cb)
        role_options_layout.addWidget(self.conflicting_roles_cb)
        role_layout.addLayout(role_options_layout)
        
        # Specific role search
        specific_role_layout = QHBoxLayout()
        specific_role_layout.addWidget(QLabel("Specific Role:"))
        self.specific_role_combo = QComboBox()
        self.specific_role_combo.setEditable(True)
        self.specific_role_combo.addItems([
            "knowledge-extraction", "default-llm", "default-vector-db",
            "default-graph-db", "default-embedding", "conversation-llm",
            "backup-llm", "primary-vector-db", "memory-manager"
        ])
        specific_role_layout.addWidget(self.specific_role_combo)
        role_layout.addLayout(specific_role_layout)
        
        filter_layout.addWidget(role_group)
        
        scroll.setWidget(filter_widget)
        layout.addWidget(scroll)
        
        return widget
    
    def create_profiles_tab(self):
        """Create saved profiles interface."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Profile management
        profile_group = QGroupBox("Saved Search Profiles")
        profile_layout = QVBoxLayout(profile_group)
        
        # Profile list
        self.profile_tree = QTreeWidget()
        self.profile_tree.setHeaderLabels(["Profile Name", "Criteria Count", "Last Used"])
        self.profile_tree.setAlternatingRowColors(True)
        profile_layout.addWidget(self.profile_tree)
        
        # Profile actions
        profile_actions = QHBoxLayout()
        
        self.load_profile_btn = QPushButton("Load Profile")
        self.save_profile_btn = QPushButton("Save Current Search")
        self.delete_profile_btn = QPushButton("Delete Profile")
        self.export_profile_btn = QPushButton("Export Profile")
        self.import_profile_btn = QPushButton("Import Profile")
        
        profile_actions.addWidget(self.load_profile_btn)
        profile_actions.addWidget(self.save_profile_btn)
        profile_actions.addWidget(self.delete_profile_btn)
        profile_actions.addStretch()
        profile_actions.addWidget(self.export_profile_btn)
        profile_actions.addWidget(self.import_profile_btn)
        
        profile_layout.addLayout(profile_actions)
        layout.addWidget(profile_group)
        
        # Search history
        history_group = QGroupBox("Recent Searches")
        history_layout = QVBoxLayout(history_group)
        
        self.history_tree = QTreeWidget()
        self.history_tree.setHeaderLabels(["Search Query", "Scope", "Results", "Time"])
        self.history_tree.setMaximumHeight(150)
        history_layout.addWidget(self.history_tree)
        
        layout.addWidget(history_group)
        
        return widget
    
    def create_action_buttons(self):
        """Create main action buttons."""
        actions_layout = QHBoxLayout()
        
        # Main actions
        self.search_btn = QPushButton("Search")
        self.search_btn.setDefault(True)
        
        self.clear_btn = QPushButton("Clear All")
        self.reset_btn = QPushButton("Reset to Defaults")
        
        # Additional actions
        self.export_results_btn = QPushButton("Export Results")
        self.create_report_btn = QPushButton("Generate Report")
        
        actions_layout.addWidget(self.search_btn)
        actions_layout.addWidget(self.clear_btn)
        actions_layout.addWidget(self.reset_btn)
        actions_layout.addStretch()
        actions_layout.addWidget(self.export_results_btn)
        actions_layout.addWidget(self.create_report_btn)
        
        self.layout.addLayout(actions_layout)
    
    def connect_signals(self):
        """Connect widget signals."""
        # Real-time search
        self.search_input.textChanged.connect(self._debounce_search)
        self.scope_combo.currentTextChanged.connect(self._debounce_search)
        
        # Quick filters
        self.active_only_cb.toggled.connect(self._debounce_search)
        self.with_issues_cb.toggled.connect(self._debounce_search)
        self.recently_modified_cb.toggled.connect(self._debounce_search)
        self.unassigned_roles_cb.toggled.connect(self._debounce_search)
        
        # Advanced filters
        for cb in self.provider_checkboxes.values():
            cb.toggled.connect(self._debounce_search)
        
        for cb in self.status_checkboxes.values():
            cb.toggled.connect(self._debounce_search)
        
        self.created_after_date.dateChanged.connect(self._debounce_search)
        self.modified_after_date.dateChanged.connect(self._debounce_search)
        
        self.has_roles_cb.toggled.connect(self._debounce_search)
        self.missing_roles_cb.toggled.connect(self._debounce_search)
        self.conflicting_roles_cb.toggled.connect(self._debounce_search)
        self.specific_role_combo.currentTextChanged.connect(self._debounce_search)
        
        # Action buttons
        self.search_btn.clicked.connect(self._execute_search)
        self.clear_btn.clicked.connect(self.clear_all_filters)
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        
        # Profile management
        self.load_profile_btn.clicked.connect(self.load_selected_profile)
        self.save_profile_btn.clicked.connect(self.save_current_profile)
        self.delete_profile_btn.clicked.connect(self.delete_selected_profile)
        self.export_profile_btn.clicked.connect(self.export_profile)
        self.import_profile_btn.clicked.connect(self.import_profile)
    
    def _debounce_search(self):
        """Debounce search input to avoid excessive API calls."""
        self.search_timer.stop()
        self.search_timer.start(300)  # 300ms delay
    
    def _execute_search(self):
        """Execute search with current criteria."""
        try:
            criteria = self._build_search_criteria()
            self.current_criteria = criteria
            
            # Update results summary
            self._update_results_summary(criteria)
            
            # Add to search history
            self._add_to_history(criteria)
            
            # Emit search signal
            self.search_updated.emit(criteria)
            
            logger.debug(f"Search executed with criteria: {len(criteria)} filters")
            
        except Exception as e:
            logger.error(f"Failed to execute search: {e}")
            self.results_label.setText(f"Search error: {str(e)}")
    
    def _build_search_criteria(self) -> Dict[str, Any]:
        """Build search criteria from UI state."""
        criteria = {}
        
        # Text search
        search_text = self.search_input.text().strip()
        if search_text:
            criteria['text'] = search_text
            
        # Search scope
        scope_map = {
            "All Items": SearchScope.ALL,
            "Configurations": SearchScope.CONFIGURATIONS,
            "Roles": SearchScope.ROLES,
            "Providers": SearchScope.PROVIDERS,
            "Environments": SearchScope.ENVIRONMENTS
        }
        scope_text = self.scope_combo.currentText()
        criteria['scope'] = scope_map[scope_text] if scope_text in scope_map else SearchScope.ALL
        
        # Quick filters
        if self.active_only_cb.isChecked():
            criteria['active_only'] = True
            
        if self.with_issues_cb.isChecked():
            criteria['with_issues'] = True
            
        if self.recently_modified_cb.isChecked():
            criteria['recently_modified'] = True
            
        if self.unassigned_roles_cb.isChecked():
            criteria['unassigned_roles'] = True
        
        # Provider type filters
        selected_providers = [
            ptype for ptype, cb in self.provider_checkboxes.items()
            if cb.isChecked()
        ]
        if selected_providers:
            criteria['provider_types'] = selected_providers
        
        # Status filters
        selected_statuses = [
            status for status, cb in self.status_checkboxes.items()
            if cb.isChecked()
        ]
        if selected_statuses:
            criteria['statuses'] = selected_statuses
        
        # Date filters
        if self.created_after_date.date() != QDate.currentDate().addDays(-30):
            criteria['created_after'] = self.created_after_date.date().toPython()
            
        if self.modified_after_date.date() != QDate.currentDate().addDays(-7):
            criteria['modified_after'] = self.modified_after_date.date().toPython()
        
        # Role filters
        if self.has_roles_cb.isChecked():
            criteria['has_roles'] = True
            
        if self.missing_roles_cb.isChecked():
            criteria['missing_roles'] = True
            
        if self.conflicting_roles_cb.isChecked():
            criteria['conflicting_roles'] = True
            
        specific_role = self.specific_role_combo.currentText().strip()
        if specific_role:
            criteria['specific_role'] = specific_role
        
        return criteria
    
    def _update_results_summary(self, criteria: Dict[str, Any]):
        """Update the results summary label."""
        if not criteria:
            self.results_label.setText("No filters applied - showing all items")
            return
        
        summary_parts = []
        
        if 'text' in criteria:
            summary_parts.append(f"Text: '{criteria['text']}'")
            
        if 'scope' in criteria and criteria['scope'] != SearchScope.ALL:
            summary_parts.append(f"Scope: {criteria['scope'].value}")
            
        filter_count = len([k for k in criteria.keys() if k not in ['text', 'scope']])
        if filter_count > 0:
            summary_parts.append(f"{filter_count} additional filters")
        
        if summary_parts:
            self.results_label.setText("Search: " + " | ".join(summary_parts))
        else:
            self.results_label.setText("Ready to search...")
    
    def _add_to_history(self, criteria: Dict[str, Any]):
        """Add search to history."""
        if not criteria:
            return
        
        # Handle scope safely - fail-fast approach
        if 'scope' not in criteria:
            raise ValueError("Search criteria missing required 'scope' field")
        scope_value = criteria['scope']
        if hasattr(scope_value, 'value'):
            scope_str = scope_value.value
        else:
            scope_str = str(scope_value)
        
        history_entry = {
            'criteria': criteria.copy(),
            'timestamp': datetime.now(),
            'query': criteria['text'] if 'text' in criteria else None,
            'scope': scope_str
        }
        
        # Add to beginning of history
        self.search_history.insert(0, history_entry)
        
        # Limit history size
        if len(self.search_history) > 50:
            self.search_history = self.search_history[:50]
        
        # Update history tree
        self._update_history_tree()
    
    def _update_history_tree(self):
        """Update the search history tree widget."""
        self.history_tree.clear()
        
        for entry in self.search_history[:10]:  # Show last 10
            item = QTreeWidgetItem([
                entry['query'],
                entry['scope'],
                "N/A",  # Results count - would be filled by parent
                entry['timestamp'].strftime("%H:%M:%S")
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, entry['criteria'])
            self.history_tree.addTopLevelItem(item)
    
    def clear_all_filters(self):
        """Clear all search filters."""
        # Clear text input
        self.search_input.clear()
        self.scope_combo.setCurrentIndex(0)
        
        # Clear quick filters
        self.active_only_cb.setChecked(False)
        self.with_issues_cb.setChecked(False)
        self.recently_modified_cb.setChecked(False)
        self.unassigned_roles_cb.setChecked(False)
        
        # Clear advanced filters
        for cb in self.provider_checkboxes.values():
            cb.setChecked(False)
            
        for cb in self.status_checkboxes.values():
            cb.setChecked(False)
        
        # Reset dates
        self.created_after_date.setDate(QDate.currentDate().addDays(-30))
        self.modified_after_date.setDate(QDate.currentDate().addDays(-7))
        
        # Clear role filters
        self.has_roles_cb.setChecked(False)
        self.missing_roles_cb.setChecked(False)
        self.conflicting_roles_cb.setChecked(False)
        self.specific_role_combo.setCurrentIndex(0)
        
        # Clear criteria and emit signal
        self.current_criteria = {}
        self.search_cleared.emit()
        
        self.results_label.setText("All filters cleared")
        logger.info("All search filters cleared")
    
    def reset_to_defaults(self):
        """Reset filters to default values."""
        self.clear_all_filters()
        
        # Set some reasonable defaults
        self.active_only_cb.setChecked(True)
        self.scope_combo.setCurrentText("All Items")
        
        self._execute_search()
        logger.info("Search filters reset to defaults")
    
    def save_current_profile(self):
        """Save current search as a profile."""
        from PySide6.QtWidgets import QInputDialog
        
        name, ok = QInputDialog.getText(
            self, "Save Search Profile", 
            "Enter profile name:"
        )
        
        if ok and name.strip():
            criteria = self._build_search_criteria()
            self.saved_profiles[name.strip()] = {
                'criteria': criteria,
                'created': datetime.now(),
                'last_used': datetime.now()
            }
            
            self._update_profiles_tree()
            self.search_saved.emit(name.strip(), criteria)
            
            logger.info(f"Search profile saved: {name.strip()}")
    
    def load_selected_profile(self):
        """Load the selected search profile."""
        current_item = self.profile_tree.currentItem()
        if not current_item:
            return
            
        profile_name = current_item.text(0)
        if profile_name in self.saved_profiles:
            profile = self.saved_profiles[profile_name]
            self._apply_criteria(profile['criteria'])
            
            # Update last used
            profile['last_used'] = datetime.now()
            self._update_profiles_tree()
            
            logger.info(f"Search profile loaded: {profile_name}")
    
    def delete_selected_profile(self):
        """Delete the selected search profile."""
        from PySide6.QtWidgets import QMessageBox
        
        current_item = self.profile_tree.currentItem()
        if not current_item:
            return
            
        profile_name = current_item.text(0)
        
        reply = QMessageBox.question(
            self, "Delete Profile",
            f"Are you sure you want to delete profile '{profile_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if profile_name in self.saved_profiles:
                del self.saved_profiles[profile_name]
                self._update_profiles_tree()
                logger.info(f"Search profile deleted: {profile_name}")
    
    def _apply_criteria(self, criteria: Dict[str, Any]):
        """Apply search criteria to UI widgets."""
        # Clear first
        self.clear_all_filters()
        
        # Apply criteria
        if 'text' in criteria:
            self.search_input.setText(criteria['text'])
            
        if 'scope' in criteria:
            scope_text_map = {
                SearchScope.ALL: "All Items",
                SearchScope.CONFIGURATIONS: "Configurations", 
                SearchScope.ROLES: "Roles",
                SearchScope.PROVIDERS: "Providers",
                SearchScope.ENVIRONMENTS: "Environments"
            }
            # Fail-fast approach - scope must be known
            if criteria['scope'] not in scope_text_map:
                raise ValueError(f"Unknown search scope '{criteria['scope']}' - must be one of: {list(scope_text_map.keys())}")
            scope_text = scope_text_map[criteria['scope']]
            index = self.scope_combo.findText(scope_text)
            if index >= 0:
                self.scope_combo.setCurrentIndex(index)
        
        # Apply quick filters - fail-fast approach
        if 'active_only' in criteria and criteria['active_only']:
            self.active_only_cb.setChecked(True)
        if 'with_issues' in criteria and criteria['with_issues']:
            self.with_issues_cb.setChecked(True)
        if 'recently_modified' in criteria and criteria['recently_modified']:
            self.recently_modified_cb.setChecked(True)
        if 'unassigned_roles' in criteria and criteria['unassigned_roles']:
            self.unassigned_roles_cb.setChecked(True)
            
        # Apply provider type filters - explicit check
        provider_types = criteria['provider_types'] if 'provider_types' in criteria else []
        for ptype, cb in self.provider_checkboxes.items():
            cb.setChecked(ptype in provider_types)
            
        # Apply status filters - explicit check
        statuses = criteria['statuses'] if 'statuses' in criteria else []
        for status, cb in self.status_checkboxes.items():
            cb.setChecked(status in statuses)
            
        # Apply date filters
        if 'created_after' in criteria:
            date_obj = criteria['created_after']
            try:
                if hasattr(date_obj, 'year'):  # datetime object
                    qdate = QDate(date_obj.year, date_obj.month, date_obj.day)
                    self.created_after_date.setDate(qdate)
                elif isinstance(date_obj, str):
                    # Handle string dates
                    qdate = QDate.fromString(date_obj, "yyyy-MM-dd")
                    if qdate.isValid():
                        self.created_after_date.setDate(qdate)
            except (AttributeError, ValueError) as e:
                logger.warning(f"Failed to apply created_after date filter: {e}")
                
        if 'modified_after' in criteria:
            date_obj = criteria['modified_after']
            try:
                if hasattr(date_obj, 'year'):  # datetime object
                    qdate = QDate(date_obj.year, date_obj.month, date_obj.day)
                    self.modified_after_date.setDate(qdate)
                elif isinstance(date_obj, str):
                    # Handle string dates
                    qdate = QDate.fromString(date_obj, "yyyy-MM-dd")
                    if qdate.isValid():
                        self.modified_after_date.setDate(qdate)
            except (AttributeError, ValueError) as e:
                logger.warning(f"Failed to apply modified_after date filter: {e}")
                
        # Apply role filters - fail-fast approach
        if 'has_roles' in criteria and criteria['has_roles']:
            self.has_roles_cb.setChecked(True)
        if 'missing_roles' in criteria and criteria['missing_roles']:
            self.missing_roles_cb.setChecked(True)
        if 'conflicting_roles' in criteria and criteria['conflicting_roles']:
            self.conflicting_roles_cb.setChecked(True)
        if 'specific_role' in criteria:
            role = criteria['specific_role']
            index = self.specific_role_combo.findText(role)
            if index >= 0:
                self.specific_role_combo.setCurrentIndex(index)
            else:
                self.specific_role_combo.setEditText(role)
        
        self._execute_search()
    
    def _update_profiles_tree(self):
        """Update the saved profiles tree widget."""
        self.profile_tree.clear()
        
        for name, profile in self.saved_profiles.items():
            criteria_count = len(profile['criteria'])
            last_used = profile['last_used'].strftime("%Y-%m-%d %H:%M")
            
            item = QTreeWidgetItem([name, str(criteria_count), last_used])
            item.setData(0, Qt.ItemDataRole.UserRole, profile)
            self.profile_tree.addTopLevelItem(item)
    
    def export_profile(self):
        """Export selected profile to file."""
        # Implementation for profile export
        pass
    
    def import_profile(self):
        """Import profile from file.""" 
        # Implementation for profile import
        pass
    
    def get_current_criteria(self) -> Dict[str, Any]:
        """Get current search criteria."""
        return self.current_criteria.copy()
    
    def set_results_count(self, count: int):
        """Update results count in UI."""
        current_text = self.results_label.text()
        if "results" not in current_text:
            self.results_label.setText(f"{current_text} ({count} results)")
        else:
            # Replace existing count
            import re
            new_text = re.sub(r'\(\d+ results\)', f'({count} results)', current_text)
            self.results_label.setText(new_text)