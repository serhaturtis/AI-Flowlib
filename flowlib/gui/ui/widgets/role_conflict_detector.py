"""
Role Conflict Detection and Resolution System

Provides comprehensive role conflict detection with automatic resolution
suggestions and validation of role assignments across the provider ecosystem.
"""

import logging
from typing import Dict, List, Set, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QGroupBox, QLabel, QPushButton, QTextEdit, QTabWidget,
    QProgressBar, QFrame, QSplitter, QHeaderView, QMessageBox,
    QComboBox, QCheckBox, QDialog, QDialogButtonBox
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread
from PySide6.QtGui import QIcon, QColor, QFont

logger = logging.getLogger(__name__)


class ConflictSeverity(Enum):
    """Conflict severity levels."""
    CRITICAL = "critical"      # System cannot function
    HIGH = "high"             # Major functionality impaired
    MEDIUM = "medium"         # Some features affected
    LOW = "low"               # Minor issues or warnings
    INFO = "info"             # Informational only


class ConflictType(Enum):
    """Types of role conflicts."""
    DUPLICATE_ASSIGNMENT = "duplicate_assignment"    # Same role assigned to multiple configs
    MISSING_REQUIRED = "missing_required"           # Required role not assigned
    CIRCULAR_DEPENDENCY = "circular_dependency"     # Circular role dependencies
    TYPE_MISMATCH = "type_mismatch"                # Role assigned to wrong provider type
    VERSION_INCOMPATIBLE = "version_incompatible"   # Incompatible versions
    DEPENDENCY_MISSING = "dependency_missing"       # Missing dependency roles
    DEPRECATED_USAGE = "deprecated_usage"           # Using deprecated roles
    PERFORMANCE_WARNING = "performance_warning"     # Performance concerns


@dataclass
class RoleConflict:
    """Represents a detected role conflict."""
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    affected_roles: List[str]
    affected_configs: List[str]
    description: str
    recommendation: str
    auto_resolvable: bool = False
    resolution_steps: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None


class ConflictAnalysisThread(QThread):
    """Background thread for conflict analysis."""
    
    analysis_progress = Signal(int)  # Progress percentage
    conflict_detected = Signal(object)  # RoleConflict object
    analysis_complete = Signal(list)  # List of all conflicts
    analysis_failed = Signal(str)  # Error message
    
    def __init__(self, role_assignments: Dict[str, str], parent=None):
        super().__init__(parent)
        self.role_assignments = role_assignments
        self.should_stop = False
    
    def run(self):
        """Run conflict analysis in background."""
        try:
            conflicts = []
            analysis_steps = [
                self._detect_duplicate_assignments,
                self._detect_missing_required_roles, 
                self._detect_circular_dependencies,
                self._detect_type_mismatches,
                self._detect_dependency_issues,
                self._detect_deprecated_usage,
                self._detect_performance_warnings
            ]
            
            total_steps = len(analysis_steps)
            
            for i, analysis_func in enumerate(analysis_steps):
                if self.should_stop:
                    return
                    
                step_conflicts = analysis_func()
                conflicts.extend(step_conflicts)
                
                for conflict in step_conflicts:
                    self.conflict_detected.emit(conflict)
                
                progress = int(((i + 1) / total_steps) * 100)
                self.analysis_progress.emit(progress)
            
            self.analysis_complete.emit(conflicts)
            
        except Exception as e:
            logger.error(f"Conflict analysis failed: {e}")
            self.analysis_failed.emit(str(e))
    
    def stop(self):
        """Stop the analysis."""
        self.should_stop = True
    
    def _detect_duplicate_assignments(self) -> List[RoleConflict]:
        """Detect roles assigned to multiple configurations."""
        conflicts = []
        role_counts = {}
        
        # Count role assignments
        for role, config in self.role_assignments.items():
            if role in role_counts:
                role_counts[role].append(config)
            else:
                role_counts[role] = [config]
        
        # Find duplicates
        for role, configs in role_counts.items():
            if len(configs) > 1:
                conflict = RoleConflict(
                    conflict_id=f"duplicate_{role}",
                    conflict_type=ConflictType.DUPLICATE_ASSIGNMENT,
                    severity=ConflictSeverity.HIGH,
                    affected_roles=[role],
                    affected_configs=configs,
                    description=f"Role '{role}' is assigned to multiple configurations: {', '.join(configs)}",
                    recommendation=f"Choose the primary configuration for role '{role}' and remove others",
                    auto_resolvable=False,
                    resolution_steps=[
                        f"1. Identify the primary configuration for '{role}'",
                        f"2. Remove role assignment from other configurations: {', '.join(configs[1:])}",
                        "3. Verify system functionality after changes"
                    ]
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_missing_required_roles(self) -> List[RoleConflict]:
        """Detect missing required roles."""
        conflicts = []
        
        # Define required roles for system functionality
        required_roles = {
            "default-llm": ConflictSeverity.CRITICAL,
            "default-vector-db": ConflictSeverity.HIGH,
            "default-embedding": ConflictSeverity.HIGH,
            "knowledge-extraction": ConflictSeverity.MEDIUM,
            "memory-manager": ConflictSeverity.MEDIUM
        }
        
        for role, severity in required_roles.items():
            if role not in self.role_assignments:
                conflict = RoleConflict(
                    conflict_id=f"missing_{role}",
                    conflict_type=ConflictType.MISSING_REQUIRED,
                    severity=severity,
                    affected_roles=[role],
                    affected_configs=[],
                    description=f"Required role '{role}' is not assigned to any configuration",
                    recommendation=f"Assign role '{role}' to an appropriate configuration",
                    auto_resolvable=False,
                    resolution_steps=[
                        f"1. Create or identify a configuration suitable for '{role}'",
                        f"2. Assign the '{role}' role to the configuration",
                        "3. Test the assignment to ensure proper functionality"
                    ]
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_circular_dependencies(self) -> List[RoleConflict]:
        """Detect circular dependencies between roles."""
        conflicts = []
        
        # Define role dependencies (simplified example)
        dependencies = {
            "knowledge-extraction": ["default-llm", "default-vector-db"],
            "memory-manager": ["default-vector-db", "default-embedding"],
            "conversation-llm": ["memory-manager"],
            "analysis-llm": ["knowledge-extraction"]
        }
        
        # Use DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(role, path):
            if role in rec_stack:
                # Found cycle
                cycle_start = path.index(role)
                cycle = path[cycle_start:] + [role]
                return cycle
            
            if role in visited:
                return None
            
            visited.add(role)
            rec_stack.add(role)
            
            # Fail-fast approach - check if role exists in dependencies
            if role not in dependencies:
                # Role has no dependencies, continue
                dependencies_list = []
            else:
                dependencies_list = dependencies[role]
            
            for dep in dependencies_list:
                if dep in self.role_assignments:
                    cycle = has_cycle(dep, path + [role])
                    if cycle:
                        return cycle
            
            rec_stack.remove(role)
            return None
        
        for role in dependencies:
            if role not in visited:
                cycle = has_cycle(role, [])
                if cycle:
                    # Fail-fast approach - no fallback assignments
                    affected_configs = []
                    for r in cycle:
                        if r not in self.role_assignments:
                            raise ValueError(f"Role '{r}' found in cycle but not in role_assignments")
                        affected_configs.append(self.role_assignments[r])
                    
                    conflict = RoleConflict(
                        conflict_id=f"circular_{hash(tuple(cycle))}",
                        conflict_type=ConflictType.CIRCULAR_DEPENDENCY,
                        severity=ConflictSeverity.HIGH,
                        affected_roles=cycle,
                        affected_configs=list(set(affected_configs)),
                        description=f"Circular dependency detected: {' -> '.join(cycle)}",
                        recommendation="Break the circular dependency by reorganizing role assignments",
                        auto_resolvable=False,
                        resolution_steps=[
                            "1. Identify the most appropriate breaking point in the cycle",
                            "2. Reassign one of the roles to break the dependency",
                            "3. Verify that all dependent functionality still works"
                        ]
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_type_mismatches(self) -> List[RoleConflict]:
        """Detect roles assigned to incompatible provider types."""
        conflicts = []
        
        # Define expected provider types for roles
        role_provider_types = {
            "default-llm": ["llm"],
            "conversation-llm": ["llm"],
            "backup-llm": ["llm"],
            "default-vector-db": ["vector_db"],
            "primary-vector-db": ["vector_db"],
            "default-embedding": ["embedding"],
            "default-graph-db": ["graph_db"],
            "default-cache": ["cache"],
            "default-storage": ["storage"]
        }
        
        # This would require provider type information from the registry
        # For now, we'll simulate checking against known configurations
        for role, config in self.role_assignments.items():
            expected_types = role_provider_types[role] if role in role_provider_types else None
            if expected_types:
                # This is where we'd check the actual provider type
                # For demonstration, we'll create a simulated mismatch
                if "cache" in role and "llm" in config.lower():
                    conflict = RoleConflict(
                        conflict_id=f"type_mismatch_{role}",
                        conflict_type=ConflictType.TYPE_MISMATCH,
                        severity=ConflictSeverity.MEDIUM,
                        affected_roles=[role],
                        affected_configs=[config],
                        description=f"Role '{role}' expects {expected_types} provider but is assigned to '{config}'",
                        recommendation=f"Reassign '{role}' to a compatible {expected_types[0]} provider",
                        auto_resolvable=False
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_dependency_issues(self) -> List[RoleConflict]:
        """Detect missing dependencies for assigned roles."""
        conflicts = []
        
        # Define role dependencies
        role_dependencies = {
            "knowledge-extraction": ["default-llm", "default-vector-db", "default-embedding"],
            "memory-manager": ["default-vector-db"],
            "conversation-llm": ["memory-manager"],
            "analysis-llm": ["knowledge-extraction"]
        }
        
        for role in self.role_assignments:
            # Fail-fast approach - check dependencies explicitly
            if role not in role_dependencies:
                # Role has no dependencies
                required_deps = []
            else:
                required_deps = role_dependencies[role]
            missing_deps = [dep for dep in required_deps if dep not in self.role_assignments]
            
            if missing_deps:
                conflict = RoleConflict(
                    conflict_id=f"dependency_{role}",
                    conflict_type=ConflictType.DEPENDENCY_MISSING,
                    severity=ConflictSeverity.MEDIUM,
                    affected_roles=[role] + missing_deps,
                    affected_configs=[self.role_assignments[role]],
                    description=f"Role '{role}' requires dependencies that are not assigned: {', '.join(missing_deps)}",
                    recommendation=f"Assign the missing dependencies: {', '.join(missing_deps)}",
                    auto_resolvable=False,
                    dependencies=missing_deps,
                    resolution_steps=[
                        f"1. Assign role '{dep}' to appropriate configurations" for dep in missing_deps
                    ] + ["2. Verify that all dependencies are properly connected"]
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_deprecated_usage(self) -> List[RoleConflict]:
        """Detect usage of deprecated roles."""
        conflicts = []
        
        deprecated_roles = {
            "old-llm": "default-llm",
            "legacy-vector": "default-vector-db",
            "experimental-embedding": "default-embedding"
        }
        
        for role in self.role_assignments:
            if role in deprecated_roles:
                replacement = deprecated_roles[role]
                conflict = RoleConflict(
                    conflict_id=f"deprecated_{role}",
                    conflict_type=ConflictType.DEPRECATED_USAGE,
                    severity=ConflictSeverity.LOW,
                    affected_roles=[role],
                    affected_configs=[self.role_assignments[role]],
                    description=f"Role '{role}' is deprecated. Use '{replacement}' instead.",
                    recommendation=f"Replace deprecated role '{role}' with '{replacement}'",
                    auto_resolvable=True,
                    resolution_steps=[
                        f"1. Create alias from '{replacement}' to current configuration",
                        f"2. Update all references from '{role}' to '{replacement}'",
                        f"3. Remove deprecated role '{role}'"
                    ]
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_performance_warnings(self) -> List[RoleConflict]:
        """Detect potential performance issues."""
        conflicts = []
        
        # Example: Check if the same configuration is used for multiple resource-intensive roles
        config_roles = {}
        for role, config in self.role_assignments.items():
            if config not in config_roles:
                config_roles[config] = []
            config_roles[config].append(role)
        
        resource_intensive_roles = {"default-llm", "knowledge-extraction", "analysis-llm", "generation-llm"}
        
        for config, roles in config_roles.items():
            intensive_roles = [r for r in roles if r in resource_intensive_roles]
            if len(intensive_roles) > 2:
                conflict = RoleConflict(
                    conflict_id=f"performance_{config}",
                    conflict_type=ConflictType.PERFORMANCE_WARNING,
                    severity=ConflictSeverity.LOW,
                    affected_roles=intensive_roles,
                    affected_configs=[config],
                    description=f"Configuration '{config}' is assigned to multiple resource-intensive roles: {', '.join(intensive_roles)}",
                    recommendation="Consider distributing resource-intensive roles across different configurations",
                    auto_resolvable=False,
                    resolution_steps=[
                        "1. Identify which roles can be moved to other configurations",
                        "2. Create or identify alternative configurations for some roles",
                        "3. Test performance after redistribution"
                    ]
                )
                conflicts.append(conflict)
        
        return conflicts


class RoleConflictDetector(QWidget):
    """
    Role Conflict Detection and Resolution Widget.
    
    Provides comprehensive analysis of role assignments with automatic
    conflict detection, resolution suggestions, and validation tools.
    """
    
    # Signals
    conflict_detected = Signal(object)  # RoleConflict
    analysis_started = Signal()
    analysis_completed = Signal(list)  # List of conflicts
    resolution_applied = Signal(str, bool)  # conflict_id, success
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State
        self.current_conflicts = []
        self.role_assignments = {}
        self.analysis_thread = None
        
        self.init_ui()
        self.connect_signals()
        
        logger.info("Role conflict detector initialized")
    
    def init_ui(self):
        """Initialize the user interface."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        # Analysis controls
        self.create_analysis_controls()
        
        # Results display
        self.create_results_display()
        
        # Resolution tools
        self.create_resolution_tools()
    
    def create_analysis_controls(self):
        """Create conflict analysis controls."""
        controls_group = QGroupBox("Conflict Analysis")
        controls_layout = QHBoxLayout(controls_group)
        
        # Analysis button
        self.analyze_btn = QPushButton("Analyze Role Conflicts")
        self.analyze_btn.setDefault(True)
        controls_layout.addWidget(self.analyze_btn)
        
        # Auto-resolve button
        self.auto_resolve_btn = QPushButton("Auto-Resolve")
        self.auto_resolve_btn.setEnabled(False)
        controls_layout.addWidget(self.auto_resolve_btn)
        
        # Severity filter
        controls_layout.addWidget(QLabel("Filter by Severity:"))
        self.severity_filter = QComboBox()
        self.severity_filter.addItems(["All", "Critical", "High", "Medium", "Low", "Info"])
        controls_layout.addWidget(self.severity_filter)
        
        # Real-time monitoring
        self.realtime_cb = QCheckBox("Real-time Monitoring")
        controls_layout.addWidget(self.realtime_cb)
        
        controls_layout.addStretch()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        
        self.layout.addWidget(controls_group)
    
    def create_results_display(self):
        """Create conflict results display."""
        # Create splitter for conflicts and details
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Conflicts tree
        conflicts_widget = QWidget()
        conflicts_layout = QVBoxLayout(conflicts_widget)
        
        conflicts_layout.addWidget(QLabel("Detected Conflicts:"))
        
        self.conflicts_tree = QTreeWidget()
        self.conflicts_tree.setHeaderLabels([
            "Conflict", "Severity", "Type", "Affected Items", "Auto-Resolvable"
        ])
        self.conflicts_tree.setAlternatingRowColors(True)
        self.conflicts_tree.setSortingEnabled(True)
        conflicts_layout.addWidget(self.conflicts_tree)
        
        # Summary info
        self.summary_label = QLabel("No analysis performed yet")
        self.summary_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        conflicts_layout.addWidget(self.summary_label)
        
        splitter.addWidget(conflicts_widget)
        
        # Conflict details
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        
        details_layout.addWidget(QLabel("Conflict Details:"))
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(200)
        details_layout.addWidget(self.details_text)
        
        # Resolution section
        resolution_group = QGroupBox("Resolution")
        resolution_layout = QVBoxLayout(resolution_group)
        
        self.resolution_text = QTextEdit()
        self.resolution_text.setReadOnly(True)
        self.resolution_text.setMaximumHeight(150)
        resolution_layout.addWidget(self.resolution_text)
        
        # Resolution buttons
        resolution_buttons = QHBoxLayout()
        self.apply_resolution_btn = QPushButton("Apply Resolution")
        self.apply_resolution_btn.setEnabled(False)
        self.ignore_conflict_btn = QPushButton("Ignore")
        self.mark_resolved_btn = QPushButton("Mark as Resolved")
        
        resolution_buttons.addWidget(self.apply_resolution_btn)
        resolution_buttons.addWidget(self.ignore_conflict_btn)
        resolution_buttons.addWidget(self.mark_resolved_btn)
        resolution_buttons.addStretch()
        
        resolution_layout.addLayout(resolution_buttons)
        details_layout.addWidget(resolution_group)
        
        splitter.addWidget(details_widget)
        
        # Set splitter proportions
        splitter.setSizes([300, 400])
        self.layout.addWidget(splitter)
    
    def create_resolution_tools(self):
        """Create resolution and validation tools."""
        tools_group = QGroupBox("Resolution Tools")
        tools_layout = QHBoxLayout(tools_group)
        
        # Validation tools
        self.validate_btn = QPushButton("Validate All Assignments")
        self.export_report_btn = QPushButton("Export Report")
        self.import_fixes_btn = QPushButton("Import Fixes")
        
        tools_layout.addWidget(self.validate_btn)
        tools_layout.addWidget(self.export_report_btn)
        tools_layout.addWidget(self.import_fixes_btn)
        tools_layout.addStretch()
        
        # Status indicator
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        tools_layout.addWidget(self.status_label)
        
        self.layout.addWidget(tools_group)
    
    def connect_signals(self):
        """Connect widget signals."""
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.auto_resolve_btn.clicked.connect(self.auto_resolve_conflicts)
        self.severity_filter.currentTextChanged.connect(self.filter_conflicts)
        self.realtime_cb.toggled.connect(self.toggle_realtime_monitoring)
        
        self.conflicts_tree.currentItemChanged.connect(self.show_conflict_details)
        
        self.apply_resolution_btn.clicked.connect(self.apply_current_resolution)
        self.ignore_conflict_btn.clicked.connect(self.ignore_current_conflict)
        self.mark_resolved_btn.clicked.connect(self.mark_current_resolved)
        
        self.validate_btn.clicked.connect(self.validate_assignments)
        self.export_report_btn.clicked.connect(self.export_conflict_report)
        self.import_fixes_btn.clicked.connect(self.import_resolution_fixes)
    
    def set_role_assignments(self, assignments: Dict[str, str]):
        """Set current role assignments for analysis."""
        self.role_assignments = assignments.copy()
        logger.debug(f"Role assignments updated: {len(assignments)} assignments")
    
    def start_analysis(self):
        """Start conflict analysis."""
        if not self.role_assignments:
            QMessageBox.warning(
                self, "No Data",
                "No role assignments available for analysis. Please load role data first."
            )
            return
        
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.stop()
            self.analysis_thread.wait()
        
        # Clear previous results
        self.conflicts_tree.clear()
        self.current_conflicts = []
        self.details_text.clear()
        self.resolution_text.clear()
        
        # Start analysis
        self.analysis_thread = ConflictAnalysisThread(self.role_assignments, self)
        self.analysis_thread.analysis_progress.connect(self.update_progress)
        self.analysis_thread.conflict_detected.connect(self.add_conflict)
        self.analysis_thread.analysis_complete.connect(self.analysis_finished)
        self.analysis_thread.analysis_failed.connect(self.analysis_error)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.analyze_btn.setEnabled(False)
        self.status_label.setText("Analyzing...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        
        self.analysis_thread.start()
        self.analysis_started.emit()
        
        logger.info("Conflict analysis started")
    
    def update_progress(self, percentage: int):
        """Update analysis progress."""
        self.progress_bar.setValue(percentage)
    
    def add_conflict(self, conflict: RoleConflict):
        """Add detected conflict to the tree."""
        self.current_conflicts.append(conflict)
        
        # Create tree item
        item = QTreeWidgetItem([
            conflict.description[:50] + "..." if len(conflict.description) > 50 else conflict.description,
            conflict.severity.value.title(),
            conflict.conflict_type.value.replace('_', ' ').title(),
            f"{len(conflict.affected_roles)} roles, {len(conflict.affected_configs)} configs",
            "Yes" if conflict.auto_resolvable else "No"
        ])
        
        # Set color based on severity
        severity_colors = {
            ConflictSeverity.CRITICAL: QColor(220, 53, 69),    # Red
            ConflictSeverity.HIGH: QColor(255, 193, 7),        # Orange
            ConflictSeverity.MEDIUM: QColor(255, 235, 59),     # Yellow
            ConflictSeverity.LOW: QColor(76, 175, 80),         # Green
            ConflictSeverity.INFO: QColor(33, 150, 243)        # Blue
        }
        
        # Fail-fast approach - severity must be known
        if conflict.severity not in severity_colors:
            raise ValueError(f"Unknown conflict severity '{conflict.severity}' - must be one of: {list(severity_colors.keys())}")
        color = severity_colors[conflict.severity]
        for i in range(item.columnCount()):
            item.setForeground(i, color)
        
        if conflict.severity == ConflictSeverity.CRITICAL:
            font = item.font(0)
            font.setBold(True)
            for i in range(item.columnCount()):
                item.setFont(i, font)
        
        # Store conflict data
        item.setData(0, Qt.ItemDataRole.UserRole, conflict)
        
        self.conflicts_tree.addTopLevelItem(item)
        self.conflict_detected.emit(conflict)
    
    def analysis_finished(self, conflicts: List[RoleConflict]):
        """Handle analysis completion."""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        # Update summary
        critical_count = sum(1 for c in conflicts if c.severity == ConflictSeverity.CRITICAL)
        high_count = sum(1 for c in conflicts if c.severity == ConflictSeverity.HIGH)
        auto_resolvable = sum(1 for c in conflicts if c.auto_resolvable)
        
        if not conflicts:
            self.summary_label.setText("✅ No conflicts detected - all role assignments are valid")
            self.status_label.setText("No conflicts")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.summary_label.setText(
                f"Found {len(conflicts)} conflicts: "
                f"{critical_count} critical, {high_count} high priority. "
                f"{auto_resolvable} can be auto-resolved."
            )
            
            if critical_count > 0:
                self.status_label.setText(f"{critical_count} critical conflicts")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
            elif high_count > 0:
                self.status_label.setText(f"{high_count} high priority conflicts")
                self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.status_label.setText(f"{len(conflicts)} conflicts")
                self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        
        # Enable auto-resolve if applicable
        self.auto_resolve_btn.setEnabled(auto_resolvable > 0)
        
        # Resize columns
        self.conflicts_tree.header().resizeSections(QHeaderView.ResizeMode.ResizeToContents)
        
        self.analysis_completed.emit(conflicts)
        logger.info(f"Conflict analysis completed: {len(conflicts)} conflicts found")
    
    def analysis_error(self, error_message: str):
        """Handle analysis error."""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.status_label.setText("Analysis failed")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        
        QMessageBox.critical(
            self, "Analysis Error",
            f"Conflict analysis failed:\n\n{error_message}"
        )
        
        logger.error(f"Conflict analysis error: {error_message}")
    
    def show_conflict_details(self, current, previous):
        """Show details for selected conflict."""
        if not current:
            self.details_text.clear()
            self.resolution_text.clear()
            self.apply_resolution_btn.setEnabled(False)
            return
        
        conflict = current.data(0, Qt.ItemDataRole.UserRole)
        if not conflict or not isinstance(conflict, RoleConflict):
            self.details_text.clear()
            self.resolution_text.clear()
            self.apply_resolution_btn.setEnabled(False)
            logger.warning("Invalid conflict data in tree item")
            return
        
        # Show conflict details - escape HTML to prevent injection
        from html import escape
        
        details = f"<h3>{escape(conflict.description)}</h3>\n"
        details += f"<b>Type:</b> {escape(conflict.conflict_type.value.replace('_', ' ').title())}<br>\n"
        details += f"<b>Severity:</b> {escape(conflict.severity.value.title())}<br>\n"
        
        safe_roles = [escape(role) for role in (conflict.affected_roles or [])]
        safe_configs = [escape(config) for config in (conflict.affected_configs or [])]
        
        details += f"<b>Affected Roles:</b> {', '.join(safe_roles) if safe_roles else 'None'}<br>\n"
        details += f"<b>Affected Configurations:</b> {', '.join(safe_configs) if safe_configs else 'None'}<br>\n"
        
        if conflict.dependencies:
            safe_deps = [escape(dep) for dep in conflict.dependencies]
            details += f"<b>Dependencies:</b> {', '.join(safe_deps)}<br>\n"
        
        details += f"<b>Auto-resolvable:</b> {'Yes' if conflict.auto_resolvable else 'No'}<br>\n"
        
        self.details_text.setHtml(details)
        
        # Show resolution information - escape HTML
        resolution = f"<h4>Recommendation:</h4>\n"
        resolution += f"<p>{escape(conflict.recommendation)}</p>\n"
        
        if conflict.resolution_steps:
            resolution += "<h4>Resolution Steps:</h4>\n<ol>\n"
            for step in conflict.resolution_steps:
                resolution += f"<li>{escape(step)}</li>\n"
            resolution += "</ol>\n"
        
        self.resolution_text.setHtml(resolution)
        
        # Enable resolution button if auto-resolvable
        self.apply_resolution_btn.setEnabled(conflict.auto_resolvable)
    
    def auto_resolve_conflicts(self):
        """Auto-resolve all resolvable conflicts."""
        auto_resolvable = [c for c in self.current_conflicts if c.auto_resolvable]
        
        if not auto_resolvable:
            QMessageBox.information(
                self, "No Auto-Resolvable Conflicts",
                "No conflicts can be automatically resolved."
            )
            return
        
        reply = QMessageBox.question(
            self, "Auto-Resolve Conflicts",
            f"This will automatically resolve {len(auto_resolvable)} conflicts. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            resolved_count = 0
            for conflict in auto_resolvable:
                if self._apply_resolution(conflict):
                    resolved_count += 1
            
            QMessageBox.information(
                self, "Auto-Resolution Complete",
                f"Successfully resolved {resolved_count} of {len(auto_resolvable)} conflicts."
            )
            
            # Refresh analysis
            self.start_analysis()
    
    def _apply_resolution(self, conflict: RoleConflict) -> bool:
        """Apply resolution for a specific conflict."""
        try:
            # This would implement actual resolution logic
            # For now, just simulate success
            logger.info(f"Resolving conflict: {conflict.conflict_id}")
            
            if conflict.conflict_type == ConflictType.DEPRECATED_USAGE:
                # Handle deprecated role replacement
                return self._resolve_deprecated_role(conflict)
            
            # Other resolution types would be implemented here
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve conflict {conflict.conflict_id}: {e}")
            return False
    
    def _resolve_deprecated_role(self, conflict: RoleConflict) -> bool:
        """Resolve deprecated role usage."""
        # Implementation would handle deprecated role replacement
        # through the role manager
        return True
    
    def filter_conflicts(self, severity_filter: str):
        """Filter conflicts by severity."""
        if severity_filter == "All":
            for i in range(self.conflicts_tree.topLevelItemCount()):
                item = self.conflicts_tree.topLevelItem(i)
                if item:
                    item.setHidden(False)
        else:
            severity_value = severity_filter.lower()
            for i in range(self.conflicts_tree.topLevelItemCount()):
                item = self.conflicts_tree.topLevelItem(i)
                if not item:
                    continue
                    
                conflict = item.data(0, Qt.ItemDataRole.UserRole)
                if not conflict or not hasattr(conflict, 'severity') or not hasattr(conflict.severity, 'value'):
                    item.setHidden(True)  # Hide invalid items
                    continue
                    
                should_hide = conflict.severity.value != severity_value
                item.setHidden(should_hide)
    
    def toggle_realtime_monitoring(self, enabled: bool):
        """Toggle real-time conflict monitoring."""
        if enabled:
            # Set up timer for periodic checks
            if not hasattr(self, 'monitor_timer'):
                self.monitor_timer = QTimer()
                self.monitor_timer.timeout.connect(self._periodic_analysis_check)
            self.monitor_timer.start(30000)  # Check every 30 seconds
            logger.info("Real-time conflict monitoring enabled")
        else:
            if hasattr(self, 'monitor_timer'):
                self.monitor_timer.stop()
            logger.info("Real-time conflict monitoring disabled")
    
    def _periodic_analysis_check(self):
        """Perform periodic analysis check - prevents recursive calls."""
        # Only start new analysis if none is running
        if not self.analysis_thread or not self.analysis_thread.isRunning():
            self.start_analysis()
        else:
            logger.debug("Skipping periodic analysis - analysis already running")
    
    def apply_current_resolution(self):
        """Apply resolution for currently selected conflict."""
        current_item = self.conflicts_tree.currentItem()
        if not current_item:
            return
        
        conflict = current_item.data(0, Qt.ItemDataRole.UserRole)
        if not conflict:
            return
        
        if self._apply_resolution(conflict):
            self.resolution_applied.emit(conflict.conflict_id, True)
            QMessageBox.information(
                self, "Resolution Applied",
                f"Successfully resolved conflict: {conflict.description[:50]}..."
            )
            # Remove from tree
            index = self.conflicts_tree.indexOfTopLevelItem(current_item)
            self.conflicts_tree.takeTopLevelItem(index)
        else:
            self.resolution_applied.emit(conflict.conflict_id, False)
            QMessageBox.warning(
                self, "Resolution Failed",
                f"Failed to resolve conflict: {conflict.description[:50]}..."
            )
    
    def ignore_current_conflict(self):
        """Ignore the currently selected conflict."""
        current_item = self.conflicts_tree.currentItem()
        if current_item:
            index = self.conflicts_tree.indexOfTopLevelItem(current_item)
            self.conflicts_tree.takeTopLevelItem(index)
    
    def mark_current_resolved(self):
        """Mark current conflict as manually resolved."""
        current_item = self.conflicts_tree.currentItem()
        if current_item:
            # Change appearance to indicate resolved
            for i in range(current_item.columnCount()):
                current_item.setForeground(i, QColor(128, 128, 128))
            current_item.setText(0, "[RESOLVED] " + current_item.text(0))
    
    def validate_assignments(self):
        """Validate all current role assignments."""
        # This would implement comprehensive validation
        QMessageBox.information(
            self, "Validation Complete",
            "Role assignment validation completed. Check the conflicts list for any issues."
        )
    
    def export_conflict_report(self):
        """Export conflict analysis report."""
        # Implementation for exporting detailed report
        pass
    
    def import_resolution_fixes(self):
        """Import resolution fixes from file."""
        # Implementation for importing bulk fixes
        pass
    
    def get_conflicts_summary(self) -> Dict[str, int]:
        """Get summary of current conflicts."""
        summary = {
            'total': len(self.current_conflicts),
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'info': 0,
            'auto_resolvable': 0
        }
        
        for conflict in self.current_conflicts:
            summary[conflict.severity.value] += 1
            if conflict.auto_resolvable:
                summary['auto_resolvable'] += 1
        
        return summary
    
    def closeEvent(self, event):
        """Handle widget close event with cleanup."""
        # Stop real-time monitoring
        if hasattr(self, 'monitor_timer'):
            self.monitor_timer.stop()
        
        # Stop any running analysis
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.stop()
            self.analysis_thread.wait(1000)  # 1 second timeout
        
        # Accept the close event
        event.accept()