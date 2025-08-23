"""
Drag-and-Drop Role Assignment Widget

Advanced drag-and-drop interface for assigning roles to providers with visual feedback,
validation, and smooth animations following CLAUDE.md principles.
"""

import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QScrollArea, QListWidget, QListWidgetItem, QGroupBox, QSplitter,
    QMessageBox, QToolTip, QApplication
)
from PySide6.QtCore import (
    Qt, Signal, QMimeData, QPoint, QRect, QTimer, QPropertyAnimation,
    QEasingCurve, QParallelAnimationGroup, Property
)
from PySide6.QtGui import (
    QDrag, QPainter, QPen, QBrush, QColor, QFont, QPixmap, QPalette,
    QLinearGradient, QDropEvent, QDragEnterEvent, QDragMoveEvent
)

logger = logging.getLogger(__name__)


class DraggableRoleItem(QListWidgetItem):
    """Draggable role item with enhanced visual feedback."""
    
    def __init__(self, role_name: str, role_data: Dict[str, Any]):
        super().__init__(role_name)
        self.role_name = role_name
        self.role_data = role_data
        self.original_parent = None
        
        # Visual styling
        self.setFlags(self.flags() | Qt.ItemFlag.ItemIsDragEnabled)
        self.setup_styling()
    
    def setup_styling(self):
        """Set up visual styling for the role item."""
        # Color coding based on role type - fail-fast approach
        if 'type' not in self.role_data:
            raise ValueError(f"Role data missing required 'type' field for role: {self.role_name}")
        role_type = self.role_data['type']
        colors = {
            'primary': QColor(52, 152, 219),    # Blue
            'secondary': QColor(155, 89, 182),  # Purple  
            'default': QColor(149, 165, 166),   # Gray
            'critical': QColor(231, 76, 60),    # Red
            'warning': QColor(241, 196, 15)     # Yellow
        }
        
        # Strict color mapping - no fallbacks
        if role_type not in colors:
            raise ValueError(f"Unknown role type '{role_type}' - must be one of: {list(colors.keys())}")
        color = colors[role_type]
        self.setBackground(QBrush(color.lighter(180)))
        self.setForeground(QBrush(color.darker(120)))
        
        # Tooltip with role information
        tooltip = f"Role: {self.role_name}\n"
        tooltip += f"Type: {role_type.title()}\n"
        if 'description' in self.role_data:
            tooltip += f"Description: {self.role_data['description']}\n"
        if 'dependencies' in self.role_data:
            deps = ', '.join(self.role_data['dependencies'])
            tooltip += f"Dependencies: {deps}"
        
        self.setToolTip(tooltip)


class DropZoneWidget(QFrame):
    """Enhanced drop zone with visual feedback and validation."""
    
    role_dropped = Signal(str, str, dict)  # provider_name, role_name, role_data
    
    def __init__(self, provider_name: str, provider_data: Dict[str, Any]):
        super().__init__()
        self.provider_name = provider_name
        self.provider_data = provider_data
        self.assigned_roles: Set[str] = set()
        self.is_drag_over = False
        self.animation_group = None
        
        self.setup_ui()
        self.setup_drag_drop()
        self.setup_animations()
    
    def setup_ui(self):
        """Set up the drop zone UI."""
        self.setMinimumHeight(120)
        self.setMaximumHeight(200)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setLineWidth(2)
        
        layout = QVBoxLayout(self)
        
        # Provider header
        header = QLabel(f"Provider: {self.provider_name}")
        header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Provider type and status
        # Fail-fast approach - no fallbacks
        if 'type' not in self.provider_data:
            raise ValueError(f"Provider data missing required 'type' field for provider: {self.provider_name}")
        provider_type = self.provider_data['type']
        status_label = QLabel(f"Type: {provider_type}")
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status_label)
        
        # Assigned roles area
        self.roles_area = QScrollArea()
        self.roles_widget = QWidget()
        self.roles_layout = QVBoxLayout(self.roles_widget)
        self.roles_area.setWidget(self.roles_widget)
        self.roles_area.setMaximumHeight(80)
        layout.addWidget(self.roles_area)
        
        # Drop instruction
        self.drop_label = QLabel("Drop roles here")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.drop_label)
        
        self.update_styling()
    
    def setup_drag_drop(self):
        """Configure drag and drop acceptance."""
        self.setAcceptDrops(True)
    
    def setup_animations(self):
        """Set up smooth animations for visual feedback."""
        self.animation_group = QParallelAnimationGroup()
        
        # Border animation
        self.border_animation = QPropertyAnimation(self, b"border_width")
        self.border_animation.setDuration(200)
        self.border_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        # Background animation  
        self.bg_animation = QPropertyAnimation(self, b"background_opacity")
        self.bg_animation.setDuration(200)
        self.bg_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        self.animation_group.addAnimation(self.border_animation)
        self.animation_group.addAnimation(self.bg_animation)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter with validation and visual feedback."""
        if event.mimeData().hasText():
            try:
                # Parse role data from mime data
                role_name = event.mimeData().text()
                
                # Validate role assignment
                if self.can_accept_role(role_name):
                    event.acceptProposedAction()
                    self.start_drag_over_animation()
                    self.is_drag_over = True
                    
                    # Show validation tooltip
                    QToolTip.showText(
                        event.position().toPoint() + self.mapToGlobal(QPoint(0, 0)),
                        f"✓ Can assign '{role_name}' to {self.provider_name}",
                        self,
                        QRect(),
                        3000
                    )
                else:
                    event.ignore()
                    # Show rejection reason
                    reason = self.get_rejection_reason(role_name)
                    QToolTip.showText(
                        event.position().toPoint() + self.mapToGlobal(QPoint(0, 0)),
                        f"✗ Cannot assign '{role_name}': {reason}",
                        self,
                        QRect(),
                        3000
                    )
                    
            except Exception as e:
                logger.error(f"Error in dragEnterEvent: {e}")
                event.ignore()
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave event."""
        self.is_drag_over = False
        self.start_drag_leave_animation()
        super().dragLeaveEvent(event)
    
    def dragMoveEvent(self, event: QDragMoveEvent):
        """Handle drag move events with continuous validation."""
        if self.is_drag_over:
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """Handle role drop with validation and feedback."""
        if event.mimeData().hasText():
            try:
                role_name = event.mimeData().text()
                
                # Validate before assignment
                if self.can_accept_role(role_name):
                    # Get role data from source
                    source_widget = event.source()
                    role_data = {}
                    
                    if hasattr(source_widget, 'currentItem'):
                        current_item = source_widget.currentItem()
                        if isinstance(current_item, DraggableRoleItem):
                            role_data = current_item.role_data
                    
                    # Assign role
                    self.assign_role(role_name, role_data)
                    
                    # Emit signal
                    self.role_dropped.emit(self.provider_name, role_name, role_data)
                    
                    event.acceptProposedAction()
                    
                    # Success animation
                    self.start_success_animation()
                    
                    logger.info(f"Successfully assigned role '{role_name}' to provider '{self.provider_name}'")
                    
                else:
                    event.ignore()
                    # Show error message
                    reason = self.get_rejection_reason(role_name)
                    QMessageBox.warning(
                        self,
                        "Role Assignment Failed",
                        f"Cannot assign role '{role_name}' to provider '{self.provider_name}':\n\n{reason}"
                    )
                    
            except Exception as e:
                logger.error(f"Error in dropEvent: {e}")
                event.ignore()
                QMessageBox.critical(self, "Drop Error", f"Failed to assign role: {str(e)}")
        
        # Reset visual state
        self.is_drag_over = False
        self.start_drag_leave_animation()
    
    def can_accept_role(self, role_name: str) -> bool:
        """Validate if this provider can accept the given role."""
        try:
            # Check if role is already assigned
            if role_name in self.assigned_roles:
                return False
            
            # Check provider compatibility
            # Fail-fast approach - no fallbacks
            if 'type' not in self.provider_data:
                raise ValueError(f"Provider data missing required 'type' field for provider: {self.provider_name}")
            provider_type = self.provider_data['type']
            
            # Role compatibility rules (can be extended)
            role_compatibility = {
                'primary': ['llm', 'database', 'vector_db'],
                'secondary': ['cache', 'storage', 'embedding'],
                'default': ['*'],  # Can be assigned to any provider
                'critical': ['llm'],  # Only LLM providers
            }
            
            # For now, accept all roles (can be customized)
            return True
            
        except Exception as e:
            logger.error(f"Error validating role assignment: {e}")
            return False
    
    def get_rejection_reason(self, role_name: str) -> str:
        """Get human-readable reason why role cannot be assigned."""
        if role_name in self.assigned_roles:
            return f"Role '{role_name}' is already assigned to this provider"
        
        # Add more specific rejection reasons as needed
        return "Provider incompatible with this role type"
    
    def assign_role(self, role_name: str, role_data: Dict[str, Any]):
        """Assign a role to this provider."""
        if role_name not in self.assigned_roles:
            self.assigned_roles.add(role_name)
            
            # Create visual role badge
            role_badge = QPushButton(role_name)
            role_badge.setMaximumHeight(25)
            role_badge.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    border-radius: 12px;
                    padding: 4px 8px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
            """)
            
            # Add remove functionality
            role_badge.clicked.connect(lambda: self.remove_role(role_name, role_badge))
            
            self.roles_layout.addWidget(role_badge)
            self.update_drop_label()
    
    def remove_role(self, role_name: str, role_badge: QPushButton):
        """Remove a role from this provider."""
        if role_name in self.assigned_roles:
            self.assigned_roles.remove(role_name)
            role_badge.deleteLater()
            self.update_drop_label()
            logger.info(f"Removed role '{role_name}' from provider '{self.provider_name}'")
    
    def update_drop_label(self):
        """Update the drop instruction label."""
        if self.assigned_roles:
            self.drop_label.setText(f"{len(self.assigned_roles)} roles assigned")
        else:
            self.drop_label.setText("Drop roles here")
    
    def start_drag_over_animation(self):
        """Start animation when drag enters."""
        self.border_animation.setStartValue(2)
        self.border_animation.setEndValue(4)
        self.bg_animation.setStartValue(0.0)
        self.bg_animation.setEndValue(0.1)
        self.animation_group.start()
    
    def start_drag_leave_animation(self):
        """Start animation when drag leaves."""
        self.border_animation.setStartValue(4)
        self.border_animation.setEndValue(2)
        self.bg_animation.setStartValue(0.1)
        self.bg_animation.setEndValue(0.0)
        self.animation_group.start()
    
    def start_success_animation(self):
        """Start success animation after successful drop."""
        # Flash green briefly
        original_style = self.styleSheet()
        self.setStyleSheet("QFrame { border: 3px solid #27ae60; background-color: rgba(39, 174, 96, 0.1); }")
        
        # Reset after delay
        QTimer.singleShot(500, lambda: self.setStyleSheet(original_style))
    
    def update_styling(self):
        """Update visual styling based on current state."""
        base_style = """
            QFrame {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                background-color: rgba(236, 240, 241, 0.3);
            }
        """
        
        if self.is_drag_over:
            base_style = """
                QFrame {
                    border: 4px solid #3498db;
                    border-radius: 8px;
                    background-color: rgba(52, 152, 219, 0.1);
                }
            """
        
        self.setStyleSheet(base_style)
    
    # Property for animations
    def get_border_width(self):
        return self._border_width if hasattr(self, '_border_width') else 2
    
    def set_border_width(self, width):
        self._border_width = width
        self.update_styling()
    
    border_width = Property(int, get_border_width, set_border_width)
    
    def get_background_opacity(self):
        return self._bg_opacity if hasattr(self, '_bg_opacity') else 0.0
    
    def set_background_opacity(self, opacity):
        self._bg_opacity = opacity
        self.update_styling()
    
    background_opacity = Property(float, get_background_opacity, set_background_opacity)


class RoleLibraryWidget(QListWidget):
    """Enhanced role library with search and categorization."""
    
    def __init__(self):
        super().__init__()
        self.setDragDropMode(QListWidget.DragDropMode.DragOnly)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.load_available_roles()
        self.setup_styling()
    
    def load_available_roles(self):
        """Load available roles from configuration or registry."""
        # Sample roles - in production, these would come from registry
        sample_roles = [
            {"name": "primary-llm", "type": "critical", "description": "Primary language model provider"},
            {"name": "backup-llm", "type": "secondary", "description": "Backup language model provider"},
            {"name": "vector-store", "type": "primary", "description": "Primary vector storage"},
            {"name": "cache-layer", "type": "secondary", "description": "Caching layer provider"},
            {"name": "embedding-service", "type": "default", "description": "Text embedding service"},
            {"name": "knowledge-graph", "type": "primary", "description": "Knowledge graph storage"},
            {"name": "monitoring", "type": "default", "description": "System monitoring role"},
        ]
        
        for role_config in sample_roles:
            role_item = DraggableRoleItem(role_config["name"], role_config)
            self.addItem(role_item)
    
    def setup_styling(self):
        """Set up visual styling for the role library."""
        self.setStyleSheet("""
            QListWidget {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: #ecf0f1;
                selection-background-color: #3498db;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #d5dbdb;
            }
            QListWidget::item:hover {
                background-color: #d5dbdb;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)


class DragDropRoleWidget(QWidget):
    """Main drag-and-drop role assignment interface."""
    
    role_assigned = Signal(str, str, dict)  # provider_name, role_name, role_data
    role_unassigned = Signal(str, str)      # provider_name, role_name
    
    def __init__(self):
        super().__init__()
        self.providers: Dict[str, Dict[str, Any]] = {}
        self.drop_zones: Dict[str, DropZoneWidget] = {}
        
        self.setup_ui()
        self.load_providers()
    
    def setup_ui(self):
        """Set up the main drag-and-drop interface."""
        layout = QHBoxLayout(self)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - Role Library
        left_panel = QGroupBox("Available Roles")
        left_layout = QVBoxLayout(left_panel)
        
        # Role library
        self.role_library = RoleLibraryWidget()
        left_layout.addWidget(self.role_library)
        
        # Instructions
        instructions = QLabel(
            "Drag roles from here to provider drop zones on the right.\n"
            "Hover over roles for detailed information."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #7f8c8d; font-size: 10px; padding: 5px;")
        left_layout.addWidget(instructions)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Provider Drop Zones
        right_panel = QGroupBox("Provider Assignment")
        right_layout = QVBoxLayout(right_panel)
        
        # Scroll area for providers
        self.providers_scroll = QScrollArea()
        self.providers_widget = QWidget()
        self.providers_layout = QVBoxLayout(self.providers_widget)
        self.providers_scroll.setWidget(self.providers_widget)
        self.providers_scroll.setWidgetResizable(True)
        right_layout.addWidget(self.providers_scroll)
        
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([300, 500])
    
    def load_providers(self):
        """Load available providers and create drop zones."""
        # Sample providers - in production, these would come from registry
        sample_providers = [
            {"name": "llamacpp-primary", "type": "llm", "status": "active"},
            {"name": "openai-backup", "type": "llm", "status": "standby"},
            {"name": "chroma-vectors", "type": "vector_db", "status": "active"},
            {"name": "redis-cache", "type": "cache", "status": "active"},
            {"name": "postgres-main", "type": "database", "status": "active"},
        ]
        
        for provider_config in sample_providers:
            self.add_provider(provider_config["name"], provider_config)
    
    def add_provider(self, provider_name: str, provider_data: Dict[str, Any]):
        """Add a new provider drop zone."""
        if provider_name not in self.providers:
            self.providers[provider_name] = provider_data
            
            # Create drop zone
            drop_zone = DropZoneWidget(provider_name, provider_data)
            drop_zone.role_dropped.connect(self._on_role_dropped)
            
            self.drop_zones[provider_name] = drop_zone
            self.providers_layout.addWidget(drop_zone)
            
            logger.info(f"Added provider drop zone: {provider_name}")
    
    def remove_provider(self, provider_name: str):
        """Remove a provider drop zone."""
        if provider_name in self.drop_zones:
            drop_zone = self.drop_zones[provider_name]
            drop_zone.deleteLater()
            del self.drop_zones[provider_name]
            del self.providers[provider_name]
            
            logger.info(f"Removed provider drop zone: {provider_name}")
    
    def _on_role_dropped(self, provider_name: str, role_name: str, role_data: Dict[str, Any]):
        """Handle role drop events."""
        self.role_assigned.emit(provider_name, role_name, role_data)
        logger.info(f"Role '{role_name}' assigned to provider '{provider_name}'")
    
    def get_role_assignments(self) -> Dict[str, List[str]]:
        """Get current role assignments for all providers."""
        assignments = {}
        for provider_name, drop_zone in self.drop_zones.items():
            assignments[provider_name] = list(drop_zone.assigned_roles)
        return assignments
    
    def clear_all_assignments(self):
        """Clear all role assignments."""
        for drop_zone in self.drop_zones.values():
            drop_zone.assigned_roles.clear()
            # Clear visual role badges
            for i in reversed(range(drop_zone.roles_layout.count())):
                child = drop_zone.roles_layout.itemAt(i).widget()
                if child:
                    child.deleteLater()
            drop_zone.update_drop_label()
        
        logger.info("Cleared all role assignments")