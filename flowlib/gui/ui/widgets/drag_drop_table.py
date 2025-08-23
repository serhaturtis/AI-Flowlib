"""
Drag and Drop Table Widget for Configuration Management.

Provides enhanced table functionality with drag-and-drop support
for file operations and configuration management.
"""

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QMessageBox
from PySide6.QtCore import Qt, Signal, QMimeData
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QDragMoveEvent

logger = logging.getLogger(__name__)


class DragDropConfigurationTable(QTableWidget):
    """
    Enhanced QTableWidget with drag-and-drop support for configuration files.
    
    Supports:
    - Drag files from external applications
    - Drop configuration files for import
    - Drag configurations within table for reordering
    """
    
    # Signals for drag-and-drop operations
    files_dropped = Signal(list)  # List of file paths dropped
    configuration_moved = Signal(int, int)  # from_row, to_row for reordering
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QTableWidget.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        
        # Configure selection behavior
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        
        # Supported file types for configuration import
        self.supported_extensions = {'.py', '.yaml', '.yml', '.json', '.toml'}
        
        logger.info("DragDropConfigurationTable initialized")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events."""
        if event.mimeData().hasUrls():
            # Check if any URLs are files with supported extensions
            valid_files = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = Path(url.toLocalFile())
                    if file_path.suffix.lower() in self.supported_extensions:
                        valid_files.append(file_path)
            
            if valid_files:
                event.acceptProposedAction()
                logger.debug(f"Drag enter accepted for {len(valid_files)} valid files")
            else:
                event.ignore()
                logger.debug("Drag enter ignored - no valid configuration files")
        elif event.mimeData().hasText():
            # Allow text drops (could be configuration content)
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dragMoveEvent(self, event: QDragMoveEvent):
        """Handle drag move events."""
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop events."""
        try:
            if event.mimeData().hasUrls():
                # Handle file drops
                dropped_files = []
                for url in event.mimeData().urls():
                    if url.isLocalFile():
                        file_path = Path(url.toLocalFile())
                        if file_path.exists() and file_path.suffix.lower() in self.supported_extensions:
                            dropped_files.append(str(file_path))
                
                if dropped_files:
                    logger.info(f"Files dropped: {dropped_files}")
                    self.files_dropped.emit(dropped_files)
                    event.acceptProposedAction()
                else:
                    QMessageBox.warning(self, "Invalid Files", 
                                      "Only configuration files (.py, .yaml, .yml, .json, .toml) are supported.")
                    event.ignore()
            
            elif event.mimeData().hasText():
                # Handle text drops (configuration content)
                text_content = event.mimeData().text()
                if text_content.strip():
                    # For now, just show the content in a message box
                    # In a full implementation, this could open the configuration editor
                    QMessageBox.information(self, "Configuration Content", 
                                          f"Received configuration content:\n\n{text_content[:200]}...")
                    event.acceptProposedAction()
                else:
                    event.ignore()
            
            else:
                event.ignore()
                
        except Exception as e:
            logger.error(f"Error handling drop event: {e}")
            QMessageBox.critical(self, "Drop Error", f"Failed to handle dropped items: {str(e)}")
            event.ignore()
    
    def startDrag(self, supportedActions):
        """Start drag operation for internal reordering."""
        selected_row = self.currentRow()
        if selected_row >= 0:
            # Create mime data with row information
            mime_data = QMimeData()
            mime_data.setText(f"row:{selected_row}")
            
            # Get configuration name for display
            config_name = ""
            if self.item(selected_row, 0):
                config_name = self.item(selected_row, 0).text()
            
            mime_data.setData("application/x-configuration-row", config_name.encode())
            
            # Start the drag
            drag = self.startDrag(supportedActions)
            if drag:
                drag.setMimeData(mime_data)
                drag.exec(supportedActions)
    
    def get_supported_file_types(self) -> List[str]:
        """Get list of supported file extensions."""
        return list(self.supported_extensions)
    
    def add_supported_file_type(self, extension: str):
        """Add a new supported file extension."""
        if not extension.startswith('.'):
            extension = '.' + extension
        self.supported_extensions.add(extension.lower())
        logger.debug(f"Added supported file type: {extension}")
    
    def remove_supported_file_type(self, extension: str):
        """Remove a supported file extension."""
        if not extension.startswith('.'):
            extension = '.' + extension
        self.supported_extensions.discard(extension.lower())
        logger.debug(f"Removed supported file type: {extension}")
    
    def validate_file_for_import(self, file_path: Path) -> bool:
        """Validate if a file can be imported as configuration."""
        if not file_path.exists():
            return False
        
        if not file_path.is_file():
            return False
        
        if file_path.suffix.lower() not in self.supported_extensions:
            return False
        
        # Additional validation could be added here
        # e.g., file size limits, content validation, etc.
        
        return True
    
    def get_drop_indicator_position(self, event_pos) -> Optional[int]:
        """Get the row position where items would be dropped."""
        item = self.itemAt(event_pos)
        if item:
            return item.row()
        
        # If no item, determine if dropping at end
        row_count = self.rowCount()
        if row_count > 0:
            last_item = self.item(row_count - 1, 0)
            if last_item:
                item_rect = self.visualItemRect(last_item)
                if event_pos.y() > item_rect.bottom():
                    return row_count  # Drop at end
        
        return 0  # Drop at beginning if no rows