"""
Enhanced File Browser Dialog

Provides advanced file browsing with search, filtering, metadata display,
and comprehensive file operations for configuration management.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QTreeWidget, QTreeWidgetItem,
    QLabel, QPushButton, QLineEdit, QComboBox, QListWidget, QListWidgetItem,
    QGroupBox, QGridLayout, QCheckBox, QMessageBox, QFileDialog, QProgressBar,
    QToolBar, QToolButton, QStatusBar, QTextEdit, QTabWidget, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QMenu, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, QDir, QFileInfo
from PySide6.QtGui import QFont, QIcon, QAction, QPixmap

logger = logging.getLogger(__name__)


class FileSearchWorker(QThread):
    """Worker thread for file searching."""
    
    search_completed = Signal(list)  # List of matching files
    search_progress = Signal(str)    # Current file being searched
    
    def __init__(self, search_path: str, search_term: str, file_types: List[str]):
        super().__init__()
        self.search_path = search_path
        self.search_term = search_term.lower()
        self.file_types = file_types
        self.is_cancelled = False
    
    def run(self):
        """Run file search in background thread."""
        try:
            matches = []
            search_path = Path(self.search_path)
            
            if not search_path.exists():
                self.search_completed.emit([])
                return
            
            # Search through files
            for file_path in search_path.rglob("*"):
                if self.is_cancelled:
                    break
                
                if file_path.is_file():
                    self.search_progress.emit(str(file_path))
                    
                    # Check file type filter
                    if self.file_types and file_path.suffix not in self.file_types:
                        continue
                    
                    # Check filename match
                    if self.search_term in file_path.name.lower():
                        matches.append({
                            'path': str(file_path),
                            'name': file_path.name,
                            'size': file_path.stat().st_size,
                            'modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                            'type': file_path.suffix or 'file'
                        })
                        continue
                    
                    # Check content match for text files
                    if file_path.suffix in ['.py', '.yaml', '.yml', '.json', '.txt', '.md']:
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read(10000)  # Read first 10KB
                                if self.search_term in content.lower():
                                    matches.append({
                                        'path': str(file_path),
                                        'name': file_path.name,
                                        'size': file_path.stat().st_size,
                                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                                        'type': file_path.suffix or 'file',
                                        'content_match': True
                                    })
                        except (IOError, OSError, UnicodeDecodeError) as read_error:
                            # Skip files that can't be read or decoded
                            logger.debug(f"Could not read file for search '{file_path}': {read_error}")
            
            self.search_completed.emit(matches)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            self.search_completed.emit([])
    
    def cancel(self):
        """Cancel the search operation."""
        self.is_cancelled = True


class FileBrowserDialog(QDialog):
    """Enhanced file browser with advanced features."""
    
    file_selected = Signal(str)     # Selected file path
    files_selected = Signal(list)   # Multiple selected file paths
    
    def __init__(self, initial_path: str = "", file_types: List[str] = None, 
                 multiple_selection: bool = False, parent=None):
        super().__init__(parent)
        self.initial_path = initial_path or str(Path.home())
        self.file_types = file_types or ['.py', '.yaml', '.yml', '.json']
        self.multiple_selection = multiple_selection
        self.current_path = Path(self.initial_path)
        self.selected_files = []
        self.search_worker = None
        self.bookmarks = []
        
        self.setWindowTitle("File Browser")
        self.setMinimumSize(1000, 700)
        self.init_ui()
        self.load_bookmarks()
        self.refresh_file_tree()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Toolbar
        self.create_toolbar(layout)
        
        # Control panel for path, search, and filters
        self.create_control_panel(layout)
        
        # Main content
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left panel - Navigation and bookmarks
        self.create_navigation_panel(main_splitter)
        
        # Center panel - File tree and list
        self.create_file_panel(main_splitter)
        
        # Right panel - File details and preview
        self.create_details_panel(main_splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Buttons
        self.create_buttons(layout)
        
        # Set splitter proportions
        main_splitter.setSizes([200, 500, 300])
    
    def create_toolbar(self, layout):
        """Create the toolbar with navigation and search."""
        toolbar = QToolBar()
        layout.addWidget(toolbar)
        
        # Navigation actions
        back_action = QAction("← Back", self)
        back_action.setShortcut("Alt+Left")
        back_action.setToolTip("Go to parent directory (Alt+Left)")
        back_action.triggered.connect(self.go_back)
        toolbar.addAction(back_action)
        
        home_action = QAction("🏠 Home", self)
        home_action.setShortcut("Alt+Home")
        home_action.setToolTip("Go to home directory (Alt+Home)")
        home_action.triggered.connect(self.go_home)
        toolbar.addAction(home_action)
    
    def create_control_panel(self, layout):
        """Create the control panel with path, search, and filters."""
        control_group = QGroupBox("Navigation & Search")
        control_layout = QHBoxLayout(control_group)
        layout.addWidget(control_group)
        
        # Path display
        control_layout.addWidget(QLabel("Path:"))
        self.path_edit = QLineEdit(str(self.current_path))
        self.path_edit.setPlaceholderText("Enter directory path...")
        self.path_edit.returnPressed.connect(self.navigate_to_path)
        control_layout.addWidget(self.path_edit)
        
        # Search
        control_layout.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search files and content...")
        self.search_edit.returnPressed.connect(self.start_search)
        control_layout.addWidget(self.search_edit)
        
        # File type filter
        control_layout.addWidget(QLabel("Type:"))
        self.type_filter = QComboBox()
        self.type_filter.addItems(["All", "Python (.py)", "YAML (.yaml)", "JSON (.json)", "Text (.txt)"])
        self.type_filter.currentTextChanged.connect(self.on_filter_changed)
        control_layout.addWidget(self.type_filter)
        
        # Search button (keeping for explicit search action)
        search_button = QPushButton("Search")
        search_button.setShortcut("Ctrl+F")
        search_button.setToolTip("Search files and content (Ctrl+F)")
        search_button.clicked.connect(self.start_search)
        control_layout.addWidget(search_button)
    
    def create_navigation_panel(self, splitter):
        """Create the navigation and bookmarks panel."""
        nav_widget = QWidget()
        layout = QVBoxLayout(nav_widget)
        
        # Quick access
        quick_group = QGroupBox("Quick Access")
        quick_layout = QVBoxLayout(quick_group)
        
        # Common locations
        locations = [
            ("🏠 Home", str(Path.home())),
            ("📁 Documents", str(Path.home() / "Documents")),
            ("⚙️ Flowlib Config", str(Path.home() / ".flowlib")),
            ("🗂️ Current Project", str(Path.cwd()))
        ]
        
        for name, path in locations:
            button = QPushButton(name)
            button.clicked.connect(lambda checked, p=path: self.navigate_to(p))
            quick_layout.addWidget(button)
        
        layout.addWidget(quick_group)
        
        # Bookmarks
        bookmarks_group = QGroupBox("Bookmarks")
        bookmarks_layout = QVBoxLayout(bookmarks_group)
        
        self.bookmarks_list = QListWidget()
        self.bookmarks_list.itemDoubleClicked.connect(self.navigate_to_bookmark)
        bookmarks_layout.addWidget(self.bookmarks_list)
        
        # Bookmark buttons
        bookmark_buttons = QHBoxLayout()
        add_bookmark_btn = QPushButton("Add")
        add_bookmark_btn.clicked.connect(self.add_bookmark)
        bookmark_buttons.addWidget(add_bookmark_btn)
        
        remove_bookmark_btn = QPushButton("Remove")
        remove_bookmark_btn.clicked.connect(self.remove_bookmark)
        bookmark_buttons.addWidget(remove_bookmark_btn)
        
        bookmarks_layout.addLayout(bookmark_buttons)
        layout.addWidget(bookmarks_group)
        
        layout.addStretch()
        splitter.addWidget(nav_widget)
    
    def create_file_panel(self, splitter):
        """Create the main file browsing panel."""
        file_widget = QWidget()
        layout = QVBoxLayout(file_widget)
        
        # View mode selector
        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("View:"))
        
        self.view_mode = QComboBox()
        self.view_mode.addItems(["Tree View", "List View", "Search Results"])
        self.view_mode.currentTextChanged.connect(self.switch_view_mode)
        view_layout.addWidget(self.view_mode)
        
        view_layout.addStretch()
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_file_tree)
        view_layout.addWidget(refresh_btn)
        
        layout.addLayout(view_layout)
        
        # File views (stacked)
        self.file_tabs = QTabWidget()
        
        # Tree view
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["Name", "Size", "Modified", "Type"])
        self.file_tree.itemSelectionChanged.connect(self.on_file_selected)
        self.file_tree.itemDoubleClicked.connect(self.on_file_double_clicked)
        self.file_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_tree.customContextMenuRequested.connect(self.show_context_menu)
        self.file_tabs.addTab(self.file_tree, "Tree View")
        
        # Search results
        self.search_results = QTableWidget()
        self.search_results.setColumnCount(5)
        self.search_results.setHorizontalHeaderLabels(["Name", "Path", "Size", "Modified", "Type"])
        self.search_results.horizontalHeader().setStretchLastSection(True)
        self.search_results.selectionModel().selectionChanged.connect(self.on_search_result_selected)
        self.search_results.itemDoubleClicked.connect(self.on_search_result_double_clicked)
        self.file_tabs.addTab(self.search_results, "Search Results")
        
        layout.addWidget(self.file_tabs)
        
        # Progress bar for operations
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        splitter.addWidget(file_widget)
    
    def create_details_panel(self, splitter):
        """Create the file details and preview panel."""
        details_widget = QWidget()
        layout = QVBoxLayout(details_widget)
        
        # File details
        details_group = QGroupBox("File Details")
        details_layout = QGridLayout(details_group)
        
        self.details_labels = {}
        details_fields = ["Name", "Path", "Size", "Modified", "Type", "Permissions"]
        
        for i, field in enumerate(details_fields):
            details_layout.addWidget(QLabel(f"{field}:"), i, 0)
            label = QLabel("No file selected")
            label.setWordWrap(True)
            self.details_labels[field] = label
            details_layout.addWidget(label, i, 1)
        
        layout.addWidget(details_group)
        
        # File preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(200)
        preview_layout.addWidget(self.preview_text)
        
        layout.addWidget(preview_group)
        
        # File operations
        operations_group = QGroupBox("Operations")
        operations_layout = QVBoxLayout(operations_group)
        
        op_buttons = [
            ("Open", self.open_selected_file),
            ("Edit", self.edit_selected_file),
            ("Copy Path", self.copy_selected_path),
            ("Show in Explorer", self.show_in_explorer),
            ("Properties", self.show_file_properties)
        ]
        
        for name, callback in op_buttons:
            button = QPushButton(name)
            button.clicked.connect(callback)
            operations_layout.addWidget(button)
        
        layout.addWidget(operations_group)
        
        layout.addStretch()
        splitter.addWidget(details_widget)
    
    def create_buttons(self, layout):
        """Create the bottom button panel."""
        button_layout = QHBoxLayout()
        
        if self.multiple_selection:
            self.select_button = QPushButton("Select Files")
            self.select_button.clicked.connect(self.accept_selection)
        else:
            self.select_button = QPushButton("Select File")
            self.select_button.clicked.connect(self.accept_selection)
        
        self.select_button.setEnabled(False)
        button_layout.addWidget(self.select_button)
        
        button_layout.addStretch()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
    
    def refresh_file_tree(self):
        """Refresh the file tree display."""
        self.file_tree.clear()
        
        if not self.current_path.exists():
            return
        
        try:
            # Add parent directory entry
            if self.current_path != self.current_path.parent:
                parent_item = QTreeWidgetItem([".. (Parent Directory)", "", "", "folder"])
                parent_item.setData(0, Qt.ItemDataRole.UserRole, str(self.current_path.parent))
                self.file_tree.addTopLevelItem(parent_item)
            
            # Add directories first
            dirs = [d for d in self.current_path.iterdir() if d.is_dir()]
            for dir_path in sorted(dirs):
                item = QTreeWidgetItem([
                    dir_path.name,
                    "",
                    datetime.fromtimestamp(dir_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "folder"
                ])
                item.setData(0, Qt.ItemDataRole.UserRole, str(dir_path))
                self.file_tree.addTopLevelItem(item)
            
            # Add files
            files = [f for f in self.current_path.iterdir() if f.is_file()]
            for file_path in sorted(files):
                # Apply file type filter
                if self.file_types and file_path.suffix not in self.file_types:
                    continue
                
                stat = file_path.stat()
                size = self.format_file_size(stat.st_size)
                modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                
                item = QTreeWidgetItem([
                    file_path.name,
                    size,
                    modified,
                    file_path.suffix or "file"
                ])
                item.setData(0, Qt.ItemDataRole.UserRole, str(file_path))
                self.file_tree.addTopLevelItem(item)
            
            # Resize columns
            for i in range(self.file_tree.columnCount()):
                self.file_tree.resizeColumnToContents(i)
            
            # Update path display
            self.path_edit.setText(str(self.current_path))
            self.status_bar.showMessage(f"Loaded {len(files)} files, {len(dirs)} folders")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load directory: {str(e)}")
    
    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        sizes = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(sizes) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {sizes[i]}"
    
    def on_file_selected(self):
        """Handle file selection in tree view."""
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            self.selected_files = []
            self.select_button.setEnabled(False)
            return
        
        item = selected_items[0]
        file_path = item.data(0, Qt.ItemDataRole.UserRole)
        
        if file_path:
            self.selected_files = [file_path]
            self.update_file_details(file_path)
            self.select_button.setEnabled(True)
    
    def on_file_double_clicked(self, item):
        """Handle double-click on file tree item."""
        file_path = item.data(0, Qt.ItemDataRole.UserRole)
        if file_path:
            path_obj = Path(file_path)
            if path_obj.is_dir():
                self.navigate_to(file_path)
            else:
                # Select and accept for files
                self.selected_files = [file_path]
                self.accept_selection()
    
    def update_file_details(self, file_path: str):
        """Update the file details panel."""
        try:
            path_obj = Path(file_path)
            
            if not path_obj.exists():
                return
            
            stat = path_obj.stat()
            
            self.details_labels["Name"].setText(path_obj.name)
            self.details_labels["Path"].setText(str(path_obj.parent))
            self.details_labels["Size"].setText(self.format_file_size(stat.st_size))
            self.details_labels["Modified"].setText(
                datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            )
            self.details_labels["Type"].setText(path_obj.suffix or "File")
            self.details_labels["Permissions"].setText(oct(stat.st_mode)[-3:])
            
            # Load preview for text files
            if path_obj.is_file() and path_obj.suffix in ['.py', '.yaml', '.yml', '.json', '.txt', '.md']:
                try:
                    with open(path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(1000)  # First 1KB
                        if len(content) == 1000:
                            content += "\n... (truncated)"
                        self.preview_text.setPlainText(content)
                except (IOError, OSError, UnicodeDecodeError) as preview_error:
                    logger.debug(f"Could not preview file '{file_path}': {preview_error}")
                    self.preview_text.setPlainText("Cannot preview this file")
            else:
                self.preview_text.setPlainText("No preview available")
                
        except Exception as e:
            logger.error(f"Error updating file details: {e}")
    
    def start_search(self):
        """Start file search operation."""
        search_term = self.search_edit.text().strip()
        if not search_term:
            return
        
        # Cancel any existing search
        if self.search_worker and self.search_worker.isRunning():
            self.search_worker.cancel()
            self.search_worker.wait()
        
        # Get file type filter
        type_filter = self.type_filter.currentText()
        file_types = []
        if type_filter != "All":
            type_map = {
                "Python (.py)": ['.py'],
                "YAML (.yaml)": ['.yaml', '.yml'],
                "JSON (.json)": ['.json'],
                "Text (.txt)": ['.txt', '.md']
            }
            file_types = type_map[type_filter] if type_filter in type_map else []
        
        # Start search
        self.search_worker = FileSearchWorker(str(self.current_path), search_term, file_types)
        self.search_worker.search_completed.connect(self.on_search_completed)
        self.search_worker.search_progress.connect(self.on_search_progress)
        self.search_worker.start()
        
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_bar.showMessage(f"Searching for '{search_term}'...")
        
        # Switch to search results view
        self.file_tabs.setCurrentIndex(1)
    
    def on_search_progress(self, current_file):
        """Handle search progress updates."""
        self.status_bar.showMessage(f"Searching: {Path(current_file).name}")
    
    def on_search_completed(self, results):
        """Handle search completion."""
        self.progress_bar.hide()
        self.search_results.setRowCount(len(results))
        
        for row, result in enumerate(results):
            self.search_results.setItem(row, 0, QTableWidgetItem(result['name']))
            self.search_results.setItem(row, 1, QTableWidgetItem(result['path']))
            self.search_results.setItem(row, 2, QTableWidgetItem(self.format_file_size(result['size'])))
            self.search_results.setItem(row, 3, QTableWidgetItem(result['modified'].strftime("%Y-%m-%d %H:%M")))
            self.search_results.setItem(row, 4, QTableWidgetItem(result['type']))
            
            # Store full path in first column
            self.search_results.item(row, 0).setData(Qt.ItemDataRole.UserRole, result['path'])
        
        self.search_results.resizeColumnsToContents()
        self.status_bar.showMessage(f"Found {len(results)} matching files")
    
    def navigate_to(self, path: str):
        """Navigate to the specified path."""
        path_obj = Path(path)
        if path_obj.exists() and path_obj.is_dir():
            self.current_path = path_obj
            self.refresh_file_tree()
    
    def navigate_to_path(self):
        """Navigate to the path entered in the path edit."""
        path = self.path_edit.text().strip()
        self.navigate_to(path)
    
    def go_back(self):
        """Go to the parent directory."""
        if self.current_path != self.current_path.parent:
            self.navigate_to(str(self.current_path.parent))
    
    
    def go_home(self):
        """Go to home directory."""
        self.navigate_to(str(Path.home()))
    
    def load_bookmarks(self):
        """Load bookmarks from file."""
        try:
            bookmarks_file = Path.home() / ".flowlib" / "file_browser_bookmarks.json"
            if bookmarks_file.exists():
                with open(bookmarks_file, 'r') as f:
                    self.bookmarks = json.load(f)
            
            self.refresh_bookmarks_list()
        except (IOError, OSError, json.JSONDecodeError) as bookmark_error:
            logger.debug(f"Could not load bookmarks: {bookmark_error}")
            self.bookmarks = []
    
    def save_bookmarks(self):
        """Save bookmarks to file."""
        try:
            bookmarks_file = Path.home() / ".flowlib" / "file_browser_bookmarks.json"
            bookmarks_file.parent.mkdir(exist_ok=True)
            with open(bookmarks_file, 'w') as f:
                json.dump(self.bookmarks, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save bookmarks: {e}")
    
    def refresh_bookmarks_list(self):
        """Refresh the bookmarks list display."""
        self.bookmarks_list.clear()
        for bookmark in self.bookmarks:
            item = QListWidgetItem(f"{bookmark['name']} ({bookmark['path']})")
            item.setData(Qt.ItemDataRole.UserRole, bookmark['path'])
            self.bookmarks_list.addItem(item)
    
    def add_bookmark(self):
        """Add current directory to bookmarks."""
        name = self.current_path.name or str(self.current_path)
        bookmark = {
            'name': name,
            'path': str(self.current_path)
        }
        
        if bookmark not in self.bookmarks:
            self.bookmarks.append(bookmark)
            self.save_bookmarks()
            self.refresh_bookmarks_list()
    
    def remove_bookmark(self):
        """Remove selected bookmark."""
        current_item = self.bookmarks_list.currentItem()
        if current_item:
            path = current_item.data(Qt.ItemDataRole.UserRole)
            self.bookmarks = [b for b in self.bookmarks if b['path'] != path]
            self.save_bookmarks()
            self.refresh_bookmarks_list()
    
    def navigate_to_bookmark(self, item):
        """Navigate to bookmarked location."""
        path = item.data(Qt.ItemDataRole.UserRole)
        self.navigate_to(path)
    
    def show_context_menu(self, position):
        """Show context menu for file operations."""
        item = self.file_tree.itemAt(position)
        if not item:
            return
        
        menu = QMenu()
        
        # File operations
        open_action = menu.addAction("Open")
        open_action.triggered.connect(self.open_selected_file)
        
        edit_action = menu.addAction("Edit")
        edit_action.triggered.connect(self.edit_selected_file)
        
        menu.addSeparator()
        
        copy_path_action = menu.addAction("Copy Path")
        copy_path_action.triggered.connect(self.copy_selected_path)
        
        properties_action = menu.addAction("Properties")
        properties_action.triggered.connect(self.show_file_properties)
        
        menu.exec(self.file_tree.mapToGlobal(position))
    
    def open_selected_file(self):
        """Open the selected file with default application."""
        if self.selected_files:
            os.startfile(self.selected_files[0])  # Windows
            # For cross-platform: subprocess.run(['xdg-open', file_path])  # Linux
            # For cross-platform: subprocess.run(['open', file_path])     # macOS
    
    def edit_selected_file(self):
        """Edit the selected file."""
        if self.selected_files:
            # Emit signal for parent to handle editing
            self.file_selected.emit(self.selected_files[0])
    
    def copy_selected_path(self):
        """Copy selected file path to clipboard."""
        if self.selected_files:
            from PySide6.QtGui import QClipboard
            from PySide6.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            clipboard.setText(self.selected_files[0])
            self.status_bar.showMessage("Path copied to clipboard")
    
    def show_in_explorer(self):
        """Show selected file in system file explorer."""
        if self.selected_files:
            path = Path(self.selected_files[0])
            if path.is_file():
                path = path.parent
            os.startfile(str(path))  # Windows
    
    def show_file_properties(self):
        """Show detailed file properties."""
        if self.selected_files:
            # Could implement a detailed properties dialog
            path_obj = Path(self.selected_files[0])
            stat = path_obj.stat()
            
            info = f"""File Properties:
Name: {path_obj.name}
Path: {path_obj.parent}
Size: {self.format_file_size(stat.st_size)}
Created: {datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")}
Modified: {datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")}
Permissions: {oct(stat.st_mode)[-3:]}"""
            
            QMessageBox.information(self, "File Properties", info)
    
    def on_search_result_selected(self):
        """Handle selection in search results."""
        selected_rows = set()
        for item in self.search_results.selectedItems():
            selected_rows.add(item.row())
        
        if selected_rows:
            row = list(selected_rows)[0]
            file_path = self.search_results.item(row, 0).data(Qt.ItemDataRole.UserRole)
            self.selected_files = [file_path]
            self.update_file_details(file_path)
            self.select_button.setEnabled(True)
    
    def on_search_result_double_clicked(self, item):
        """Handle double-click in search results."""
        file_path = self.search_results.item(item.row(), 0).data(Qt.ItemDataRole.UserRole)
        self.selected_files = [file_path]
        self.accept_selection()
    
    def switch_view_mode(self, mode):
        """Switch between different view modes."""
        if mode == "Tree View":
            self.file_tabs.setCurrentIndex(0)
        elif mode == "Search Results":
            self.file_tabs.setCurrentIndex(1)
    
    def on_filter_changed(self):
        """Handle file type filter changes."""
        self.refresh_file_tree()
    
    def accept_selection(self):
        """Accept the current file selection."""
        if not self.selected_files:
            return
        
        if self.multiple_selection:
            self.files_selected.emit(self.selected_files)
        else:
            self.file_selected.emit(self.selected_files[0])
        
        self.accept()