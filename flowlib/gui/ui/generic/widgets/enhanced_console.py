from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
                               QComboBox, QPushButton, QLabel, QCheckBox,
                               QLineEdit, QMenu, QToolBar, QApplication)
from PySide6.QtCore import Qt, Signal, QObject, QDateTime, QTimer
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor, QAction, QKeySequence
import logging
import re
from datetime import datetime
from collections import deque
from typing import Optional


class LogEmitter(QObject):
    """Emits log messages as Qt signals"""
    log_message = Signal(str, int)  # message, level


class ConsoleLogHandler(logging.Handler):
    """Custom logging handler that emits to Qt console"""
    
    def __init__(self, emitter: LogEmitter):
        super().__init__()
        self.emitter = emitter
        
    def emit(self, record):
        try:
            msg = self.format(record)
            self.emitter.log_message.emit(msg, record.levelno)
        except Exception:
            self.handleError(record)


class EnhancedConsole(QWidget):
    """Enhanced console widget with logging integration and filtering"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.max_lines = 1000
        self.log_history = deque(maxlen=10000)  # Store more history than displayed
        self.current_filter = ""
        self.log_levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        self.level_colors = {
            logging.DEBUG: QColor('#808080'),      # Gray
            logging.INFO: QColor('#FFFFFF'),       # White
            logging.WARNING: QColor('#FFA500'),    # Orange
            logging.ERROR: QColor('#FF4444'),      # Red
            logging.CRITICAL: QColor('#FF00FF')    # Magenta
        }
        
        self.init_ui()
        self.setup_logging()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(5, 2, 5, 2)
        layout.addLayout(toolbar_layout)
        
        # Log level filter
        toolbar_layout.addWidget(QLabel("Level:"))
        self.level_combo = QComboBox()
        self.level_combo.addItems(['ALL', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
        self.level_combo.setCurrentText('INFO')
        self.level_combo.currentTextChanged.connect(self.filter_logs)
        self.level_combo.setFixedWidth(100)
        toolbar_layout.addWidget(self.level_combo)
        
        # Search filter
        toolbar_layout.addWidget(QLabel("Filter:"))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Enter text to filter...")
        self.filter_input.textChanged.connect(self.filter_logs)
        self.filter_input.setClearButtonEnabled(True)
        toolbar_layout.addWidget(self.filter_input)
        
        # Auto-scroll checkbox
        self.auto_scroll_check = QCheckBox("Auto-scroll")
        self.auto_scroll_check.setChecked(True)
        toolbar_layout.addWidget(self.auto_scroll_check)
        
        # Wrap lines checkbox
        self.wrap_lines_check = QCheckBox("Wrap")
        self.wrap_lines_check.setChecked(False)
        self.wrap_lines_check.toggled.connect(self.toggle_line_wrap)
        toolbar_layout.addWidget(self.wrap_lines_check)
        
        toolbar_layout.addStretch()
        
        # Action buttons
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_console)
        toolbar_layout.addWidget(self.clear_button)
        
        self.copy_button = QPushButton("Copy All")
        self.copy_button.clicked.connect(self.copy_all)
        toolbar_layout.addWidget(self.copy_button)
        
        self.save_button = QPushButton("Save Log")
        self.save_button.clicked.connect(self.save_log)
        toolbar_layout.addWidget(self.save_button)
        
        # Console text area
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #3c3c3c;
                border-top: none;
            }
            QTextEdit::selection {
                background-color: #264f78;
            }
        """)
        
        # Set up context menu
        self.setup_context_menu()
        
        layout.addWidget(self.console_text)
        
        # Status bar
        self.status_label = QLabel("0 messages")
        self.status_label.setStyleSheet("padding: 2px; font-size: 10px; color: #888;")
        layout.addWidget(self.status_label)
        
    def setup_context_menu(self):
        """Set up right-click context menu"""
        self.console_text.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.console_text.customContextMenuRequested.connect(self.show_context_menu)
        
    def show_context_menu(self, position):
        """Show context menu at cursor position"""
        menu = QMenu(self)
        
        # Copy action
        copy_action = QAction("Copy", self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(self.console_text.copy)
        copy_action.setEnabled(self.console_text.textCursor().hasSelection())
        menu.addAction(copy_action)
        
        # Select All action
        select_all_action = QAction("Select All", self)
        select_all_action.setShortcut(QKeySequence.StandardKey.SelectAll)
        select_all_action.triggered.connect(self.console_text.selectAll)
        menu.addAction(select_all_action)
        
        menu.addSeparator()
        
        # Clear action
        clear_action = QAction("Clear Console", self)
        clear_action.triggered.connect(self.clear_console)
        menu.addAction(clear_action)
        
        # Find action
        find_action = QAction("Find...", self)
        find_action.setShortcut(QKeySequence.StandardKey.Find)
        find_action.triggered.connect(lambda: self.filter_input.setFocus())
        menu.addAction(find_action)
        
        menu.exec(self.console_text.mapToGlobal(position))
        
    def setup_logging(self):
        """Set up logging handler"""
        # Create log emitter and handler
        self.log_emitter = LogEmitter()
        self.log_handler = ConsoleLogHandler(self.log_emitter)
        
        # Set up formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.log_handler.setFormatter(formatter)
        self.log_handler.setLevel(logging.DEBUG)
        
        # Connect signal
        self.log_emitter.log_message.connect(self.append_log)
        
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        
        # Also capture flowlib logs specifically
        flowlib_logger = logging.getLogger('flowlib')
        if not any(isinstance(h, ConsoleLogHandler) for h in flowlib_logger.handlers):
            flowlib_logger.addHandler(self.log_handler)
            flowlib_logger.setLevel(logging.DEBUG)
        
        # Initial message
        logging.info("Console logging initialized")
        
    def append_log(self, message: str, level: int):
        """Append a log message to the console"""
        # Store in history
        log_entry = {
            'timestamp': datetime.now(),
            'message': message,
            'level': level
        }
        self.log_history.append(log_entry)
        
        # Check if should display based on current filter
        if self.should_display_log(message, level):
            self.display_log_entry(message, level)
            
        # Update status
        self.update_status()
        
    def should_display_log(self, message: str, level: int) -> bool:
        """Check if log should be displayed based on filters"""
        # Level filter
        selected_level = self.level_combo.currentText()
        if selected_level != 'ALL':
            min_level = self.log_levels[selected_level] if selected_level in self.log_levels else logging.INFO
            if level < min_level:
                return False
                
        # Text filter
        if self.current_filter and self.current_filter.lower() not in message.lower():
            return False
            
        return True
        
    def display_log_entry(self, message: str, level: int):
        """Display a single log entry with color coding"""
        # Move cursor to end
        cursor = self.console_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.console_text.setTextCursor(cursor)
        
        # Set color based on level
        color = self.level_colors[level] if level in self.level_colors else QColor('#FFFFFF')
        format = QTextCharFormat()
        format.setForeground(color)
        
        # Insert text with color
        cursor.insertText(message + '\n', format)
        
        # Limit lines
        if self.console_text.document().lineCount() > self.max_lines:
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            cursor.movePosition(QTextCursor.MoveOperation.Down, QTextCursor.MoveMode.KeepAnchor, 
                              self.console_text.document().lineCount() - self.max_lines)
            cursor.removeSelectedText()
        
        # Auto scroll if enabled
        if self.auto_scroll_check.isChecked():
            scrollbar = self.console_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
    def filter_logs(self):
        """Re-filter and display logs based on current filters"""
        self.current_filter = self.filter_input.text()
        
        # Clear console
        self.console_text.clear()
        
        # Re-display filtered logs from history
        displayed_count = 0
        for entry in self.log_history:
            if displayed_count >= self.max_lines:
                break
                
            if self.should_display_log(entry['message'], entry['level']):
                self.display_log_entry(entry['message'], entry['level'])
                displayed_count += 1
                
        self.update_status()
        
    def toggle_line_wrap(self, checked: bool):
        """Toggle line wrapping"""
        if checked:
            self.console_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        else:
            self.console_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
            
    def clear_console(self):
        """Clear the console display (keeps history)"""
        self.console_text.clear()
        self.update_status()
        logging.info("Console cleared")
        
    def copy_all(self):
        """Copy all console text to clipboard"""
        text = self.console_text.toPlainText()
        QApplication.clipboard().setText(text)
        logging.info(f"Copied {len(text)} characters to clipboard")
        
    def save_log(self):
        """Save log to file"""
        from PySide6.QtWidgets import QFileDialog
        from pathlib import Path
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Log File",
            str(Path.home() / f"flowlib_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"),
            "Text Files (*.txt);;Log Files (*.log);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    # Write all history, not just visible
                    for entry in self.log_history:
                        f.write(f"{entry['message']}\n")
                logging.info(f"Log saved to: {file_path}")
            except Exception as e:
                logging.error(f"Failed to save log: {e}")
                
    def update_status(self):
        """Update status label"""
        visible_lines = self.console_text.document().lineCount() - 1  # Subtract 1 for accuracy
        total_logs = len(self.log_history)
        
        if self.current_filter or self.level_combo.currentText() != 'ALL':
            filtered_count = sum(1 for entry in self.log_history 
                               if self.should_display_log(entry['message'], entry['level']))
            status = f"Showing {visible_lines} of {filtered_count} filtered messages ({total_logs} total)"
        else:
            status = f"{visible_lines} messages ({total_logs} in history)"
            
        self.status_label.setText(status)
        
    def log(self, message: str, level: int = logging.INFO):
        """Public method to log a message directly"""
        self.append_log(message, level)
        
    def clear(self):
        """Clear console and history"""
        self.console_text.clear()
        self.log_history.clear()
        self.update_status()
        
    def set_log_level(self, level: str):
        """Set the minimum log level to display"""
        if level in self.log_levels:
            self.level_combo.setCurrentText(level)
            
    def closeEvent(self, event):
        """Clean up logging handler on close"""
        # Remove our handler from loggers
        root_logger = logging.getLogger()
        root_logger.removeHandler(self.log_handler)
        
        flowlib_logger = logging.getLogger('flowlib')
        if self.log_handler in flowlib_logger.handlers:
            flowlib_logger.removeHandler(self.log_handler)
            
        super().closeEvent(event)