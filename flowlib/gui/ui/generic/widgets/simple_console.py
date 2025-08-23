from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PySide6.QtCore import Qt

class Console(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setMaximumHeight(100)
        self.console_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
                border: 1px solid #3c3c3c;
            }
        """)
        
        layout.addWidget(self.console_text)
        self.setLayout(layout)
        
        # Add welcome message
        self.log("Flowlib Configuration Manager Console")
        self.log("Ready for configuration management operations...")
    
    def log(self, message):
        """Add a message to the console"""
        self.console_text.append(message)
    
    def clear(self):
        """Clear the console"""
        self.console_text.clear()