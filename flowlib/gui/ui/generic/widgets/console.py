from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QTextEdit

from logic.generic import LogManager


class Console(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.textedit = QTextEdit()
        self.textedit.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(0)
        layout.addWidget(self.textedit)
        self.setLayout(layout)

        self.log_manager = LogManager.get_instance()
        self.log_manager.log_message.connect(self.append_message)

    def append_message(self, message):
        self.textedit.append(message)
