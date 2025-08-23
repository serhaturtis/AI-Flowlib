from PySide6.QtWidgets import QGroupBox
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtCore import Slot

class FileBrowser(QGroupBox):

    def __init__(self, title='File'):
        super().__init__(title)
        self.line_edit = None
        self.button = None
        self.last_browsed_file = ''
        size_policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        self.line_edit = QLineEdit()
        self.line_edit.setReadOnly(True)
        size_policy.setHorizontalStretch(4)
        size_policy.setVerticalStretch(0)
        self.line_edit.setSizePolicy(size_policy)

        self.button = QPushButton('Browse')
        size_policy.setHorizontalStretch(1)
        size_policy.setVerticalStretch(0)
        self.button.setSizePolicy(size_policy)
        self.button.clicked.connect(self.open_file_dialog)

        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.button)
        self.setLayout(layout)

    @Slot()
    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select File', self.last_browsed_file)
        if file_path:
            self.line_edit.setText(file_path)
            self.last_browsed_file = file_path

    def get_value(self):
        return self.line_edit.text()
