from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QGroupBox
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtCore import Slot


class FolderBrowser(QWidget):

    def __init__(self, title='Folder', groupbox=True):
        super().__init__()
        self.last_browsed_folder = ''
        size_policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        self.line_edit = QLineEdit()
        self.line_edit.setReadOnly(True)
        self.line_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        size_policy.setHorizontalStretch(4)
        size_policy.setVerticalStretch(0)
        self.line_edit.setSizePolicy(size_policy)

        self.button = QPushButton('Browse')
        size_policy.setHorizontalStretch(1)
        size_policy.setVerticalStretch(0)
        self.button.setSizePolicy(size_policy)
        self.button.clicked.connect(self.open_folder_dialog)

        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.button)

        if groupbox:
            main_layout = QHBoxLayout()
            group_box = QGroupBox(title)
            main_layout.addWidget(group_box)
            group_box.setLayout(layout)
            self.setLayout(main_layout)
        else:
            self.setLayout(layout)

    @Slot()
    def open_folder_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder', self.last_browsed_folder)
        if folder_path:
            self.line_edit.setText(folder_path)
            self.last_browsed_folder = folder_path

    def get_value(self):
        return self.line_edit.text()
