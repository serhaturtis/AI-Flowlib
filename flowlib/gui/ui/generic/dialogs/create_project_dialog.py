from PySide6.QtWidgets import QDialog
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QGridLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QPushButton

from PySide6.QtCore import Slot


class CreateProjectDialog(QDialog):

    def __init__(self, parent=None):
        super(CreateProjectDialog, self).__init__(parent)
        self.setWindowTitle("Create New Project")
        self.setFixedWidth(800)
        self.init_ui()
        

    def init_ui(self):
        self.name_edit = QLineEdit(self)
        
        self.create_button = QPushButton("Create")
        self.cancel_button = QPushButton("Cancel")
        
        self.create_button.clicked.connect(self.create)
        self.cancel_button.clicked.connect(self.cancel)
        
        layout = QGridLayout()
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 3)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)

        layout.addWidget(QLabel("Project Name:"), 0, 0)
        layout.addWidget(self.name_edit, 0, 1)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.create_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout, 1, 0, 1, 2)
        self.setLayout(layout)
        self.adjustSize()
        self.setFixedHeight(self.geometry().height())
    
    @Slot()
    def create(self):
        self.accept()

    @Slot()
    def cancel(self):
        self.reject()

    def get_value(self):
        return self.name_edit.text()