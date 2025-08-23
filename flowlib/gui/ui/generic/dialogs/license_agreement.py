from datetime import datetime

from PySide6.QtWidgets import QDialog
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QTextEdit
from PySide6.QtWidgets import QPushButton

from PySide6.QtCore import Slot

from flowlib.gui import config


class LicenseAgreement(QDialog):
    def __init__(self):
        super().__init__()
        self.resize(600, 800)
        self.setWindowTitle("License Agreement")

        layout = QVBoxLayout()

        try:
            with open(config.NOTICE_FILE, "r") as file:
                text = file.read()
        except FileNotFoundError:
            text = "Notice file not found."
        except IOError:
            text = "Error reading the notice file."

        text_edit = QTextEdit()
        text_edit.setPlainText(text)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)

        button_layout = QHBoxLayout()
        agree_button = QPushButton("I Agree")
        agree_button.clicked.connect(self.agree_clicked)
        decline_button = QPushButton("Decline")
        decline_button.clicked.connect(self.decline_clicked)
        button_layout.addWidget(decline_button)
        button_layout.addWidget(agree_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    @Slot()
    def agree_clicked(self):
        try:
            with open(config.LCS_FILE, "w") as file:
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                file.write("License Agreement: [Object Detection Toolkit] Accepted\r\n")
                file.write(dt_string + "\r\n")
            self.accept()
        except IOError:
            pass

    @Slot()
    def decline_clicked(self):
        self.reject()