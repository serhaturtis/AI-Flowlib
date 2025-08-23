from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QSizePolicy

from PySide6.QtCore import Qt

from flowlib.gui import config


class BottomBar(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_logic()

    def init_ui(self):
        self.bar_layout = QHBoxLayout()
        self.bar_layout.setContentsMargins(0,0,0,0)
        self.bar_layout.setSpacing(0)

        bottom_text = QLabel(f"Copyright © 2022-2025 {config.DEV_NAME} - {config.BUILD_TYPE} - v{config.BUILD_VERSION}")
        size_policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        bottom_text.setSizePolicy(size_policy)

        self.bar_layout.addWidget(bottom_text, Qt.AlignmentFlag.AlignCenter, Qt.AlignmentFlag.AlignCenter)

        self.setLayout(self.bar_layout)

    def init_logic(self):
        pass
