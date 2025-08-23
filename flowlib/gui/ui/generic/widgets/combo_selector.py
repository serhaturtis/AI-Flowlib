from PySide6.QtWidgets import QGroupBox
from PySide6.QtWidgets import QComboBox
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QSizePolicy


class ComboSelector(QGroupBox):

    def __init__(self, title="Select", items=[]):
        super().__init__(title)
        self.combobox = None
        self.setStyleSheet("border:0;")
        size_policy = QSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )

        self.combobox = QComboBox()
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        self.combobox.setSizePolicy(size_policy)

        self.combobox.addItems(items)

        # Set up layout
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.combobox)
        self.setLayout(layout)

    def set_content_list(self, items):
        self.combobox.clear()
        self.combobox.addItems(items)

    def get_current_selection(self):
        return self.combobox.currentText()
