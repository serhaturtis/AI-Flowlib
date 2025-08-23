from functools import partial

from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import QSpacerItem
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtCore import Qt
from PySide6.QtCore import Slot

from flowlib.gui import config


class NavigationPanel(QWidget):

    def __init__(self, change_page_callback):
        super().__init__()
        self.application_logo = None

        self.pages = None
        self.buttons = []
        self.main_layout = None
        self.change_page_callback = change_page_callback
        self.init_ui()
        self.init_logic()

    def init_ui(self):
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout()
        
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.application_logo = QSvgWidget(config.APP_LOGO_SVG_PATH)
        self.application_logo.renderer().setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)

        logo_widget = QWidget()
        logo_widget.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred))
        logo_layout = QHBoxLayout()
        logo_layout.addWidget(self.application_logo)
        logo_widget.setLayout(logo_layout)
        
        self.main_layout.addWidget(logo_widget)

        self.main_widget.setLayout(self.main_layout)
        self.main_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        self.toggle_button = QPushButton("<")
        self.toggle_button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.toggle_button.setFixedWidth(15)
        self.toggle_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.toggle_button.clicked.connect(self.toggle_main_layout)

        self.container_layout = QHBoxLayout()
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(0)
        
        self.container_layout.addWidget(self.main_widget)
        self.container_layout.addWidget(self.toggle_button)

        self.setLayout(self.container_layout)

    def init_logic(self):
        pass

    def set_pages(self, pages):
        self.pages = pages
        self.buttons = []
        group_indices = [1, 3]
        
        for i, page in enumerate(pages):
            button = QPushButton(page.get_title())
            button.setFlat(True)
            button.setCheckable(True)
            button.clicked.connect(partial(self.on_clicked, self.pages.index(page)))
            self.buttons.append(button)
            self.main_layout.addWidget(button)
            
            if i in group_indices:
                spacer = QSpacerItem(20, 20, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
                self.main_layout.addSpacerItem(spacer)
        
        self.main_layout.addStretch()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        parent_width = self.width()
        self.application_logo.setFixedSize(parent_width // 2, parent_width // 2)

    @Slot(int)
    def on_clicked(self, index):
        self.change_page_callback(index)
        for button in self.buttons:
            button.setChecked(False)

        self.buttons[index].setChecked(True)

    @Slot()
    def toggle_main_layout(self):
        if self.main_widget.isVisible():
            self.main_widget.setVisible(False)
            self.toggle_button.setText(">")
        else:
            self.main_widget.setVisible(True)
            self.toggle_button.setText("<")
