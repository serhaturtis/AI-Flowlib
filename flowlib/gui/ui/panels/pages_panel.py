from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QStackedWidget
from PySide6.QtWidgets import QHBoxLayout


class PagesPanel(QWidget):

    def __init__(self):
        super().__init__()
        self.pages = []
        self.stack = None
        self.main_layout = None
        self.init_ui()
        self.init_logic()

    def init_ui(self):
        self.main_layout = QHBoxLayout()
        self.stack = QStackedWidget()
        self.main_layout.addWidget(self.stack)

        self.main_layout.setContentsMargins(4, 4, 4, 4)
        self.main_layout.setSpacing(0)

        self.setLayout(self.main_layout)

    def init_logic(self):
        pass

    def set_pages(self, pages):
        self.pages = pages
        for page in self.pages:
            self.stack.addWidget(page)

    def show_page(self, index):
        self.stack.setCurrentIndex(index)
        self.pages[index].page_visible()
