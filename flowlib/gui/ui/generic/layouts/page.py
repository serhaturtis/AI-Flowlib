from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QTabWidget

from PySide6.QtCore import Slot

class Page(QWidget):
    def __init__(self):
        super().__init__()
        self.title = None
        self.tabs = []

        self.tab_widget = QTabWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def register_tab(self, tab):
        if tab.title:
            self.tabs.append(tab)
            self.tab_widget.addTab(tab, tab.title)
        else:
            raise AttributeError("Registered widget must have a 'title' property")
        
    def get_tabs(self):
        return self.tabs
        
    def get_title(self):
        return self.title

    def init_ui(self):
        raise NotImplementedError

    def init_logic(self):
        raise NotImplementedError

    def deinit_logic(self):
        raise NotImplementedError
    
    @Slot(int)
    def on_tab_changed(self, index):
        for i in range(self.tab_widget.count()):
            if i != index:
                self.tabs[i].tab_invisible()
        
        self.tabs[index].tab_visible()

    def show_tab(self, index):
        self.tab_widget.setCurrentIndex(index)

    def page_visible(self):
        self.init_logic()
        self.show_tab(0)

    def page_invisible(self):
        for i in range(self.tab_widget.count()):
            self.tabs[i].tab_invisible()
        self.deinit_logic()

    def update_ui(self):
        raise NotImplementedError
    
    def get_state(self):
        state_data = {}

        for tab in self.tabs:
            state_data[tab.get_title()] = tab.get_state()

        return state_data
    
    def set_state(self, state_data):
        for tab in self.tabs:
            tab.set_state(state_data[tab.get_title()])