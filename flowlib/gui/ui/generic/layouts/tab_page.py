from PySide6.QtWidgets import QWidget


class TabPage(QWidget):

    def __init__(self):
        super().__init__()
        self.title = None

    def get_title(self):
        return self.title

    def init_ui(self):
        raise NotImplementedError

    def init_logic(self):
        raise NotImplementedError

    def deinit_logic(self):
        raise NotImplementedError

    def tab_visible(self):
        raise NotImplementedError
    
    def tab_invisible(self):
        raise NotImplementedError

    def update_ui(self):
        raise NotImplementedError
    
    def get_state(self):
        raise NotImplementedError
    
    def set_state(self):
        raise NotImplementedError
