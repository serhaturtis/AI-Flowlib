from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import QVBoxLayout

from flowlib.gui.ui.generic.widgets.menu_bar import MenuBar
from flowlib.gui.ui.panels.main_container_panel import MainContainerPanel
from flowlib.gui.ui.panels.bottom_bar import BottomBar
from flowlib.gui.ui.generic.widgets.enhanced_console import EnhancedConsole

class IDEPanel(QWidget):

    def __init__(self, parent = None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.ide_menu_bar = MenuBar(self.parent())

        self.ide_main_container_panel = MainContainerPanel()
        self.ide_main_container_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ide_main_container_policy.setHorizontalStretch(1)
        self.ide_main_container_policy.setVerticalStretch(1)
        self.ide_main_container_panel.setSizePolicy(self.ide_main_container_policy)
        
        # Set the main container panel reference in the menu bar
        self.ide_menu_bar.set_main_container_panel(self.ide_main_container_panel)

        # Add enhanced console between main container and bottom bar
        self.console = EnhancedConsole()
        self.console.setFixedHeight(150)  # Slightly taller for enhanced features
        console_policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.console.setSizePolicy(console_policy)

        self.ide_bottom_bar = BottomBar()
        self.ide_bottom_bar.setFixedHeight(20)
        self.ide_bottom_policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.ide_bottom_policy.setHorizontalStretch(0)
        self.ide_bottom_bar.setSizePolicy(self.ide_bottom_policy)

        self.ide_layout = QVBoxLayout()
        self.ide_layout.setContentsMargins(0, 0, 0, 0)
        self.ide_layout.setSpacing(0)

        self.ide_layout.addWidget(self.ide_menu_bar)
        self.ide_layout.addWidget(self.ide_main_container_panel)
        self.ide_layout.addWidget(self.console)
        self.ide_layout.addWidget(self.ide_bottom_bar)

        self.setLayout(self.ide_layout)

    def get_state(self):
        pages_list = self.ide_main_container_panel.get_pages()
        state_data = {}

        for page in pages_list:
            try:
                state_data[page.get_title()] = page.get_state() if hasattr(page, 'get_state') else {}
            except Exception as state_error:
                logger.warning(f"Could not get state for page '{page.get_title()}': {state_error}")
                state_data[page.get_title()] = {}

        return state_data


    def set_state(self, state_data):
        pages_list = self.ide_main_container_panel.get_pages()

        for page in pages_list:
            if page.get_title() in state_data and hasattr(page, 'set_state'):
                try:
                    page.set_state(state_data[page.get_title()])
                except Exception as restore_error:
                    logger.warning(f"Could not restore state for page '{page.get_title()}': {restore_error}")