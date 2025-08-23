from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QSplitter
from PySide6.QtCore import Qt

from .navigation_panel import NavigationPanel
from .pages_panel import PagesPanel

from flowlib.gui.ui.pages.provider_repository.page import ProviderRepositoryPage
from flowlib.gui.ui.pages.configuration_manager.page import ConfigurationManagerPage
from flowlib.gui.ui.pages.preset_manager.page import PresetManagerPage
from flowlib.gui.ui.pages.knowledge_plugin_manager.page import KnowledgePluginManagerPage
from flowlib.gui.ui.pages.import_export.page import ImportExportPage
from flowlib.gui.ui.pages.testing_tools.page import TestingToolsPage

from flowlib.gui.ui.generic.widgets.simple_console import Console


class MainContainerPanel(QWidget):

    def __init__(self):
        super().__init__()

        self.pages = []
        self.navigation_panel = None
        self.pages_panel = None

        self.main_content_layout = None

        self.init_ui()
        self.init_logic()

    def init_ui(self):
        self.navigation_panel = NavigationPanel(self.change_page)
        nav_size_policy = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
        self.navigation_panel.setSizePolicy(nav_size_policy)

        # Create the pages panel (takes main content area)
        self.pages_panel = PagesPanel()
        pages_size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        pages_size_policy.setHorizontalStretch(1)
        pages_size_policy.setVerticalStretch(1)
        self.pages_panel.setSizePolicy(pages_size_policy)

        self.main_content_layout = QHBoxLayout()
        self.main_content_layout.addWidget(self.navigation_panel)
        self.main_content_layout.addWidget(self.pages_panel)

        self.main_content_layout.setContentsMargins(0, 0, 0, 0)
        self.main_content_layout.setSpacing(0)

        self.setLayout(self.main_content_layout)

        # Add configuration pages
        self.pages.append(ProviderRepositoryPage())
        self.pages.append(ConfigurationManagerPage())
        self.pages.append(PresetManagerPage())
        self.pages.append(KnowledgePluginManagerPage())
        self.pages.append(ImportExportPage())
        self.pages.append(TestingToolsPage())

    def init_logic(self):
        self.navigation_panel.set_pages(self.pages)
        self.pages_panel.set_pages(self.pages)
        self.navigation_panel.on_clicked(0)

    def change_page(self, index):
        self.pages_panel.show_page(index)

    def get_pages(self):
        return self.pages