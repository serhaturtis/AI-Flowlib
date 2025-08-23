import os
import time

from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QListWidget
from PySide6.QtWidgets import QSpacerItem
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import QListWidgetItem
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QGroupBox
from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QFileDialog

from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from flowlib.gui.logic.generic.application.settings_controller import SettingsController
from flowlib.gui.logic.generic.application.project_controller import ProjectController

from flowlib.gui.ui.generic.widgets import IconButton
from flowlib.gui.ui.generic.dialogs import CreateProjectDialog

from flowlib.gui import config


class ProjectSelectionPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings_controller = SettingsController.get_instance()
        self.project_controller = ProjectController.get_instance()

        self.init_ui()

        if not self.settings_controller.workspace_path:
            self.change_workspace()
        else:
            self.set_workspace_projects_list()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
    
        content_widget = QWidget(self)
        content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
    
        # GroupBox for the project list
        list_groupbox = QGroupBox("Projects in Workspace:")
        list_layout = QVBoxLayout()
    
        self.projects_list = QListWidget(self)
        list_layout.addWidget(self.projects_list)
    
        # Move the "Open Project" button below the list
        self.open_project_button = IconButton(
            "Open Project", config.PROJECT_OPEN_ICON_PATH
        )
        self.open_project_button.clicked.connect(self.open_selected_project)
    
        list_layout.addWidget(self.open_project_button)  # Add the button to the list layout
    
        list_groupbox.setLayout(list_layout)
        content_layout.addWidget(list_groupbox, 2)
    
        # Create the rest of the buttons in a separate layout
        buttons_layout = QVBoxLayout()
    
        self.create_new_button = IconButton(
            "Create New Project", config.PROJECT_NEW_ICON_PATH
        )
        self.remove_project_button = IconButton(
            "Delete Project", config.PROJECT_REMOVE_ICON_PATH
        )
        self.change_workspace_button = IconButton(
            "Change Workspace", config.WORKSPACE_ICON_PATH
        )
    
        self.create_new_button.clicked.connect(self.create_new_project)
        self.remove_project_button.clicked.connect(
            self.remove_selected_project_from_workspace
        )
        self.change_workspace_button.clicked.connect(self.change_workspace)
    
        buttons_layout.addWidget(self.create_new_button)
        buttons_layout.addWidget(self.remove_project_button)
        buttons_layout.addWidget(self.change_workspace_button)
    
        spacer = QSpacerItem(
            20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        buttons_layout.addItem(spacer)
    
        content_layout.addLayout(buttons_layout, 1)
    
        main_layout.addWidget(content_widget)
    
        self.setLayout(main_layout)

    def set_workspace_projects_list(self):
        self.projects_list.clear()
        workspace_path = self.settings_controller.workspace_path
        if workspace_path and os.path.exists(workspace_path):
            for project_name in os.listdir(workspace_path):
                project_path = os.path.join(workspace_path, project_name)
                if os.path.isdir(project_path):
                    bsp_file = os.path.join(project_path, project_name + ".bsp")
                    if os.path.exists(bsp_file):
                        item = QListWidgetItem(project_name, self.projects_list)
                        font = QFont()
                        font.setBold(True)
                        item.setFont(font)
                        item.setForeground(Qt.GlobalColor.white)
                        item.setToolTip(project_path)
                        item.setData(Qt.ItemDataRole.UserRole, project_path)
        else:
            QMessageBox.warning(
                self, "Workspace Error", "Please set a valid workspace path."
            )

    def create_new_project(self):
        dialog = CreateProjectDialog()
        if dialog.exec():
            name = dialog.get_value()
            if name == "":
                QMessageBox.warning(self, "Error", "Enter a valid project name.")
            else:
                workspace_path = self.settings_controller.workspace_path
                if workspace_path and os.path.exists(workspace_path):
                    self.project_controller.create_project(name, workspace_path)
                    self.set_workspace_projects_list()
                    self.open_project(name, os.path.join(workspace_path, name))
                else:
                    QMessageBox.warning(
                        self, "Workspace Error", "Please set a valid workspace path."
                    )

    def open_selected_project(self):
        selected_item = self.projects_list.currentItem()
        if selected_item:
            name = selected_item.text()
            project_path = selected_item.data(Qt.ItemDataRole.UserRole)
            self.open_project(name, project_path)
        else:
            QMessageBox.warning(
                self,
                "Open Project",
                "Please select a project to open.",
            )

    def remove_selected_project_from_workspace(self):
        selected_item = self.projects_list.currentItem()
        if selected_item:
            name = selected_item.text()
            project_path = selected_item.data(Qt.ItemDataRole.UserRole)
            confirm = QMessageBox.question(
                self,
                "Delete Project",
                f"Are you sure you want to delete project '{name}'?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if confirm == QMessageBox.Yes:
                import shutil

                shutil.rmtree(project_path)
                self.set_workspace_projects_list()
        else:
            QMessageBox.warning(
                self,
                "Delete Project",
                "Please select a project to delete.",
            )

    def change_workspace(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Workspace Directory")
        if dir_path:
            self.settings_controller.workspace_path = dir_path
            self.settings_controller.save_settings()
            self.set_workspace_projects_list()

    def open_project(self, name, path):
        self.project_controller.open_project(name, path)

    def closeEvent(self, event):
        super().closeEvent(event)
