from PySide6.QtWidgets import QPushButton
from PySide6.QtGui import QIcon
from PySide6.QtCore import QSize
from PySide6.QtSvg import QSvgRenderer

class IconButton(QPushButton):
    def __init__(self, text="", icon_path=None, icon_size=QSize(24, 24), parent=None):
        super().__init__(text, parent)

        if icon_path:
            self.setIcon(QIcon(icon_path))
            self.setIconSize(icon_size)
        
        # Custom style for icon alignment
        self.setStyleSheet("""
            QPushButton {
                text-align: center;
                padding-left: 40px;  /* Adjust this to make space for the icon */
            }
            QPushButton::icon {
                position: absolute;
                left: 10px;  /* Adjust this to control the icon's distance from the left edge */
            }
        """)

    def set_icon(self, icon_path, icon_size=QSize(24, 24)):
        """Method to set or update the icon."""
        self.setIcon(QIcon(icon_path))
        self.setIconSize(icon_size)