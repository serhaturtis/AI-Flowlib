from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QGridLayout
from PySide6.QtCore import Qt
from PySide6.QtCore import QPoint
from PySide6.QtCore import QSize
from PySide6.QtGui import QPixmap
from PySide6.QtGui import QPainter
from PySide6.QtGui import QIcon
from PySide6.QtSvg import QSvgRenderer


from flowlib.gui import config


class TitleBar(QWidget):
    def __init__(self, main_window: QMainWindow = None):
        super(TitleBar, self).__init__(main_window)
        self.main_window = main_window
        self.current_project_name = ""
        self.setFixedHeight(36)

        self.bar_layout = QGridLayout()
        self.bar_layout.setContentsMargins(4, 4, 4, 4)
        self.bar_layout.setSpacing(4)

        self.icon_label = QLabel()
        self.icon_label.setContentsMargins(4,0,0,0)
        svg_renderer = QSvgRenderer(config.APP_LOGO_SVG_PATH)
        icon_pixmap = QPixmap(24, 24)
        icon_pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(icon_pixmap)
        svg_renderer.render(painter)
        painter.end()
        
        self.icon_label.setPixmap(icon_pixmap)
        self.bar_layout.addWidget(self.icon_label, 0, 0)

        self.title_label = QLabel(f"{config.APP_NAME}")
        self.title_label.setStyleSheet("color: white; font-weight: bold;")
        self.bar_layout.addWidget(self.title_label, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        button_layout = QHBoxLayout()

        self.minimize_button = QPushButton()
        self.minimize_button.setIcon(QIcon(config.APP_MINIMIZE_ICON_PATH))
        self.minimize_button.setIconSize(QSize(12, 12))
        self.minimize_button.setFixedSize(40, 28)
        self.minimize_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.minimize_button.clicked.connect(self.main_window.showMinimized)
        button_layout.addWidget(self.minimize_button, alignment=Qt.AlignmentFlag.AlignRight)

        self.restore_button = QPushButton()
        self.restore_button.setIcon(QIcon(config.APP_RESTORE_ICON_PATH))
        self.restore_button.setIconSize(QSize(12, 12))
        self.restore_button.setFixedSize(40, 28)
        self.restore_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.restore_button.clicked.connect(self.toggle_maximize_restore)
        button_layout.addWidget(self.restore_button)

        self.close_button = QPushButton()
        self.close_button.setIcon(QIcon(config.APP_CLOSE_ICON_PATH))
        self.close_button.setIconSize(QSize(12, 12))
        self.close_button.setFixedSize(40, 28)
        self.close_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.close_button.clicked.connect(self.main_window.close)
        button_layout.addWidget(self.close_button)

        self.bar_layout.addLayout(button_layout, 0, 2)

        self.setLayout(self.bar_layout)

        self.offset = QPoint(0, 0)
        self.pressing = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.offset = (event.globalPosition().toPoint() - self.main_window.frameGeometry().topLeft())
            self.pressing = True

    def mouseMoveEvent(self, event):
        if self.pressing:
            self.main_window.move(event.globalPosition().toPoint() - self.offset)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressing = False

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_maximize_restore()

    def set_project_name(self, name):
        self.current_project_name = name
        self.title_label.setText(f"{config.APP_NAME} - {self.current_project_name}")

    def toggle_maximize_restore(self):
        if self.main_window.isMaximized():
            self.main_window.showNormal()
            # Update restore button icon to maximize icon
            self.restore_button.setIcon(QIcon(config.APP_RESTORE_ICON_PATH))
        else:
            self.main_window.showMaximized()
            # Update restore button icon to restore icon  
            self.restore_button.setIcon(QIcon(config.APP_RESTORE_ICON_PATH))
