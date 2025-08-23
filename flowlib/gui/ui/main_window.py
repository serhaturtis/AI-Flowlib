from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QVBoxLayout

from PySide6.QtWidgets import QSizeGrip
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QStackedWidget

from PySide6.QtCore import Qt
from PySide6.QtCore import QRect

from flowlib.gui.ui.generic.widgets.title_bar import TitleBar
from flowlib.gui.ui.generic.widgets.resize_grips import ResizeGrip
from flowlib.gui.ui.panels.ide_panel import IDEPanel


class MainWindow(QMainWindow):
    _grip_size = 8

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.init_ui()

    def init_ui(self):
        # Set minimum window size to prevent layout constraints
        self.setMinimumSize(800, 600)
        
        self.title_bar = TitleBar(self)

        # Create the main layout
        self.application_layout = QVBoxLayout()
        self.application_layout.setContentsMargins(0, 0, 0, 0)
        self.application_layout.setSpacing(0)
        
        # Add title bar and main content directly
        self.ide_panel = IDEPanel(self)
        self.application_layout.addWidget(self.title_bar)
        self.application_layout.addWidget(self.ide_panel)
        
        self.application_widget = QWidget()
        self.application_widget.setLayout(self.application_layout)
        self.setCentralWidget(self.application_widget)

        self.side_grips = [
            ResizeGrip(self, Qt.Edge.LeftEdge),
            ResizeGrip(self, Qt.Edge.TopEdge),
            ResizeGrip(self, Qt.Edge.RightEdge),
            ResizeGrip(self, Qt.Edge.BottomEdge),
        ]

        self.corner_grips = [QSizeGrip(self) for i in range(4)]

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Message",
            "Are you sure to quit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()

    @property
    def grip_size(self) -> int:
        return self._grip_size

    def set_grip_size(self, size):
        if size == self._grip_size:
            return
        self._grip_size = max(2, size)
        self.update_grips()

    def update_grips(self):
        out_rect = self.rect()
        in_rect = out_rect.adjusted(
            self.grip_size, self.grip_size, -self.grip_size, -self.grip_size
        )

        self.corner_grips[0].setGeometry(QRect(out_rect.topLeft(), in_rect.topLeft()))
        self.corner_grips[1].setGeometry(
            QRect(out_rect.topRight(), in_rect.topRight()).normalized()
        )
        self.corner_grips[2].setGeometry(
            QRect(in_rect.bottomRight(), out_rect.bottomRight())
        )
        self.corner_grips[3].setGeometry(
            QRect(out_rect.bottomLeft(), in_rect.bottomLeft()).normalized()
        )

        self.side_grips[0].setGeometry(
            0, in_rect.top(), self.grip_size, in_rect.height()
        )
        self.side_grips[1].setGeometry(
            in_rect.left(), 0, in_rect.width(), self.grip_size
        )
        self.side_grips[2].setGeometry(
            in_rect.left() + in_rect.width(),
            in_rect.top(),
            self.grip_size,
            in_rect.height(),
        )
        self.side_grips[3].setGeometry(
            self.grip_size,
            in_rect.top() + in_rect.height(),
            in_rect.width(),
            self.grip_size,
        )

        [grip.raise_() for grip in self.side_grips + self.corner_grips]

    def resizeEvent(self, event):
        QMainWindow.resizeEvent(self, event)
        self.update_grips()

    def get_state(self):
        return self.ide_panel.get_state()

    def set_state(self, state_data):
        self.ide_panel.set_state(state_data)