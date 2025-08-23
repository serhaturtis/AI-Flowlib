from PySide6.QtWidgets import QCheckBox
from PySide6.QtCore import Qt
from PySide6.QtCore import QPropertyAnimation
from PySide6.QtCore import Property
from PySide6.QtCore import Slot
from PySide6.QtCore import QPoint
from PySide6.QtCore import QEasingCurve
from PySide6.QtCore import QRect
from PySide6.QtGui import QPainter
from PySide6.QtGui import QFont
from PySide6.QtGui import QColor

#HEH, KZL, NVB, AN1

class ToggleSwitch(QCheckBox):
    def __init__(
        self,
        name,
        bg_color="#777",
        circle_color="#DDD",
        active_color="#00BCFF",
        animation_curve=QEasingCurve.Type.OutBounce,
        is_three_state=False,
        is_vertical=False,
    ):
        super().__init__()
        self.setText(name)
        self.is_vertical = is_vertical
        if is_vertical:
            self.setFixedWidth(28)
        else:
            self.setFixedHeight(28)      

        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # COLORS
        self._bg_color = bg_color
        self._circle_color = circle_color
        self._active_color = active_color

        self.is_three_state = is_three_state
        self._position = 3 if not is_vertical else 3
        self.animation = QPropertyAnimation(self, b"position")
        self.animation.setEasingCurve(animation_curve)
        self.animation.setDuration(500)
        self.stateChanged.connect(self.setup_animation)

        if is_three_state:
            self.setTristate(True)
        else:
            self.setTristate(False)

    @Property(float)
    def position(self):
        return self._position

    @position.setter
    def position(self, pos):
        self._position = pos
        self.update()

    @Slot()
    def setup_animation(self, value):
        self.animation.stop()
        if self.is_three_state:
            if self.checkState() == Qt.CheckState.Checked:
                end_value = self.height() - 26 if self.is_vertical else self.width() - 26
            elif self.checkState() == Qt.CheckState.PartiallyChecked:
                end_value = (self.height() - 26) / 2 if self.is_vertical else (self.width() - 26) / 2
            else:
                end_value = 4
        else:
            end_value = self.height() - 26 if value and self.is_vertical else self.width() - 26 if value else 4

        self.animation.setEndValue(end_value)
        self.animation.start()

    def hitButton(self, pos: QPoint):
        return self.contentsRect().contains(pos)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(QFont("Segoe UI", 9))

        p.setPen(Qt.PenStyle.NoPen)

        rect = QRect(0, 0, self.width(), self.height())

        if not self.isChecked() and not self.checkState() == Qt.CheckState.PartiallyChecked:
            p.setBrush(QColor(self._bg_color))
            if self.is_vertical:
                p.drawRoundedRect(0, 0, 28, rect.height(), 14, 14)
                p.setBrush(QColor(self._circle_color))
                p.drawEllipse(3, self._position, 22, 22)
            else:
                p.drawRoundedRect(0, 0, rect.width(), 28, 14, 14)
                p.setBrush(QColor(self._circle_color))
                p.drawEllipse(self._position, 3, 22, 22)
        else:
            p.setBrush(QColor(self._active_color))
            if self.is_vertical:
                p.drawRoundedRect(0, 0, 28, rect.height(), 14, 14)
                p.setBrush(QColor(self._circle_color))
                p.drawEllipse(3, self._position, 22, 22)
            else:
                p.drawRoundedRect(0, 0, rect.width(), 28, 14, 14)
                p.setBrush(QColor(self._circle_color))
                p.drawEllipse(self._position, 3, 22, 22)

        p.end()
