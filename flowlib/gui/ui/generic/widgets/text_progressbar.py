from PySide6.QtWidgets import QProgressBar
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtGui import QColor

class TextProgressBar(QProgressBar):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stext = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def paintEvent(self, event):
        super().paintEvent(event)
        
        painter = QPainter(self)
        
        progress_text = f"{self.stext} - {self.value()}/{self.maximum()}"
        
        painter.setPen(QColor(Qt.GlobalColor.white))
        
        rect = self.rect()
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, progress_text)
        painter.end()

    def set_text(self, text):
        self.stext = text
        self.update()