import logging

from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QDial

from PySide6.QtCore import Slot

class Knob(QWidget):
    
    def __init__(self, min, max, change_callback=None):
        super().__init__()
        self.callback = change_callback
        layout = QVBoxLayout(self)
        
        self.dial = QDial()
        self.dial.setMinimum(min)
        self.dial.setMaximum(max)
        if self.callback:
            self.dial.valueChanged.connect(self.on_value_changed)
        
        layout.addWidget(self.dial)
        self.setLayout(layout)

    @Slot()
    def on_value_changed(self, value):
        logging.info(f"Dial value: {value}")
        self.callback(value)