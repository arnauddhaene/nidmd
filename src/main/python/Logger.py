# This Python file uses the following encoding: utf-8
import sys
from PyQt5 import QtWidgets, QtCore
import logging

# Uncomment below for terminal log messages
# logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(name)s - %(levelname)s - %(message)s')

class QTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = QtWidgets.QPlainTextEdit(parent)
        self.widget.setReadOnly(True)

    def emit(self, record):
        message = self.format(record)
        self.widget.appendPlainText(message)


class Logger(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumSize(QtCore.QSize(600, 120))
        self.setMaximumSize(QtCore.QSize(16777215, 120))

        self.logTextBox = QTextEditLogger(self)
        
        # You can format what is printed to text box
        self.logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(self.logTextBox)
        # You can control the logging level
        logging.getLogger().setLevel(logging.DEBUG)

        self.clearButton = QtWidgets.QPushButton(self)
        self.clearButton.setText('Clear log')
        self.clearButton.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        layout = QtWidgets.QVBoxLayout()
        # Add the new logging box widget to the layout
        layout.addWidget(self.logTextBox.widget)
        layout.addWidget(self.clearButton)
        self.setLayout(layout)

        # Connect signal to slot
        self.clearButton.clicked.connect(self.clear)
        

    def clear(self):
        self.logTextBox.widget.clear()