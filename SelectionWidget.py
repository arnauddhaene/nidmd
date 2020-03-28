# This Python file uses the following encoding: utf-8
from PyQt5 import QtCore
from PyQt5 import QtWidgets


class SelectionWidget(QtWidgets.QWidget):
    """
    Class describing the Selection Widget. Useful for selecting
    files and resetting the program.
    """
    def __init__(self, centralWidget):
        """
        Constructor
        """
        QtWidgets.QWidget.__init__(self, centralWidget)
        
        self.setGeometry(QtCore.QRect(15, 15, 600, 100))
        self.setObjectName("selectionWidget")
        
        self.selectionWidgetVerticalLayout = QtWidgets.QVBoxLayout(self)
        self.selectionWidgetVerticalLayout.setContentsMargins(0, 0, 0, 0)
        self.selectionWidgetVerticalLayout.setObjectName("selectionWidgetVerticalLayout")

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        self.buttonsVerticalLayout = QtWidgets.QVBoxLayout()
        self.buttonsVerticalLayout.setObjectName("buttonsVerticalLayout")

        # Choose File Button
        self.chooseFileButton = QtWidgets.QPushButton(self)
        self.chooseFileButton.setText("Choose File")
        self.chooseFileButton.setObjectName("chooseFileButton")

        self.buttonsVerticalLayout.addWidget(self.chooseFileButton)

        # Clear Selection Button
        self.resetButton = QtWidgets.QPushButton(self)
        self.resetButton.setText("Reset")
        self.resetButton.setObjectName("resetButton")

        self.buttonsVerticalLayout.addWidget(self.resetButton)
        
        self.horizontalLayout.addLayout(self.buttonsVerticalLayout)

        # Selected files list
        self.filesList = QtWidgets.QListWidget(self)
        self.filesList.setProperty("isWrapping", True)
        self.filesList.setResizeMode(QtWidgets.QListView.Adjust)
        self.filesList.setBatchSize(3)
        self.filesList.setObjectName("filesList")
        
        self.horizontalLayout.addWidget(self.filesList)
        
        self.selectionWidgetVerticalLayout.addLayout(self.horizontalLayout)
