# This Python file uses the following encoding: utf-8
import sys
import os
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, QtWebEngineWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEnginePage
from PyQt5.QtWebEngineWidgets import QWebEngineView
from Decomposition import Decomposition
from BrainView import BrainView
from SelectionWidget import SelectionWidget
from MenuBar import MenuBar

class ToolboxWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setObjectName("ToolboxWindow")
        self.resize(1980, 1020)
        self.initUI()

    def initUI(self):
        # Set window title
        self.setWindowTitle("Dynamic Mode Toolbox")
        
        self.centralWidget = QtWidgets.QWidget(self)
        self.centralWidget.setObjectName("centralWidget")

        # Selection Widget for file selection and reset of program
        self.selectionWidget = SelectionWidget(self.centralWidget)
        
        # Signals
        self.selectionWidget.chooseFileButton.clicked.connect(self.chooseFile)
        self.selectionWidget.resetButton.clicked.connect(self.reset)

#        self.mainVerticalLayout = QtWidgets.QVBoxLayout()
#        self.mainVerticalLayout.setObjectName("mainVerticalLayout")
        
#        vSpacer = QtWidgets.QSpacerItem(20, 40, 
#                                           QtWidgets.QSizePolicy.Minimum, 
#                                           QtWidgets.QSizePolicy.Expanding)
#        self.mainVerticalLayout.addItem(vSpacer)

#        self.gridLayout.addLayout(self.mainVerticalLayout, 0, 0, 1, 1)
        
        self.setCentralWidget(self.centralWidget)
        
        self.menuBar = MenuBar(self)
        self.setMenuBar(self.menuBar)
        self.menuBar.addActionsToMenu();
#        self.menuBar.actionOpen.triggered.connect()
#        self.menuBar.actionSave.triggered.connect()

        # Define statusbar
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        
        QtCore.QMetaObject.connectSlotsByName(self)

    def updateSelectionWidgetFilesList(self):
        self.selectionWidget.filesList.addItems([os.path.split(name)[1] for name in self.fileNames])

    def chooseFile(self):
        self.fileNames = self.openFileNamesDialog()
        if self.fileNames:
            self.updateSelectionWidgetFilesList()
            self.decomposition = Decomposition(self.fileNames)

            # Testing with mode 0 for now
            paths = self.decomposition.saveHTML()[0]
            self.visualHorizontalLayout = BrainView(paths)
        
        # self.visualHorizontalLayout = BrainView()
#        self.mainVerticalLayout.addLayout(self.visualHorizontalLayout, 1)

    def reset(self):
        self.fileNames = ['']
        if (self.decomposition):
            self.decomposition.reset()
        self.updateSelectionWidgetFilesList()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                  "QFileDialog.getOpenFileName()",
                                                  "",
                                                  "MATLAB File (*.mat)",
                                                  options=options)
        return fileName

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()",
                                                      "",
                                                      "MATLAB Files (*.mat)",
                                                      options=options)
        return files

    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,
                                                  "QFileDialog.getSaveFileName()",
                                                  "",
                                                  "All Files (*);;Text Files (*.txt)",
                                                  options=options)
        if fileName:
            print(fileName)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ToolboxWindow()
    window.show()
    sys.exit(app.exec_())
