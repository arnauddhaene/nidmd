# This Python file uses the following encoding: utf-8
import sys
import os
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QUrl
from Decomposition import Decomposition
from BrainView import BrainView
from SelectionWidget import SelectionWidget
from MenuBar import MenuBar
from Logger import Logger
from ParametersDialog import ParametersDialog
import logging

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
        self.gridLayout = QtWidgets.QGridLayout(self.centralWidget)

        # Selection Widget for file selection and reset of program
        self.selectionWidget = SelectionWidget(self.centralWidget)
        
        # Signals
        self.selectionWidget.chooseFileButton.clicked.connect(self.choose_files)
        self.selectionWidget.resetButton.clicked.connect(self.reset)
        
        self.gridLayout.addWidget(self.selectionWidget)

        self.radarLayout = QtWidgets.QHBoxLayout()
        self.gridLayout.addLayout(self.radarLayout, 1, 0)
        
        self.brainGridLayout = QtWidgets.QGridLayout()

        self.brainViews = []
        for mode in range(2):
            self.brainViews.append(BrainView(self.centralWidget))
            self.brainGridLayout.addWidget(self.brainViews[mode], 0 if (mode < 2 != 0) else 1, 0 if (mode % 2 != 0) else 1, 1, 1)
        
        self.gridLayout.addLayout(self.brainGridLayout, 1, 1)
        
        self.menuBar = MenuBar(self)
        self.setMenuBar(self.menuBar)
        self.menuBar.addActionsToMenu()
#        self.menuBar.actionOpen.triggered.connect()
#        self.menuBar.actionSave.triggered.connect()

        # Define statusbar
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        
        # Initialize logger
        self.logger = Logger(self)
        self.gridLayout.addWidget(self.logger)
        
        self.setCentralWidget(self.centralWidget)
        QtCore.QMetaObject.connectSlotsByName(self)

    def _update_files_selection_list(self):
        self.selectionWidget.filesList.addItems([os.path.split(name)[1] for name in self.fileNames])

    def accept_params(self):

        if self.paramDialog != None:
            self.paramDialog.accept()

        params = {
            'surf': self.paramDialog.comboSurfMesh.currentText(),
            'colorbar': self.paramDialog.comboColorbar.currentText(),
            'shadow': True if (self.paramDialog.comboShadow.currentText() == 'Yes') else False
            }
        logging.info("Parameters chosen.")

        self.decomposition = Decomposition(self.fileNames)
        self.radarLayout.addWidget(self.decomposition.radar_plot(5))
        self._add_brain_views(**params)


    def reject_params(self):

        if self.paramDialog != None:
            self.paramDialog.reject()

        logging.info("Parameters not chosen. Using default parameters.")

        self.decomposition = Decomposition(self.fileNames)
        self._add_brain_views()

    def _add_brain_views(self, **params):
        if self.decomposition != None:
            for mode in range(len(self.brainViews)):
                self.brainViews[mode].addBrain(self.decomposition.saveHTML(**params)[mode])
        else:
            logging.error("Was not able to create Decomposition.")

    def choose_files(self):
        
        self.fileNames = self.openFileNamesDialog()
        
        if self.fileNames:
            self._update_files_selection_list()

            self.paramDialog = ParametersDialog()

            self.paramDialog.buttonBox.accepted.connect(self.accept_params)
            self.paramDialog.buttonBox.rejected.connect(self.reject_params)

            self.paramDialog.show()

    def reset(self):
        self.fileNames = ['']
        if (self.decomposition):
            self.decomposition.reset()
        self._update_files_selection_list()

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
        files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()",
                                                      "",
                                                      "MATLAB Files (*.mat)",
                                                      options=options)
        return files

    def saveFileDialog(self):
        options = QFileDialog.Options()
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
