# This Python file uses the following encoding: utf-8
import os
import logging
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from elements import *
from plotting import *
from decomposition import Decomposition


class ToolboxWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setObjectName("ToolboxWindow")
        self.resize(1980, 1020)
        self.decomposition = None
        self.init_ui()

    def init_ui(self):
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
        for mode in range(1):
            self.brainViews.append(BrainView(self.centralWidget))
            self.brainGridLayout.addWidget(self.brainViews[mode], 0 if (mode < 2 != 0) else 1, 0 if (mode % 2 != 0) else 1, 1, 1)
        
        self.gridLayout.addLayout(self.brainGridLayout, 1, 1)
        
        self.menuBar = MenuBar(self)
        self.setMenuBar(self.menuBar)
        self.menuBar.add_actions_to_menu()
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

        self.selectionWidget.filesList.clear()

        if not self.fileNames:
            pass
        else:
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
        if not self.decomposition:
            logging.error('DecompositionError: Decomposition does not exist.')
        else:
            htmls = self.decomposition.save_HTML(**params)
            if len(self.brainViews) <= len(htmls):
                for i in range(len(self.brainViews)):
                    self.brainViews[i].add_brain(htmls[i])
            else:
                logging.error('HTML number less than Brain View number.')

    def choose_files(self):
        
        self.fileNames = self.open_file_names_dialog()
        
        if self.fileNames:

            self._update_files_selection_list()

            self.paramDialog = ParametersDialog()

            self.paramDialog.buttonBox.accepted.connect(self.accept_params)
            self.paramDialog.buttonBox.rejected.connect(self.reject_params)

            self.paramDialog.show()

    def reset(self):
        self.fileNames = []
        if self.decomposition != None:
            self.decomposition.reset()
        self._update_files_selection_list()

    def open_file_name_dialog(self):
        options = QFileDialog.Options()
        _file_name, _ = QFileDialog.getOpenFileName(self,
                                                  "QFileDialog.getOpenFileName()",
                                                  "",
                                                  "MATLAB File (*.mat)",
                                                  options=options)
        return _file_name

    def open_file_names_dialog(self):
        options = QFileDialog.Options()
        _files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()",
                                                      "",
                                                      "MATLAB Files (*.mat)",
                                                      options=options)

        return _files

    def save_file_dialog(self):
        options = QFileDialog.Options()
        _file_name, _ = QFileDialog.getSaveFileName(self,
                                                  "QFileDialog.getSaveFileName()",
                                                  "",
                                                  "All Files (*);;Text Files (*.txt)",
                                                  options=options)
        return _file_name
