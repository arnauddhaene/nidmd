# This Python file uses the following encoding: utf-8
from PyQt5 import QtCore
from PyQt5 import QtWidgets

class MenuBar(QtWidgets.QMenuBar):
    def __init__(self, mainWindow):
        """
        Constructor
        """
        QtWidgets.QMenuBar.__init__(self, mainWindow)
        
        self.setGeometry(QtCore.QRect(0, 0, 588, 22))
        self.setObjectName("menuBar")
        
        self.menuFile = QtWidgets.QMenu(self)
        self.menuFile.setObjectName("menuFile")
        
        self.menuEdit = QtWidgets.QMenu(self)
        self.menuEdit.setObjectName("menuEdit")
        
        # Define menubar actions
        # Open
        self.actionOpen = QtWidgets.QAction(' &Open', mainWindow)
        self.actionOpen.setStatusTip("Open file(s)")
        self.actionOpen.setObjectName("actionOpen")
        self.actionOpen.setShortcut("Crtl+O")
        
        # Save
        self.actionSave = QtWidgets.QAction(' &Save', mainWindow)
        self.actionSave.setStatusTip("Save file(s)")
        self.actionSave.setObjectName("actionSave")
        self.actionSave.setShortcut("Crtl+S")
        
    def add_actions_to_menu(self):
        """
        Add actions to menus
        """
        # Add actions to menus
        self.menuFile = self.addMenu(' &File')
        self.menuFile.setTitle("File")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave)
