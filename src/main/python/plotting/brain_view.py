# This Python file uses the following encoding: utf-8
import logging
from PyQt5 import QtCore, QtWidgets
from plotting import BrainPlot
from utils import *


class BrainView(QtWidgets.QTabWidget):
    def __init__(self, parent):
        """
        Constructor
        
        :param parent: parent widget
        """
        QtWidgets.QTabWidget.__init__(self, parent)
        
        self.setMinimumSize(QtCore.QSize(600, 350))
        self.layout = QtWidgets.QHBoxLayout(self)

    def add_brain(self, paths):
        """
        Add html files to load brain view
        
        :param paths: dictionary with 'left' and 'right' html local paths
        """
        # Create BrainPlot instance for each hemisphere
        lHemisphere = BrainPlot(TARGET_DIR.joinpath(paths['left']))
        rHemisphere = BrainPlot(TARGET_DIR.joinpath(paths['right']))
        
        logging.info("HTML files loaded to BrainView.")
    
        # Add each BrainPlot to the Horizontal Layout
        self.addTab(lHemisphere, "Left")
        self.addTab(rHemisphere, "Right")