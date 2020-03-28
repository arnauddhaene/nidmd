# This Python file uses the following encoding: utf-8
import os
from PyQt5 import QtCore, QtWidgets
from BrainPlot import BrainPlot
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QHBoxLayout


class BrainView(QtWidgets.QHBoxLayout):
    def __init__(self, paths):
        """
        Constructor
        
        :param paths: dictionary with 'left' and 'right' html files
        """
        QtWidgets.QHBoxLayout.__init__(self)
           
        # Create BrainPlot instance for each hemisphere
        self.lHemisphere = BrainPlot(paths['left'])
        self.rHemisphere = BrainPlot(paths['right'])
    
        # Add each BrainPlot to the Horizontal Layout
        self.addWidget(self.lHemisphere)
        self.addWidget(self.rHemisphere)
