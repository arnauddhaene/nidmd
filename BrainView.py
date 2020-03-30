# This Python file uses the following encoding: utf-8
import os
from PyQt5 import QtCore, QtWidgets
from BrainPlot import BrainPlot
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QHBoxLayout
import logging

class BrainView(QtWidgets.QWidget):
    def __init__(self, parent):
        """
        Constructor
        
        :param parent: parent widget
        """
        QtWidgets.QWidget.__init__(self, parent)
        
        self.setMinimumSize(QtCore.QSize(600, 350))
        self.layout = QtWidgets.QHBoxLayout(self);
        
        
    def addBrain(self, paths):
        """
        Add html files to load brain view
        
        :param paths: dictionary with 'left' and 'right' html files
        """
        # Create BrainPlot instance for each hemisphere
        self.lHemisphere = BrainPlot(paths['left'])
        self.rHemisphere = BrainPlot(paths['right'])
        
        logging.info("HTML files loaded to BrainView.")
    
        # Add each BrainPlot to the Horizontal Layout
        self.layout.addWidget(self.lHemisphere)
        self.layout.addWidget(self.rHemisphere)
