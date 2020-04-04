# This Python file uses the following encoding: utf-8
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWebEngineWidgets import QWebEnginePage
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import os

class BrainPlot(QWebEngineView):
    def __init__(self, filepath):
        QWebEngineView.__init__(self)
        
        globalpath = os.path.join(os.getcwd(), filepath)
        self.load(QUrl('file:///' + globalpath))