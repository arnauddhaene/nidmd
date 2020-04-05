# This Python file uses the following encoding: utf-8
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl


class BrainPlot(QWebEngineView):
    def __init__(self, path):
        """
        Initialize Brain Plot.

        :param path: absolute Path to html file
        """
        QWebEngineView.__init__(self)

        self.load(QUrl('file:///' + path.as_posix()))
