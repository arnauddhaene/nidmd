# This Python file uses the following encoding: utf-8
import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import logging

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


class SpectrePlot(FigureCanvas):

    def __init__(self, data):
        """
        Initialize Spectre Plot.

        :param data: input data tuple (x, y)
        """
        FigureCanvas.__init__(self, Figure(figsize=(5, 3)))

        self.x, self.y = data

        self._create_plot()
        self.show()

    def _create_plot(self):
        """
        Create plot.
        """

        # Data for plotting
        ax = self.figure.subplots()
        ax.bar(self.x, self.y, width=0.1)
        ax.plot(self.x, self.y, linewidth=1)

        ax.set(xlabel='Mode', ylabel='Eigenvalue',
               title='Spectre')
        ax.grid()
