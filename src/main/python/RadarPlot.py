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

class RadarPlot(FigureCanvas):

    def __init__(self, data, labels, title='Network Activation', mode_titles=None):
        """
        Constructor.
        :param data: dict with keys ('real','complex') and list of lists of mode values
        :param labels: list of labels
        :param mode_titles: list of mode titles
        """
        # the figure is now accessible via self.figure
        FigureCanvas.__init__(self, Figure(figsize=(4, 4)))

        # load data
        if mode_titles is None:
            self.mode_titles = ['Mode {}'.format(i + 1) for i in range(len(data['real']))]
            logging.info('Automatic mode titles created for Radar Chart.')
        else:
            self.mode_titles = mode_titles

        if not isinstance(data, dict):
            logging.error('DataError: data must be a dict with keys \'real\' and \'complex\' containing a list of lists.')

        self.data = [
            labels,
            ('Real', data['real']),
            ('Complex', data['complex'])
        ]

        self._create_plot()
        self.show()

    def _create_plot(self):

        spoke_labels = self.data.pop(0)
        N = len(spoke_labels)
        theta = radar_factory(N, frame='polygon')

        axes = self.figure.subplots(subplot_kw={'projection': 'radar'}, nrows=2, ncols=1)

        self.figure.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

        # colors = ['#ff4d4d', '#1b9868', '#ffd700', '#fff0cf', '#abcdef']
        colors = ['b', 'r', 'g', 'm', 'y']

        for ax, (title, mode_data) in zip(axes.flat, self.data):
            ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
            ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                         horizontalalignment='center', verticalalignment='center')

            for datum, color in zip(mode_data, colors):

                if len(datum) != len(spoke_labels):
                    logging.error('ValueError: data and label counts don\'t match')

                ax.plot(theta, datum, color=color)
                ax.fill(theta, datum, facecolor=color, alpha=0.25)

            ax.set_varlabels(spoke_labels)


        # add legend relative to top-left plot
        labels = tuple(self.mode_titles)
        legend = axes[0].legend(labels, loc=(-0.4, -0.2),
                           labelspacing=0.3, fontsize='small')

        self.figure.text(0.5, 0.965, 'Dynamical Mode Network Decomposition',
                 horizontalalignment='center', color='black', weight='bold',
                 size='large')

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.
    source: https://matplotlib.org/3.1.0/gallery/specialty_plots/radar_chart.html

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels, rotation_mode='anchor')

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta