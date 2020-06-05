import matplotlib
from matplotlib import cm
import numpy as np


class Colors:
    """ Class for representing colors. """

    def __init__(self):

        self.coolwarm = self.matplotlib_to_plotly(matplotlib.cm.get_cmap('coolwarm'))

    @staticmethod
    def matplotlib_to_plotly(cmap, pl_entries=255):
        """
        Matplotlib to Plotly colorscale
        from https://plotly.com/python/v3/matplotlib-colorscales/

        :param cmap: Union[Colormap, LinearSegmentedColormap, ListedColormap] mpl cm
        :param pl_entries: [int] number of entries (default 255)
        :return: plotly cm
        """
        h = 1.0 / (pl_entries - 1)
        pl_colorscale = []

        for k in range(pl_entries):
            color = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
            pl_colorscale.append([k * h, 'rgb' + str((color[0], color[1], color[2]))])

        return pl_colorscale

    @staticmethod
    def colorscale(colorscale: str):
        return Colors.matplotlib_to_plotly(matplotlib.cm.get_cmap(colorscale))