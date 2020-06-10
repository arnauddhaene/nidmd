import matplotlib
from matplotlib import cm
import numpy as np

colorlist = ['#1abc9c', '#2ecc71', '#3498db', '#9b59b6', '#34495e', '#f1c40f', '#e67e22', '#e74c3c', '#95a5a6',
             '#16a085', '#27ae60', '#2980b9', '#8e44ad', '#2c3e50', '#f39c12', '#d35400', '#c0392b', '#7f8c8d']

def matplotlib_to_plotly(cmap, pl_entries=255):
    """
    Matplotlib to Plotly colorscale
    from `<https://plotly.com/python/v3/matplotlib-colorscales/>`_

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


def colorscale(colorscale: str):
    return matplotlib_to_plotly(matplotlib.cm.get_cmap(colorscale))