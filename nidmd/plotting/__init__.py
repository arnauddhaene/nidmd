"""
Plotting code for nidmd
"""

from .brain import Brain
from .radar import Radar
from .timeplot import TimePlot
from .spectre import Spectre
from .colors import matplotlib_to_plotly, colorscale

__all__ = ['Brain', 'Radar', 'Spectre', 'TimePlot', 'matplotlib_to_plotly', 'colorscale']
