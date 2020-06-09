"""
Radarplot
---------
Example for gallery with radarplot
"""

from dmd import Decomposition, Radar
import plotly


dcp = Decomposition(filenames='/home/adhaene/Documents/Dynamic Mode Toolbox Data/Rest1LR_Sub100307_Glasser.mat')

fig = Radar(dcp.df, dcp.atlas).figure()

plotly.io.show(fig)
