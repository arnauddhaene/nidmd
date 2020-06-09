"""
Radarplot
---------
Example for gallery with radarplot
"""

from dmd import Decomposition, Radar


dcp = Decomposition(filenames='/home/adhaene/Documents/Dynamic Mode Toolbox Data/Rest1LR_Sub100307_Glasser.mat')

plot = Radar(dcp.df, dcp.atlas).figure()
