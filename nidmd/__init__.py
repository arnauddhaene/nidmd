"""
Dynamic Mode Decomposition
--------------------------------------------------
Documentation is available in the docstrings and online at
http://arnauddhaene.github.io/nidmd
Contents
--------
nidmd aims at creating a comprehensible analysis tool for
dynamic mode decomposition of time-series fMRI data.
Submodules
---------
plotting                --- Plotting code for nidmd
core                    --- core framework for nidmd
"""

from pathlib import *

from nidmd.core import (Decomposition, TimeSeries)
from nidmd.plotting import (Brain, Spectre, Radar, TimePlot)
from nidmd.datasets import Atlas

__all__ = ['Decomposition', 'TimeSeries', 'Brain', 'Spectre', 'Radar', 'TimePlot', 'Atlas']

ROOT_DIR = Path(__file__).parent
