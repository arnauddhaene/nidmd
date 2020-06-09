"""
Dynamic Mode Decomposition
--------------------------------------------------
Documentation is available in the docstrings and online at
http://arnauddhaene.github.io/dmd
Contents
--------
dmd aims at creating a comprehensible analysis tool for 
dynamic mode decomposition of time-series fMRI data.
Submodules
---------
plotting                --- Plotting code for dmd
decomposition           --- Dynamic Mode Decomposition
"""

from pathlib import *

ROOT_DIR = Path(__file__).parent

from dmd.core import (Decomposition, TimeSeries)

from dmd.plotting import (Brain, Spectre, Radar, TimePlot)

from dmd.datasets import Atlas

__all__ = ['core', 'plotting', 'datasets']
