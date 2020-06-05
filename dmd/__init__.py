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

ROOT_DIR = Path.cwd()

__all__ = ['decomposition', 'plotting', 'datasets', 'cm']
