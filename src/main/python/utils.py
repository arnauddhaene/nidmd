import math
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import cm
from pathlib import *
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import load_surf_mesh


# Directories

# Absolute path to DynamicModeToolbox directory
ROOT_DIR = (Path.cwd().parent.parent.parent if Path.cwd().name == 'python' else Path.cwd())
# Resource directory
RES_DIR = ROOT_DIR.joinpath('src/main/resources')
# Target directory for file output
TARGET_DIR = ROOT_DIR.joinpath('target')
# Cache directory
CACHE_DIR = ROOT_DIR.joinpath('cache')

# Math

PI = math.pi

# Neuroimaging Atlas Resource Access

def _get_surface():
    """
    Get surface for plotting.

    :return FSAVERAGE: surface locations as in nilearn
    :return SURFACE: surface for plotting
    """

    FSAVERAGE = fetch_surf_fsaverage('fsaverage', RES_DIR.as_posix())
    SURFACE = {}

    for key in [t + '_' + h for t in ['pial', 'infl'] for h in ['left', 'right']]:

        surf = load_surf_mesh(FSAVERAGE[key])
        x, y, z = np.asarray(surf[0].T, dtype='<f4')
        i, j, k = np.asarray(surf[1].T, dtype='<i4')

        SURFACE[key] = dict(x=x, y=y, z=z, i=i, j=j, k=k)

    return FSAVERAGE, SURFACE

ATLAS = json.load(open(RES_DIR.joinpath('ATLAS.JSON')))
# Use surface as follows: go.Mesh3d(**SURFACE['pial_left'], vertexcolor=...)
FSAVERAGE, SURFACE = _get_surface()
# 2D coordinates for ATLASSES
ATLAS2D = dict(schaefer=pd.read_json(RES_DIR.joinpath('schaefer.json').as_posix()),
               glasser=pd.read_json(RES_DIR.joinpath('glasser.json').as_posix()))

# Colors
COLORS = ['#50514f', '#f25f5c', '#ffe066', '#247ba0', '#70c1b3']


def matplotlib_to_plotly(cmap, pl_entries=255):
    """
    Matplotlib to Plotly colorscale

    :param cmap: mpl cm
    :param pl_entries: number of entries (default 255)
    :return: plotly cm
    """
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale


COOLWARM = matplotlib_to_plotly(matplotlib.cm.get_cmap('coolwarm'))

# File handling

def file_format(filename):
    """
    Find file format.

    :param filename: filename
    :return format: str of the file format
    """
    return Path(filename).suffix


def clear(dir):
    """
    Clear a directory

    :param dir: directory
    """
    if Path(dir).exists() and Path(dir).is_dir():
        for file in Path(dir).iterdir():
            try:
                if Path(file).is_dir():
                    clear(file)
                    Path(file).rmdir()
                else:
                    Path(file).unlink()
            except OSError as e:
                logging.error('PathError: failed to remove {0} from {1}.'.format(file, dir))


def clear_cache():
    """
    Clear cache directory.
    """
    clear(CACHE_DIR)


def clear_target():
    """
    Clear target directory.
    """
    clear(TARGET_DIR)


def reset_target():
    """
    Reset target workgin directory.
    """
    if TARGET_DIR.exists():
        clear_target()
    else:
        TARGET_DIR.mkdir()
