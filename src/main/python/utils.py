import math
import json
import logging
from pathlib import *
from nilearn.datasets import fetch_surf_fsaverage


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

ATLAS = json.load(open(RES_DIR.joinpath('ATLAS.JSON')))
FSAVERAGE = fetch_surf_fsaverage('fsaverage', RES_DIR.as_posix())

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
