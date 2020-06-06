import json
import numpy as np
import pandas as pd
from pathlib import *

from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import load_surf_mesh

from ..errors import AtlasError


class Atlas:
    """ Class for fetching atlas data. """

    def __init__(self, nroi: int):
        """
        Constructor

        :param nroi: [int] number of roi
        """

        datapath = Path(__file__).parent.joinpath('data')

        metadata = json.load(open(Path.joinpath(datapath, 'ATLAS.JSON')))

        if str(nroi) not in metadata['atlas'].keys():
            raise AtlasError('Number of ROIs ({}) incoherent with any installed cortical parcellation.'.format(nroi))

        self.size = nroi
        self.name = metadata['atlas'][str(nroi)]
        self.coords_2d = pd.read_json(Path.joinpath(datapath, '{}.json'.format(self.name)))
        self.networks = metadata['networks'][self.name]

    def __eq__(self, other):
        """
        Check that two Atlas instances use the same cortical parcellation
        :param other: [Atlas] other atlas instance
        :return: [boolean]
        """
        
        if isinstance(other, Atlas):
            return self.size == other.size

        return False

    @staticmethod
    def _get_surface():
        """
        Get surface for plotting.

        :return fsaverage: surface locations as in nilearn
        :return surf: surface for plotting
        """

        fsaverage = fetch_surf_fsaverage('fsaverage')
        surf = {}

        for key in [t + '_' + h for t in ['pial', 'infl'] for h in ['left', 'right']]:

            surf = load_surf_mesh(fsaverage[key])
            x, y, z = np.asarray(surf[0].T, dtype='<f4')
            i, j, k = np.asarray(surf[1].T, dtype='<i4')

            surf[key] = dict(x=x, y=y, z=z, i=i, j=j, k=k)

        return fsaverage, surf
