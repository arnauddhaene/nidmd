import json
import numpy as np
import pandas as pd
from pathlib import *

from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import load_surf_mesh


class AtlasError(Exception):
    """Exception for wrong call to cortical surface atlas."""
    pass


class Atlas:
    """ Representation of a Cortical Parcellation Atlas. """

    def __init__(self, nroi: int):
        """
        Atlas Constructor.

        Parameters
        ----------
        nroi : int
            Number of ROI.

        Yields
        ------
        size : int
            Number of ROI
        name : {'glasser', 'schaefer'}
            Name of cortical parcellation atlas. Supported are Glasser and Schaefer atlasses.
        coords_2d : pd.DataFrame
            Pandas DataFrame containing 2D coordinates of cortical surface plotting
        networks : Dictionary-like
            Object containing indices relevant to each network.
        """

        datapath = Path(__file__).parent.joinpath('data')

        metadata = json.load(open(Path.joinpath(datapath, 'ATLAS.JSON').as_posix()))

        if str(nroi) not in metadata['atlas'].keys():
            raise AtlasError('Number of ROIs ({}) incoherent with any installed cortical parcellation.'.format(nroi))

        self.size = nroi
        self.name = metadata['atlas'][str(nroi)]
        self.coords_2d = pd.read_json(Path.joinpath(datapath, '{}.json'.format(self.name)).as_posix())
        self.networks = metadata['networks'][self.name]

    def __eq__(self, other):
        """
        Check that two Atlas instances use the same cortical parcellation.

        Parameters
        ----------
        other : nidmd.Atlas
            other Atlas instance

        Returns
        -------
        equal : boolean
            True if both Atlas instances have the same number of ROIs
        """
        
        if isinstance(other, Atlas):
            return self.size == other.size

        return False

    @staticmethod
    def surface():
        """
        Returns  3D surface coordinates.

        Returns
        -------
        fsaverage : Dictionary-like
            (from the Nilearn documentation) The interest attributes are : - 'pial_left': Gifti file,
            left hemisphere pial surface mesh - 'pial_right': Gifti file, right hemisphere
            pial surface mesh - 'infl_left': Gifti file, left hemisphere inflated pial surface
            mesh - 'infl_right': Gifti file, right hemisphere inflated pial surface mesh
            - 'sulc_left': Gifti file, left hemisphere sulcal depth data - 'sulc_right': Gifti
            file, right hemisphere sulcal depth data
        surf : Dictionary-like
            Object containing the x, y, z coordinates as well as the i, j, k triangulation coordinates
        """
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
