# This Python file uses the following encoding: utf-8
import scipy.io as scp
import numpy as np
import cmath
import pandas as pd
import logging
from nilearn import datasets, plotting, surface
from nibabel.freesurfer.io import (read_annot, write_annot)
from plotting import Dashboard
from utils import *


class Decomposition:
    def __init__(self, filenames, how_many_modes=5):

        # Get Dynamic Modes from filenames  
        X, Y, self.atlas = self._check_data(  # this defines X, Y, N, T
                self._extract_data(filenames)  # this defines data
            )
        # this defines eigVal, eigVec, eigIdx, A
        self.eigVal, self.eigVec, self.eigIdx, _ = self._get_dynamic_modes(X, Y)

        # List of Dictionaries [ {'left':'pathL1', 'right':'pathR1'}, ...]
        self.annots = [self._write_labels(dir, np.real(self.eigVec[:, mode + 1]))
                       for mode, dir in enumerate(self._create_mode_dirs(how_many_modes)) ]

    def dashboard(self, modes=5):
        """
        Create RadarPlot instance displaying active networks.

        :param modes: modes to plot in radar plot (int)
        :return: RadarPlot instance
        """

        labels = [ATLAS['networks'][self.atlas][network]['name'] for network in ATLAS['networks'][self.atlas]]
        idx = [ATLAS['networks'][self.atlas][network]['index'] for network in ATLAS['networks'][self.atlas]]
        # Global Variables contain MATLAB (1->) vs. Python (0->) indices
        index = [np.add(np.asarray(idx[i]), -1) for i in range(len(idx))]

        df_radar = pd.DataFrame(columns=['mode', 'network', 'strength', 'complex'])

        for axis in ['real', 'imag']:

            to_axis = (np.real if axis == 'real' else np.imag)

            for mode in range(1, modes + 1):

                for n, network in enumerate(labels):

                    df_radar = pd.concat([df_radar,
                                          pd.DataFrame({'Mode': ['Mode {}'.format(mode)],
                                                        'Network': [network],
                                                        'Strength': [np.mean(np.abs(to_axis(self.eigVec[index[n],
                                                                                                        mode + 1])))],
                                                        'complex': [axis]
                                                        })
                                          ])

        df_spectre = pd.DataFrame(
            data={'Mode': np.flip(range(1, np.unique(np.abs(self.eigVal)).shape[0] + 1)),
                  'Absolute Value of Eigenvalue': np.unique(np.abs(self.eigVal))})

        return Dashboard(df_radar, df_spectre)

    @staticmethod
    def reset():
        """
        Reset
        """
        clear_cache()
        reset_target()

    def _create_mode_dirs(self, how_many_modes):
        """
        Make directories to store created files.

        :param how_many_modes: how many modes should be created
        :return modes: list of paths
        """
        reset_target()

        modes = [self._create_mode_dir(mode + 1) for mode in range(how_many_modes)]

        return modes

    @staticmethod
    def _create_mode_dir(mode):
        """
        Create individual mode directory
        :param mode: mode number
        :return: path
        """
        name = 'mode-{}'.format(mode)
        Path(TARGET_DIR.joinpath(name)).mkdir()

        return name

    def save_HTML(self, surf='inflated', colorbar='RdYlGn', shadow=True):
        """
        Save 3D brain surface plots to html files.

        :return htmls: list of dict of hemispheric .html filepaths
        """
        htmls = []

        for mode in range(len(self.annots)):
            
            htmls.append(mode)
            htmls[mode] = {}
            
            for hemi in ['left', 'right']:

                dir = Path(self.annots[mode][hemi]).parent

                # TODO ADD FILE EXTENSION PROBLEM
                htmls[mode][hemi] = dir.joinpath('{}.html'.format(hemi))

                labels = surface.load_surf_data(TARGET_DIR.joinpath(self.annots[mode][hemi]).as_posix())

                args = {'surf_mesh': FSAVERAGE['{0}_{1}'.format(surf[:4], hemi)],
                        'surf_map': labels,
                        'cmap': colorbar,
                        'bg_map': FSAVERAGE['sulc_{}'.format(hemi)] if shadow else None,
                        'symmetric_cmap': False,
                        'colorbar_fontsize': 10,
                        'title': 'Dynamic Mode {0}: {1} hemisphere'.format(str(mode+1), hemi),
                        'title_fontsize': 12}

                view = plotting.view_surf(**args)

                view.save_as_html(TARGET_DIR.joinpath(htmls[mode][hemi]))
        
        return htmls

    def _write_labels(self, modedir, eigenVector):
        """
        Write Data mapping .annot file for Dynamic Modes.

        :param modedir: directory of mode
        :param eigenVector: eigenvector of selected Dynamic Mode.
        :return writepath: dict of 'left' and 'right' hemispheric .annot filepaths
        """
        writepath = {}

        for hemi in ['left', 'right']:
            # extract parcellation
            labels, ctab, names = read_annot(RES_DIR.joinpath(ATLAS['label'][self.atlas][hemi]))

            # initialize new color tab
            atlasLength = ctab.shape[0]
            newCtab = np.empty((atlasLength, 5))

            # first row is always the same as old colortab
            newCtab[0] = ctab[0]

            # change color according to eigenvector values
            scale = 127 / (np.max(eigenVector) - np.min(eigenVector))
            mean = np.mean(eigenVector)

            for roi in range(1, atlasLength):
                eigVecIdx = (roi - 1) if (hemi == 'left') else (roi + (atlasLength - 2))
                color = 127 + scale * (eigenVector[eigVecIdx] - mean)
                newCtab[roi] = np.array([color, color, color, 0, labels[roi]])

            writepath[hemi] = Path(modedir).joinpath('{}.annot'.format(hemi))

            write_annot(TARGET_DIR.joinpath(writepath[hemi]), labels, newCtab, names, fill_ctab=True)

        return writepath

    def _get_dynamic_modes(self, X, Y):
        """
        Get dynamic modes by Least Squares optimization.
        To use the index simply use eigVal[eigIdx] and eigVec[:,eigIdx]

        :param X: data for t (1->T)
        :param Y: data for t (0->T-1)
        :return eigVal: eigenvalues of dynamic mode decomposition
        :return eigVec: eigenvectors of dynamic mode decomposition
        :return eigIdx: eigen-indices for descendent sorting
        :return A: decomposition matrix
        """
        A = (X @ Y.T) @ (np.linalg.inv(Y @ Y.T))

        # extract eigenvalues and eigenvectors
        eigVal, eigVec = np.linalg.eig(A)

        # sort descending - from https://stackoverflow.com/questions/8092920/
        # simply use index for later use
        eigIdx = np.abs(eigVal).argsort()[::-1]

        # adjust eigenvectors' phases for orthogonality
        eigVec = self._adjust_phase(eigVec)

        return eigVal, eigVec, eigIdx, A

    def _check_data(self, data):
        """
        Check and format data into autoregressive model.

        :param data: list of NumPy arrays
        :return X: data for t (1->T)
        :return Y: data for t (0->T-1)
        :return atlas: cortical parcellations 'glasser' or 'schaefer'
        """
        # 'empty' arrays for creating X and Y
        X = np.array([]).reshape(data[0].shape[0], 0)
        Y = np.array([]).reshape(data[0].shape[0], 0)

        for matrice in data:
            # check for zero rows
            # indices of rows that are zero (full zero ROIs)
            zIdx = np.where(~matrice.any(axis=1))[0]
            logging.warning("ROIError: Matrice contains " + str(zIdx.shape) + " zero rows.")
            # TODO: remove row ? how will that be shown in the graph then ????

            # normalize matrices
            matriceN, _, _ = self.normalize(matrice, direction=1, demean=True, destandard=False)

            # concatenate matrices
            Xn = matriceN[:, 1:  ]
            Yn = matriceN[:,  :-1]
            X = np.concatenate((X, Xn), axis=1)
            Y = np.concatenate((Y, Yn), axis=1)

        if str(X.shape[0]) in ATLAS['atlas'].keys():
            atlas = ATLAS['atlas'][str(X.shape[0])]
            return X, Y, atlas
        else:
            logging.error("ROIError: Number of ROIs does not correspond to any known parcellation.")
            return X, Y, None

    def _extract_data(self, filenames):
        """
        Extracts fMRI data from files.
        Supported formats are: .mat

        :param filenames: list of filenames
        :return data: list of NumPy arrays
        """
        data = []

        if isinstance(filenames, list):
            for file in filenames:
                if file_format(file) == '.mat':
                    mat = scp.loadmat(file)
                    data.append(mat['TCSnf'])

        return data

    @staticmethod
    def normalize(x, direction=1, demean=True, destandard=False):
        """
        Normalize the original data set.

        :param x: data
        :param direction: 0 for columns, 1 for lines, None for global
        :param demean: demean
        :param destandard: remove standard deviation (to 1)
        :return x: standardized data
        :return mean: mean of original data
        """
        r, c = x.shape
        std_x = np.std(x, axis=direction)
        mean = np.mean(x, axis=direction)

        # removal of mean
        if demean:
            x -= mean.reshape( (mean.shape[0]), 1 )
            # different implementation
            # x -= mean[:, None]

        # removal of standard deviation
        if destandard:
            # not sure about formulation - not needed in this protocol should usually be F
            # x = x / std_x
            print('Standardization formula not too sure')

        return x, mean, std_x

    @staticmethod
    def _adjust_phase(x):
        """
        Adjust phase as to have orthogonal eigenVectors

        :param x: original eigenvectors
        :return ox: orthogonalized eigenvectors
        """

        # create empty instance for ox
        ox = np.empty(shape = x.shape, dtype=complex)

        for j in range(1, x.shape[1]):

            # seperate real and imaginary parts
            a = np.real(x[:,j])
            b = np.imag(x[:,j])

            # phase calculation
            phi = 0.5 * np.arctan( 2 * (a @ b) / (b.T @ b - a.T @ a) )

            # compute normalised a, b
            anorm = np.linalg.norm( math.sin(phi) * a + math.cos(phi) * b )
            bnorm = np.linalg.norm( math.cos(phi) * a - math.sin(phi) * b )

            if bnorm > anorm:
                if phi < 0:
                    phi -= PI / 2
                else:
                    phi += PI / 2

            ox[:,j] = np.multiply(x[:,j], cmath.exp( complex(0,1) * phi ))

        return ox