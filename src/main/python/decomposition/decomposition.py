# This Python file uses the following encoding: utf-8
import scipy.io as scp
import numpy.linalg as la
import cmath
from nilearn import plotting, surface
from nibabel.freesurfer.io import (read_annot, write_annot)
from .mode import Mode
from utils import *


class Decomposition:
    def __init__(self):

        self.data = []

    def run(self):
        """
        Run decomposition

        :return:
        """

        if self.data is not None:

            # Get Dynamic Modes from filenames
            X, Y, self.atlas = self._check_data(  # this defines X, Y, N, T
                    self.data
            )

            # this defines eigVal, eigVec, eigIdx, A
            self.eigVal, self.eigVec, self.eigIdx, self.A = self._get_decomposition(X, Y)

            self.Z = la.inv(self.eigVec) @ X

            # this defines the general data frame
            self.df = self._compute(self.eigVal, self.eigVec, self.eigIdx, self.Z)

    def _compute(self, val, vec, index, time):

        modes = []

        atlas = ATLAS['atlas'][str(val.shape[0])]

        order = 1
        idx = 0
        val_sorted = val[index]
        vec_sorted = vec[:, index]
        time_sorted = time[index, :]

        labels = list(ATLAS['networks'][atlas].keys())
        netidx = [ATLAS['networks'][atlas][network]['index'] for network in
               ATLAS['networks'][atlas]]

        # Global Variables contain MATLAB (1->) vs. Python (0->) indices
        netindex = [np.add(np.asarray(netidx[i]), -1) for i in range(len(netidx))]

        while idx < index.shape[0]:

            conj = (idx < index.shape[0] - 1) and (val_sorted[idx] == val_sorted[idx + 1].conjugate())

            value = val_sorted[idx]

            strength_real = []
            strength_imag = []

            for n, network in enumerate(labels):
                strength_real.append(np.mean(np.abs(np.real(vec_sorted[netindex[n], idx]))))
                strength_imag.append(np.mean(np.abs(np.imag(vec_sorted[netindex[n], idx]))))

            modes.append(
                dict(
                    mode=order,
                    value=value,
                    intensity=vec_sorted[:, idx],
                    damping_time=(-1 / np.log(np.abs(value))) * self.sampling_time,
                    period=((2 * PI) / np.abs(np.angle(value))) * self.sampling_time if conj else np.inf,
                    conjugate=conj,
                    networks=labels,
                    strength_real=strength_real,
                    strength_imag=strength_imag,
                    activity=np.real(time_sorted[idx, :])
                )
            )

            order += 1
            idx += 1 if not conj else 2

        return pd.DataFrame(modes)

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

    def _get_decomposition(self, X, Y):
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
            if zIdx.shape[0] > 0:
                logging.warning("ROIError: Matrice contains " + str(zIdx.shape) + " zero rows.")

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

    def add_data(self, data, sampling_time):
        """
        Add data to decomposition.

        :param data: data matrix (N x T)
        :param sampling_time: sampling time (s)
        """

        self.sampling_time = sampling_time

        self.data.append(data)

    @staticmethod
    def normalize(X, direction=1, demean=True, destandard=True):
        """
        Normalize the original data set.

        :param x: data
        :param direction: 0 for columns, 1 for rows, None for global
        :param demean: demean
        :param destandard: remove standard deviation (to 1)
        :return x: standardized data
        :return mean: mean of original data
        """
        x = X.copy()
        r, c = x.shape
        std_x = np.std(x, axis=direction)
        mean = np.mean(x, axis=direction)

        # removal of mean
        if demean:
            x -= mean.reshape((mean.shape[0]), 1)

        # removal of standard deviation
        if destandard:
            x /= std_x.reshape((std_x.shape[0]), 1)

        return x, mean, std_x

    @staticmethod
    def _adjust_phase(x):
        """
        Adjust phase as to have orthogonal eigenVectors

        :param x: original eigenvectors
        :return ox: orthogonalized eigenvectors
        """

        # create empty instance for ox
        ox = np.empty(shape=x.shape, dtype=complex)

        for j in range(x.shape[1]):

            # seperate real and imaginary parts
            a = np.real(x[:, j])
            b = np.imag(x[:, j])

            # phase calculation
            phi = 0.5 * np.arctan(2 * (a @ b) / (b.T @ b - a.T @ a))

            # compute normalised a, b
            anorm = np.linalg.norm(math.cos(phi) * a - math.sin(phi) * b)
            bnorm = np.linalg.norm(math.sin(phi) * a + math.cos(phi) * b)

            if bnorm > anorm:
                if phi < 0:
                    phi -= PI / 2
                else:
                    phi += PI / 2

            adjed = np.multiply(x[:, j], cmath.exp(complex(0, 1) * phi))
            ox[:, j] = adjed if np.mean(adjed) >= 0 else -1 * adjed

        return ox

    def _match_mode(self, other):
        """
        Match mode to group (self)

        :param other: Decomposition instance of subject
        :return: pd.DataFrame containing 'mode', 'value', 'damping_time', 'period', 'conjugate'
        """

        modes = []

        atlas = ATLAS['atlas'][str(val.shape[0])]

        order = 1
        idx = 0
        val_sorted = val[index]

        labels = list(ATLAS['networks'][atlas].keys())


        while idx < index.shape[0]:

            conj = (idx < index.shape[0] - 1) and (val_sorted[idx] == val_sorted[idx + 1].conjugate())

            # TODO: modify sampling time with user input

            value = val_sorted[idx]

            strength_real = []
            strength_imag = []

            for n, network in enumerate(labels):
                strength_real.append(np.mean(np.abs(np.real(vec_sorted[netindex[n], idx]))))
                strength_imag.append(np.mean(np.abs(np.imag(vec_sorted[netindex[n], idx]))))

            modes.append(
                dict(
                    mode=order,
                    value=value,
                    damping_time=(-1 / np.log(np.abs(value))) * self.sampling_time,
                    period=((2 * PI) / np.abs(np.angle(value))) * self.sampling_time if conj else np.inf,
                    conjugate=conj
                )
            )

            order += 1
            idx += 1 if not conj else 2

        return pd.DataFrame(modes)