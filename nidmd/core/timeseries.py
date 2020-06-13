"""
timeseries.py
================
The core class to define time-series data.
"""

import cmath
import logging
import scipy.io as scp
import numpy as np
import numpy.linalg as la

from pathlib import *


class TimeSeries:
    """
    Representation of time-series data.
    """

    def __init__(self, data=None, filenames=None, sampling_time=None):
        """
        TimeSeries Constructor.

        Parameters
        ----------
        data : Array-like, optional
            Preprocessed time-series fMRI data. Can be a list of Array-like.
        filenames : str, optional
            Filenames of :code:`.mat` files containing data. Can be a list of :strong:`str`
        sampling_time : float, optional
            Sampling time of time-series recording.
        """

        self.sampling_time = sampling_time
        self.data = []

        if data is not None:
            if isinstance(data, np.ndarray):
                self.add(data)
            else:
                assert isinstance(data, list)
                for d in data:
                    self.add(d)
        elif filenames is not None:
            if isinstance(filenames, str):
                self.extract(filenames)
            else:
                assert isinstance(filenames, list)
                for f in filenames:
                    self.extract(f)

    def dmd(self, normalize=True):
        """
        Returns a dictionary-like object containing Dynamic Mode Decomposition elements.

        Parameters
        ----------
        normalize : boolean
            Normalize data before decomposition (default True)

        Returns
        -------
        dmd : Dictionary-like
            Dynamic Mode Decomposition elements with keys: {values, vectors, indices, A, activity}
        """

        if self.data is None:
            raise ValueError("No data to perform DMD on.")

        x, y = self.split(self.data, normalize)

        eig_val, eig_vec, eig_idx, a = self.get_decomposition(x, y)

        activity_t = la.inv(eig_vec) @ x

        return dict(values=eig_val, vectors=eig_vec, indices=eig_idx, A=a, activity=activity_t)

    def get_decomposition(self, x, y):
        """
        Get dynamic modes by Least Squares optimization of Auto-regressive model.

        Parameters
        ----------
        x : Array-like
            data for t (1->T)
        y : Array-like
            data for t (0->T-1)

        Returns
        -------
        eig_val : Array-like
            Eigenvalues of the eigen-decomposition of the Auto-regressive matrix
        eig_vec : Array-like
            Eigenvectors of the eigen-decomposition of the Auto-regressive matrix
        eig_idx : Array-like
            Indices that sort the eigenvalues in descending order
        A : Array-like
            The Auto-regressive matrix.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.shape == y.shape

        a = (x @ y.T) @ (np.linalg.inv(y @ y.T))

        # extract eigenvalues and eigenvectors
        eig_val, eig_vec = np.linalg.eig(a)

        # sort descending - from https://stackoverflow.com/questions/8092920/
        # simply use index for later use
        eig_idx = np.abs(eig_val).argsort()[::-1]

        # adjust eigenvectors' phases to assure their orthogonality
        eig_vec = self.adjust_phase(eig_vec)

        return eig_val, eig_vec, eig_idx, a

    @staticmethod
    def split(data, normalize=True):
        """
        Split time-series into X: [1->T] and Y:[0->T-1].

        Parameters
        ----------
        data : Array-like
            Time-series data. Can be list of Array-like.
        normalize : boolean
            For normalization of the input data.

        Returns
        -------
        x : Array-like
            Time-series data from t:1->T
        y : Array-like
            Time-series data from t:0->T-1

        Raises
        ------
        ValueError
            If input is invalid
        """

        if isinstance(data, np.ndarray):

            if normalize:
                data, _, _ = TimeSeries.normalize(data, direction=1, demean=True, destandard=True)

            return data[:, 1:], data[:, :-1]
        elif isinstance(data, list):
            # 'empty' arrays for creating X and Y
            x = np.array([]).reshape(data[0].shape[0], 0)
            y = np.array([]).reshape(data[0].shape[0], 0)

            for matrix in data:

                matrix = np.asarray(matrix)
                assert isinstance(matrix, np.ndarray)

                # check for zero rows
                # indices of rows that are zero (full zero ROIs)
                z_idx = np.where(~matrix.any(axis=1))[0]
                if z_idx.shape[0] > 0:
                    logging.warning('Matrix contains {} zero rows.'.format(z_idx.shape))

                # normalize matrices
                if normalize:
                    matrix, _, _ = TimeSeries.normalize(matrix, direction=1, demean=True, destandard=True)

                # concatenate matrices
                x_temp = matrix[:, 1:]
                y_temp = matrix[:, :-1]
                x = np.concatenate((x, x_temp), axis=1)
                y = np.concatenate((y, y_temp), axis=1)

            return x, y
        else:
            raise ValueError("Wrong input. Must be Array-like or list of Array-like.")

    def extract(self, filename):
        """
        Extracts fMRI data from file. Supported formats are: { code:`.mat` }

        Parameters
        ----------
        filename : str
            Path to file containing time-series data.

        Raises
        ------
        ImportError
            If file does not contain matrix
        """

        assert isinstance(filename, str)
        assert Path(filename).exists()

        if Path(filename).suffix == '.mat':
            mat = scp.loadmat(filename)
            d = None

            for key in mat.keys():
                if key[:2] != '__':
                    d = mat[key]
                    logging.info("Extracted matrix from file {} from key {}".format(filename, key))
                    continue

            if d is None:
                logging.error("Can not find matrix inside .mat file.")
                raise ImportError("Can not find matrix inside .mat file.")

        elif Path(filename).suffix == '.csv':

            d = np.genfromtxt(filename, delimiter=",")

        self.add(d)

    def add(self, data):
        """
        Add data

        Parameters
        ----------
        data : Array-like
            Time-series data.
        """

        data = np.asarray(data)
        assert isinstance(data, np.ndarray)

        self.data.append(data)

    @staticmethod
    def normalize(data, direction=1, demean=True, destandard=True):
        """
        Normalize a matrix

        Parameters
        ----------
        data : Array-like
            data matrix
        direction : int, optional
            0 for columns, 1 for rows (default), None for global
        demean : boolean, optional
            Normalize mean (default true)
        destandard : boolean, optional
            Normalize standard-deviation (default true)

        Returns
        -------
        x : Array-like
            Normalized matrix
        mean : float
            Mean of original data.
        std : float
            Standard deviation of original data.
        """

        # Handle data
        x = data.copy()
        x = np.asarray(x)
        assert isinstance(x, np.ndarray)

        if direction is None:
            return (x - np.mean(x)) / np.std(x), np.mean(x), np.std(x)

        # Fetch statistical information
        std = np.std(x, axis=direction)
        mean = np.mean(x, axis=direction)

        assert mean.shape[0] == std.shape[0]

        shape = (1, mean.shape[0]) if direction == 0 else (mean.shape[0], 1)

        # normalization of mean
        if demean:
            x -= mean.reshape(shape)

        # normalization of standard deviation
        if destandard:
            x /= std.reshape(shape)

        return x, mean, std

    @staticmethod
    def adjust_phase(x):
        """
        Adjust phase of matrix for orthogonalization of columns.

        Parameters
        ----------
        x : Array-like
            data matrix

        Returns
        -------
        ox : Array-like
            data matrix with orthogonalized columns
        """

        x = np.asarray(x)
        assert isinstance(x, np.ndarray)

        # create empty instance for ox
        ox = np.empty(shape=x.shape, dtype=complex)

        for j in range(x.shape[1]):

            # seperate real and imaginary parts
            a = np.real(x[:, j])
            b = np.imag(x[:, j])

            # phase calculation
            phi = 0.5 * np.arctan(2 * (a @ b) / (b.T @ b - a.T @ a))

            # compute normalised a, b
            anorm = np.linalg.norm(np.cos(phi) * a - np.sin(phi) * b)
            bnorm = np.linalg.norm(np.sin(phi) * a + np.cos(phi) * b)

            if bnorm > anorm:
                if phi < 0:
                    phi -= np.pi / 2
                else:
                    phi += np.pi / 2

            adjed = np.multiply(x[:, j], cmath.exp(complex(0, 1) * phi))
            ox[:, j] = adjed if np.mean(adjed) >= 0 else -1 * adjed

        return ox

    @staticmethod
    def match_modes(tc, s, m):
        """
        Match modes using Time Series data of match group and eigenvectors of reference group.

        Parameters
        ----------
        tc : Array-like
            Raw time-series data from match group
        s : Array-like
            Eigenvectors from the eigen-decomposition of the auto-regressive model of the reference group.
        m : int
            number of modes analyzed for approximation

        Returns
        -------
        d : Array-like
            Approximation of the :code:`m` first modes matched to the Reference group.

        Raises
        ------
        AtlasError
            If cortical parcellation is not supported.
        """
        s_inv = la.inv(s)

        n, t = tc.shape

        b = np.empty([m, 1], dtype=complex)
        a = np.empty([m, m], dtype=complex)

        if tc.shape[0] != s.shape[0]:
            logging.error("Cortical parcellation of reference and match group do not correspond.")
            raise ValueError("Cortical parcellation of reference and match groups do not correspond.")

        t2, t1 = TimeSeries.split([tc])
        t2 = t2.T
        t1 = t1.T

        for r in range(m):

            r1 = s[:, r].reshape(n, 1) @ s_inv[r, :].reshape(1, n)

            for c in range(r, m):

                c1 = s[:, c].reshape(n, 1) @ s_inv[c, :].reshape(1, n)

                if r != c:

                    middle_matrix = (c1.T @ r1 + r1.T @ c1)

                    a[r, c] = t1.flatten() @ (t1 @ middle_matrix.T).flatten()

                    a[c, r] = a[r, c]

                else:

                    a[r, c] = 2 * t1.flatten() @ (t1 @ r1.T @ r1).flatten()

            b[r] = 2 * t2.flatten() @ (t1 @ r1.T).flatten()

        d = la.solve(a, b)

        return np.around(d, decimals=8)