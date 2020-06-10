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

    @staticmethod
    def split(data):
        """
        Split time-series into X: [1->T] and Y:[0->T-1].

        Parameters
        ----------
        data : Array-like
            Time-series data. Can be list of Array-like.

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
                matrix_n, _, _ = TimeSeries.normalize(matrix, direction=1, demean=True, destandard=True)

                # concatenate matrices
                x_temp = matrix_n[:, 1:]
                y_temp = matrix_n[:, :-1]
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

        assert Path(filename).exists()

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

        # Fetch statistical information
        std = np.std(x, axis=direction)
        mean = np.mean(x, axis=direction)

        # normalization of mean
        if demean:
            x -= mean.reshape((mean.shape[0]), 1)

        # normalization of standard deviation
        if destandard:
            x /= std.reshape((std.shape[0]), 1)

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