"""
decomposition.py
================
The core class to define a decomposition.
"""


import logging
import numpy as np
import numpy.linalg as la
import pandas as pd

from sklearn.linear_model import LinearRegression

from .timeseries import TimeSeries
from ..datasets.atlas import Atlas, AtlasError


class Decomposition(TimeSeries):
    """
    Representation of a Decomposition.
    """

    def __init__(self, data=None, filenames=None, sampling_time=None):
        """
        Decomposition Constructor.

        Parameters
        ----------
        data : Array-like
            Preprocessed time-series fMRI data. Can be list of Array-like
        filenames : str
            filenames of :code:`.mat` files containing data. Can be list of :strong:`str`.
        sampling_time : float, optional
            Sampling time of time-series recording.

        Yields
        ------
        data : Array-like
            Time-series raw data
        X : Array-like
            Time-series data from t:1->T
        Y : Array-like
            Time-series data from t:0->T-1
        atlas : nidmd.Atlas
            Cortical Parcellation atlas used for this decomposition
        eig_val : Array-like
            Eigenvalues of the eigen-decomposition of the Auto-regressive matrix
        eig_vec : Array-like
            Eigenvectors of the eigen-decomposition of the Auto-regressive matrix
        eig_idx : Array-like
            Indices for descending order of the eigen-decomposition of the Auto-regressive matrix
        A : Array-like
            Auto-regressive matrix
        Z : Array-like
            Approximation of the activity versus time for each mode
        df : pd.DataFrame
            Pandas DataFrame containing the following columns: mode, value, intensity, damping_time, period, conjugate,
            strength_real, strength_imag, activity
        """
        # Call to super class
        super().__init__(data, filenames, sampling_time)

        self.X = None
        self.Y = None
        self.atlas = None
        self.eig_val = None
        self.eig_vec = None
        self.eig_idx = None
        self.A = None
        self.Z = None
        self.df = None

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

        self.run()

    def add(self, data):
        """
        Add data to Decomposition.

        Parameters
        ----------
        data : Array-like
            Time-series data.
            
        Yields
        ------
        atlas : nidmd.Atlas
            Cortical Parcellation atlas used for this decomposition

        Raises
        ------
        ImportError
            If the import fails.
        """

        # Verify that data is correctly formatted
        try:
            self.atlas = Atlas(data.shape[0])
        except AtlasError:
            raise ImportError('Data import attempt failed.')

        super().add(data)

        logging.info('Data added to Decomposition using {} atlas.'.format(self.atlas))

    def run(self):
        """
        Run Decomposition.

        Returns
        -------
        df : pd.DataFrame
            Pandas DataFrame containing the following columns: mode, value, intensity, damping_time, period, conjugate,
            strength_real, strength_imag, activity
        """

        assert self.data is not None

        # Split data in X and Y
        self.X, self.Y = self.split(self.data)

        # Perform eigendecomposition
        self.eig_val, self.eig_vec, self.eig_idx, self.A = self._get_decomposition(self.X, self.Y)

        # Fetch time course for each mode
        self.Z = la.inv(self.eig_vec) @ self.X

        # Define general data frame
        self.df = self._compute(self.eig_val, self.eig_vec, self.eig_idx, self.Z)

        return self.df

    def _compute(self, val, vec, index, time):
        """
        Compute Decomposition to fetch DataFrame with all relevant info.

        Parameters
        ----------
        val : Array-like
            Eigenvalues of the eigen-decomposition of the AutoRegressive matrix
        vec : Array-like
            Eigenvectors of the eigen-decomposition of the AutoRegressive matrix
        index : Array-like
            Indices that sort the eigenvalues in descending order
        time : Array-like
            Approximation of the activity versus time for each mode

        Returns
        -------
        df : pd.DataFrame
            Pandas DataFrame containing the following columns: mode, value, intensity, damping_time, period, conjugate,
            strength_real, strength_imag, activity
        """

        modes = []

        assert val.shape[0] == vec.shape[0]
        assert vec.shape[0] == vec.shape[1]
        assert index.shape[0] == val.shape[0]
        assert time.shape[0] == val.shape[0]
        assert self.atlas is not None

        if self.sampling_time is None:
            self.sampling_time = 1.0

        order = 1
        idx = 0

        # Sort eigendecomposition and time course matrix
        val_sorted = val[index]
        vec_sorted = vec[:, index]
        time_sorted = time[index, :]

        # Fetch network labels
        labels = list(self.atlas.networks.keys())

        # Fetch indices of networks ROIs
        netidx = [self.atlas.networks[network]['index'] for network in
                  self.atlas.networks]

        # Global Variables contain MATLAB (1->) vs. Python (0->) indices
        netindex = [np.add(np.asarray(netidx[i]), -1) for i in range(len(netidx))]

        while idx < index.shape[0]:

            # Check if mode iterated on is a complex conjugate with the next
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
                    period=((2 * np.pi) / np.abs(np.angle(value))) * self.sampling_time if conj else np.inf,
                    conjugate=conj,
                    strength_real=strength_real,
                    strength_imag=strength_imag,
                    activity=np.real(time_sorted[idx, :])
                )
            )

            order += 1
            idx += 1 if not conj else 2

        return pd.DataFrame(modes)

    def _get_decomposition(self, x, y):
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
    def _match_modes(tc, s, m):
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
            raise AtlasError("Cortical parcellation of reference and match groups do not correspond.")

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

    def compute_match(self, other, m):
        """
        Get approximated matched modes for match group with self as a reference.
        Predicts amplification of approximated modes using linear regression.

        Parameters
        ----------
        other : nidmd.Decomposition
            match group
        m : int
            number of modes analyzed for approximation

        Returns
        -------
        modes : pd.DataFrame
            Pandas DataFrame containing the following columns: mode, value, damping_time, period, conjudate
        x : Array-like
            Vector containing absolute value of top 10 approximated eigenvalues of self (by mode matching to self)
        y : Array-like
            Vector containing absolute value of top 10 real eigenvalues of self
        """

        # First the modes should be matched with myself to get regression params
        logging.info('Fetching reference mode matching for regression parameter estimation.')

        borderline = self.eig_val[self.eig_idx][m].conj() == self.eig_val[self.eig_idx][m + 1]
        mm = (m + 1) if borderline else m

        own = self._match_modes(self.X, self.eig_vec[:, self.eig_idx], mm)
        assert np.asarray(own).shape[0] == mm

        # Top 10 modes are used in the dashboard
        reg = LinearRegression().fit(np.abs(own).reshape(-1, 1), np.abs(self.eig_val[self.eig_idx][:mm]).reshape(-1, 1))

        logging.info('Regression parameters estimated.')
        logging.info('Fetching mode estimation for match group.')

        others = self._match_modes(other.X, self.eig_vec[:, self.eig_idx], (m + 1) if borderline else m)

        # complex prediction of top
        others = reg.intercept_ + reg.coef_ * others[:10]
        others = others.flatten()

        logging.info("Matching modes approximation predicted.")

        modes = []

        order = 1
        idx = 0

        while idx < others.shape[0]:

            conj = (idx < others.shape[0] - 1) and (others[idx] == others[idx + 1].conjugate())

            value = others[idx]

            modes.append(
                dict(
                    mode=order,
                    value=value,
                    damping_time=(-1 / np.log(np.abs(value))) * self.sampling_time,
                    period=((2 * np.pi) / np.abs(np.angle(value))) * self.sampling_time if conj else np.inf,
                    conjugate=conj
                )
            )

            order += 1
            idx += 1 if not conj else 2

        return pd.DataFrame(modes), np.abs(own.flatten()), np.abs(self.eig_val[self.eig_idx][:mm])
