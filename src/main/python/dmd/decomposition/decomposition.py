# This Python file uses the following encoding: utf-8
import scipy.io as scp
import numpy.linalg as la
from sklearn.linear_model import LinearRegression
import cmath
from utils import *


class Decomposition:
    def __init__(self, data=None, filenames=None):
        """
        Decomposition constructor

        :param data: [list of Array-like]
        :param filenames: [list of str]
        """

        if data is None and filenames is None:
            self.data = []
            self.X = None
            self.Y = None
            self.atlas = None
            self.eigVal = None
            self.eigVec = None
            self.eigIdx = None
            self.A = None
            self.Z = None
            self.df = None
            self.sampling_time = None
        else:
            if data is not None:
                assert isinstance(data, list)
                for d in data:
                    self.add_data(d)
            elif filenames is not None:
                assert isinstance(filenames, list)
                for f in filenames:
                    self._extract_data(f)

            self.run()

    def run(self):
        """ Run decomposition. """

        assert self.data is not None

        # Split data in X and Y
        self.X, self.Y = self._check_data(self.data)

        # Perform eigendecomposition
        self.eigVal, self.eigVec, self.eigIdx, self.A = self._get_decomposition(self.X, self.Y)

        # Fetch time course for each mode
        self.Z = la.inv(self.eigVec) @ self.X

        # Define general data frame
        self.df = self._compute(self.eigVal, self.eigVec, self.eigIdx, self.Z)

    def _compute(self, val, vec, index, time):
        """
        Compute general data frame

        :param val: [Array-like] eigenvalues
        :param vec: [Array-like] eigenvectors
        :param index: [Array-like] sorting index of eigendecomposition
        :param time: [Array-like] time course of each mode
        :return: [pd.DataFrame] general data frame
        """

        modes = []

        assert val.shape[0] == vec.shape[0]
        assert vec.shape[0] == vec.shape[1]
        assert index.shape[0] == val.shape[0]
        assert time.shape[0] == val.shape[0]
        assert self.atlas is not None
        assert self.sampling_time is not None

        order = 1
        idx = 0

        # Sort eigendecomposition and time course matrix
        val_sorted = val[index]
        vec_sorted = vec[:, index]
        time_sorted = time[index, :]

        # Fetch network labels
        labels = list(ATLAS['networks'][self.atlas].keys())

        # Fetch indices of networks ROIs
        netidx = [ATLAS['networks'][self.atlas][network]['index'] for network in
                  ATLAS['networks'][self.atlas]]

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

    def _get_decomposition(self, x, y):
        """
        Get dynamic modes by Least Squares optimization.
        To use the index simply use eigVal[eigIdx] and eigVec[:,eigIdx]

        :param X: [Array-like] data for t (1->T)
        :param Y: [Array-like] data for t (0->T-1)
        :return eigVal: [Array-like] eigenvalues of dynamic mode decomposition
        :return eigVec: [Array-like] eigenvectors of dynamic mode decomposition
        :return eigIdx: [Array-like] eigenindices for descendent sorting
        :return A: [Array-like] decomposition matrix
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

        # adjust eigenvectors' phases for orthogonality
        eig_vec = self._adjust_phase(eig_vec)

        return eig_val, eig_vec, eig_idx, a

    @staticmethod
    def _check_data(data):
        """
        Check and format data into autoregressive model.

        :param data: [list of Array-like]
        :return X: [Array-like] data for t (1->T)
        :return Y: [Array-like] data for t (0->T-1)
        """

        assert isinstance(data, list)

        # 'empty' arrays for creating X and Y
        x = np.array([]).reshape(data[0].shape[0], 0)
        y = np.array([]).reshape(data[0].shape[0], 0)

        for matrix in data:

            matrix = np.asarray(matrix)
            assert isinstance(matrix, np.ndarray)
            assert str(matrix.shape[0]) in ATLAS['atlas'].keys()

            # check for zero rows
            # indices of rows that are zero (full zero ROIs)
            z_idx = np.where(~matrix.any(axis=1))[0]
            if z_idx.shape[0] > 0:
                logging.warning("Matrice contains " + str(z_idx.shape) + " zero rows.")

            # normalize matrices
            matrix_n, _, _ = Decomposition.normalize(matrix, direction=1, demean=True, destandard=True)

            # concatenate matrices
            x_temp = matrix_n[:, 1:  ]
            y_temp = matrix_n[:,  :-1]
            x = np.concatenate((x, x_temp), axis=1)
            y = np.concatenate((y, y_temp), axis=1)

        return x, y

    def _extract_data(self, filename):
        """
        Extracts fMRI data from files.
        Supported formats are: .mat

        :param filename: [str] filename containing .mat matrix
        """

        assert file_format(filename) == '.mat'
        mat = scp.loadmat(filename)
        for key in mat.keys():
            if key[:2] != '__':
                d = mat[key]
                logging.info("Extracted matrice from file {} from key {}".format(filename, key))
                continue

        self.add_data(d)

    def add_data(self, data, sampling_time=None):
        """
        Add data to decomposition. Also defines Decomposition.atlas

        :param data: [Array-like] data matrix (N x T)
        :param sampling_time: [float] sampling time (s)
        """

        data = np.asarray(data)
        assert isinstance(data, np.ndarray)
        # Verify that data is correctly formatted
        assert str(data.shape[0]) in ATLAS['atlas'].keys()

        if len(self.data) == 0:
            self.atlas = ATLAS['atlas'][str(data.shape[0])]
            logging.info('Data added to Decomposition using {} atlas.'.format(self.atlas))
        else:
            assert self.atlas == ATLAS['atlas'][str(data.shape[0])]

        if self.sampling_time is None and sampling_time is not None:
            self.sampling_time = float(sampling_time)

        self.data.append(data)

    @staticmethod
    def normalize(data, direction=1, demean=True, destandard=True):
        """
        Normalize the original data set.

        :param x: [Array-like] data
        :param direction: [int] 0 for columns, 1 for rows, None for global
        :param demean: [boolean] demean
        :param destandard: [boolean] remove standard deviation (to 1)
        :return x: [Array-like] standardized data
        :return mean: [float] mean of original data
        :return std: [float] standard deviation of original data
        """

        # Handle data
        x = data.copy()
        x = np.asarray(x)
        assert isinstance(x, np.ndarray)

        # Fetch statistical information
        std = np.std(x, axis=direction)
        mean = np.mean(x, axis=direction)

        # removal of mean
        if demean:
            x -= mean.reshape((mean.shape[0]), 1)

        # removal of standard deviation
        if destandard:
            x /= std.reshape((std.shape[0]), 1)

        return x, mean, std

    @staticmethod
    def _adjust_phase(x):
        """
        Adjust phase as to have orthogonal vectors

        :param x: [Array-like] original vectors
        :return ox: [Array-like] orthogonal vectors
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

    @staticmethod
    def _match_modes(tc, s, m):
        """
        Match modes using Time Series data and eigenvectors of reference group.
        Please forgive me for the MATLAB-like code.

        :param tc: Time Series data
        :param s: matrix of column eigenvectors
        :param m: number of modes to analyse
        :return:
        """

        s_inv = la.inv(s)

        n, t = tc.shape

        b = np.empty([m, 1], dtype=complex)
        a = np.empty([m, m], dtype=complex)

        if ATLAS['atlas'][str(tc.shape[0])] != ATLAS['atlas'][str(s.shape[0])]:
            logging.error("ATLAS of reference and match group do not correspond.")

        t2, t1 = Decomposition._check_data([tc])
        t2 = t2.T
        t1 = t1.T

        for r in range(m):

            r1 = s[:, r].reshape(n, 1) @ s_inv[r, :].reshape(1, n)

            for c in range(r, m):

                c1 = s[:, c].reshape(n, 1) @ s_inv[c, :].reshape(1, n)

                if r != c:

                    M = (c1.T @ r1 + r1.T @ c1)

                    a[r, c] = t1.flatten() @ (t1 @ M.T).flatten()

                    a[c, r] = a[r, c]

                else:

                    a[r, c] = 2 * t1.flatten() @ (t1 @ r1.T @ r1).flatten()

            b[r] = 2 * t2.flatten() @ (t1 @ r1.T).flatten()

        d = la.solve(a, b)

        return np.around(d, decimals=8)

    def compute_match(self, other, m):
        """
        Get matched modes for match group with self (reference group).
        Predicts amplification of approximated modes using linear regression.

        :param other: [Decomposition]
        :param m: [int] number of modes approximated
        :return: [pd.DataFrame] containing 'mode', 'value', 'damping_time', 'period', 'conjugate'
        :return x: [Array-like] vector containing absolute value of top 10 approximated eigenvalues of self
        :return y: [Array-like] vector containing absolute value of real 10 eigenvalues of self
        """

        # First the modes should be matched with myself to get regression params
        logging.info('Fetching reference mode matching for regression parameter estimation.')

        borderline = self.eigVal[self.eigIdx][m].conj() == self.eigVal[self.eigIdx][m + 1]
        mm = (m + 1) if borderline else m

        own = self._match_modes(self.X, self.eigVec[:, self.eigIdx], mm)
        assert np.asarray(own).shape[0] == mm

        # Top 10 modes are used in the dashboard
        reg = LinearRegression().fit(np.abs(own).reshape(-1, 1), np.abs(self.eigVal[self.eigIdx][:mm]).reshape(-1, 1))

        logging.info('Regression parameters estimated.')
        logging.info('Fetching mode estimation for match group.')

        others = self._match_modes(other.X, self.eigVec[:, self.eigIdx], (m + 1) if borderline else m)

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
                    period=((2 * PI) / np.abs(np.angle(value))) * self.sampling_time if conj else np.inf,
                    conjugate=conj
                )
            )

            order += 1
            idx += 1 if not conj else 2

        return pd.DataFrame(modes), np.abs(own.flatten()), np.abs(self.eigVal[self.eigIdx][:mm])
