"""
Test the core module
"""

import pytest

import scipy.io as spio
import numpy as np
from pathlib import *

from nidmd.core import (TimeSeries)


def test_phase_adjustment():

    # TODO

    complexes = np.random.rand(5, 5) + 1j * np.random.rand(5, 5)


def test_split():

    ar = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ]
    )

    # For a simple array

    x, y = TimeSeries.split(ar, normalize=False)

    assert np.all(x[:,  0] == 2)
    assert np.all(x[:, -1] == 9)

    assert np.all(y[:,  0] == 1)
    assert np.all(y[:, -1] == 8)

    # For a list of arrays

    xl, yl = TimeSeries.split([ar, ar], normalize=False)

    assert np.all(xl[:,  0] == 2)
    assert np.all(xl[:,  7] == 9)
    assert np.all(xl[:,  8] == 2)
    assert np.all(xl[:, -1] == 9)

    assert np.all(yl[:,  0] == 1)
    assert np.all(yl[:,  7] == 8)
    assert np.all(yl[:,  8] == 1)
    assert np.all(yl[:, -1] == 8)

    pytest.raises(ValueError, TimeSeries.split, 1)
    pytest.raises(ValueError, TimeSeries.split, dict(name='split'))


def test_extraction_mat():

    path_glasser = Path(__file__).parent.joinpath('data/glasser.mat').as_posix()
    path_schaefer = Path(__file__).parent.joinpath('data/schaefer.mat').as_posix()

    ts = TimeSeries()
    ts.extract(path_glasser)

    assert(np.asarray(spio.loadmat(path_glasser)['TCSnf'][12][12]) == ts.data[0][12][12])

    ts2 = TimeSeries()
    ts2.extract(path_schaefer)

    assert(np.asarray(spio.loadmat(path_schaefer)['TS'][12][12]) == ts2.data[0][12][12])


def test_extraction_csv():

    path_glasser = Path(__file__).parent.joinpath('data/glasser.csv').as_posix()
    path_schaefer = Path(__file__).parent.joinpath('data/schaefer.csv').as_posix()

    ts = TimeSeries()
    ts.extract(path_glasser)

    assert(np.genfromtxt(path_glasser, delimiter=",")[12][12] == ts.data[0][12][12])

    ts2 = TimeSeries()
    ts2.extract(path_schaefer)

    assert(np.genfromtxt(path_schaefer, delimiter=",")[12][12] == ts2.data[0][12][12])


def test_normalisation():

    mat = np.random.rand(5, 5)

    # Check directions

    normalized, _, _ = TimeSeries.normalize(mat)

    assert np.all(
        np.mean(normalized, axis=1).round(10) == 0
    )

    assert np.all(
        np.std(normalized, axis=1).round(10) == 1.0
    )

    normalized0, _, _ = TimeSeries.normalize(mat, direction=0)

    assert np.all(
        np.mean(normalized0, axis=0).round(10) == 0
    )

    assert np.all(
        np.std(normalized0, axis=0).round(10) == 1
    )

    normalized_none, _, _ = TimeSeries.normalize(mat, direction=None)

    assert np.all(
        np.mean(normalized_none).round(10) == 0
    )

    assert np.all(
        np.std(normalized_none).round(10) == 1
    )

    # Check demean

    normalized_mean, _, _ = TimeSeries.normalize(mat, demean=False)

    assert np.all(
        np.mean(normalized_mean, axis=1).round(10) != 0
    )

    assert np.all(
        np.std(normalized_mean, axis=1).round(10) == 1
    )

    # Check destandard

    normalized_std, _, _ = TimeSeries.normalize(mat, destandard=False)

    assert np.all(
        np.mean(normalized_std, axis=1).round(10) == 0
    )

    assert np.all(
        np.std(normalized_std, axis=1).round(10) != 1
    )
