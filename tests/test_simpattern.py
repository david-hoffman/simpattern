from nose.tools import *
import os
import numpy as np
import unittest

from simpattern import *


def test_localize():
    """
    test localize, make sure it finds peak in right place.
    """
    pat = np.zeros((3, 3))
    pat[1, 1] = 1
    y0, x0 = localize_peak(pat)
    assert_almost_equals(0, y0, x0)


def test_localize2():
    """
    test localize, more complicated
    """
    x = np.arange(-1, 2)
    xr = -0.2
    yr = 0.6
    # make fake peak that is dot of two parabolas
    pat = (-(x - yr)**2).reshape(-1, 1) * (-(x - xr)**2)
    y0, x0 = localize_peak(pat)
    assert_almost_equal(yr, y0)
    assert_almost_equal(xr, x0)


def test_pattern_return_bool():
    """
    test return type
    """
    pat = pattern(0, 10)
    assert_equal(pat.dtype, np.bool)


def test_pattern_onfrac_error():
    """
    test correct error throwing
    """
    assert_raises(ValueError, pattern, 0, 10, onfrac=1)
    assert_raises(ValueError, pattern, 0, 10, onfrac=0)
    assert_raises(ValueError, pattern, 0, 10, onfrac=2)
    assert_raises(ValueError, pattern, 0, 10, onfrac=-1)


def test_pattern_period_error():
    """
    test correct error throwing
    """
    assert_raises(ValueError, pattern, 0, 1)
    assert_raises(ValueError, pattern, 0, 10000)
    assert_raises(ValueError, pattern, 0, -1)
    assert_raises(ValueError, pattern, 0, 0)
    # test edge case
    pattern(0, 2)

# def test_onfrac(self):
#     """
#     test whether the onfrac is the mean of the values
#     """
#     onfrac = np.random.random_sample() * 0.75 + 0.25
#     pat = pattern(0, 10, onfrac=onfrac)
#     assert_almost_equal(onfrac, pat.mean(), places=1)


def test_pattern_angles():
    """
    ensure that the requested pattern angle is close to the requested one.
    """
    angles = np.linspace(-np.pi / 2, np.pi / 2, 10) * 0.99
    for angle in angles:
        pat = pattern(angle, 10)
        param = pattern_params(pat)
        assert_almost_equal(angle, param['angle'], 2)


def test_pattern_period():
    """
    Ensure that the requested pattern period is close to the actual one
    """
    # periods close to the DC peak can't be accurately measured.
    periods = np.linspace(2, 512, 10)
    for period in periods:
        pat = pattern(np.pi / 4, period)
        param = pattern_params(pat)
        assert np.isclose(period, param['period'], 1e-1), (
            "{} != {}".format(period, param['period'])
        )
