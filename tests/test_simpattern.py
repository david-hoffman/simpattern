from nose.tools import *
import os
import numpy as np
import unittest

from simpattern import *
from simpattern.slm import *


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
    pat = (-(x - yr) ** 2).reshape(-1, 1) * (-(x - xr) ** 2)
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
        assert_almost_equal(angle, param["angle"], 2)


def test_pattern_period():
    """
    Ensure that the requested pattern period is close to the actual one
    """
    # periods close to the DC peak can't be accurately measured.
    periods = np.linspace(2, 512, 10)
    for period in periods:
        pat = pattern(np.pi / 4, period)
        param = pattern_params(pat)
        assert np.isclose(period, param["period"], 1e-1), "{} != {}".format(
            period, param["period"]
        )


class TestSIMRepertoire(unittest.TestCase):
    """
    A class to test the SIMRepertoire class and internal functions.
    """

    def setUp(self):
        """
        Set up an internal rep.
        """
        self.seq = Sequence(
            os.path.join(os.path.dirname(__file__), "..", "HHMI_R11_Seq", "48070 HHMI 10ms.seq11")
        )
        self.simrep = SIMRepertoire("dummyrep", 488, 0.85, 2, 3, self.seq)

    def check_tuples(self):
        """
        Make sure entered values that should be tuples are
        """
        assert_equal(self.simrep.wls, (488,))
        assert_equal(self.simrep.nas, (0.85,))
        assert_equal(self.simrep.orders, (2,))

    def test_make_sim_frame_list_single(self):
        """
        Make sure make_sim_frame_list works with single entries
        """
        simrep = self.simrep
        output = simrep.make_sim_frame_list(0)
        output_true = [(self.seq, 0, False, True, False), (self.seq, 0, True, False, True)]
        assert_equal(output, output_true)

    def test_make_sim_frame_list_multiple(self):
        """
        Make sure make_sim_frame_list works with multiple entries
        """
        simrep = self.simrep
        phase_list = (0, 1, 2)
        output = simrep.make_sim_frame_list(phase_list)
        output_true = [
            (self.seq, 0, False, True, False),
            (self.seq, 0, True, False, True),
            (self.seq, 1, False, True, False),
            (self.seq, 1, True, False, True),
            (self.seq, 2, False, True, False),
            (self.seq, 2, True, False, True),
        ]
        assert_equal(output, output_true)

    def test_make_sim_frame_list_zip(self):
        """
        Make sure make_sim_frame_list works with zipped entries
        Implicitly shows it works with generators as expected.
        """
        simrep = self.simrep
        phase_list1 = ((0, 0), (0, 1))
        phase_list2 = ((1, 0), (1, 1))
        output = simrep.make_sim_frame_list(zip(phase_list1, phase_list2))
        output_true = [
            (self.seq, (0, 0), False, True, False),
            (self.seq, (0, 0), True, False, True),
            (self.seq, (1, 0), False, True, False),
            (self.seq, (1, 0), True, False, True),
            (self.seq, (0, 1), False, True, False),
            (self.seq, (0, 1), True, False, True),
            (self.seq, (1, 1), False, True, False),
            (self.seq, (1, 1), True, False, True),
        ]
        assert_equal(output, output_true)
