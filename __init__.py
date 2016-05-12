# -*- coding: utf-8 -*-
"""
Created on 5/10/2016

@author: david-hoffman
@copyright : David Hoffman

Package holding the necessary functions for generating SIM patterns for
the QXGA-3DM writing repz11 files and ini files for labVIEW
"""

import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
import numexpr as ne
try:
    from pyfftw.interfaces.numpy_fft import (ifftshift, fftshift,
                                             rfftn, irfftn)
    import pyfftw
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import ifftshift, fftshift, rfftn, irfftn
from dphutils import slice_maker
from .slm import *
# define pi for later use
pi = np.pi


# newest one
def pattern(angle, period, phi=0, onfrac=0.5, shape=(1536, 2048), SIM_2D=True):
    '''
    Generates a binary SLM pattern for SIM by generating a 2D sine wave
    and binarizing it

    Parameters
    ----------
    angle : float
        Defines the pattern orientation
    period : float
        Defines the period of the pattern
    phi : float
        phase of the pattern.
    onfrac : float
        The fraction of on pixels in the pattern
    shape : tuple
        shape of the pattern

    Returns
    -------
    pattern : ndarray
        A binary array representing a single bitplane to send to the SLM
    '''
    if not 0 < onfrac < 1:
        raise ValueError(('onfrac must have a value between'
                          ' 0 and 1. onfrac = {}').format(onfrac))
    if not 2 <= period <= max(shape):
        raise ValueError(('period must be larger than 2 (nyquist limit)'
                          ' and smaller than the size of the array (DC limit)'
                          '. period = {}').format(period))
    # generate grids
    yy, xx = np.indices(shape)
    # here's the pattern frequency
    freq = 2 * pi / period
    # our sine goes from -1 to 1 while onfrac goes from 0,1 so move onfrac
    # into the right range
    onfrac = onfrac * 2 - 1
    # do the evaluation
    toreturn = ne.evaluate("sin(freq*(cos(angle)*xx + "
                           "sin(angle)*yy)+phi) < onfrac")
    return toreturn


def localize_peak(data):
    """
    Small utility function to localize a peak center. Assumes passed data has
    peak at center and that data.shape is odd and symmetric. Then fits a
    parabola through each line passing through the center. This is optimized
    for FFT data which has a non-circularly symmetric shaped peaks.
    """
    # make sure passed data is symmetric along all dimensions
    assert len(set(data.shape)) == 1, "data.shape = {}".format(data.shape)
    # pull center location
    center = data.shape[0] // 2
    # generate the fitting lines
    my_pat_fft_suby = data[:, center]
    my_pat_fft_subx = data[center, :]
    # fit along lines, consider the center to be 0
    x = np.arange(data.shape[0]) - center
    xfit = np.polyfit(x, my_pat_fft_subx, 2)
    yfit = np.polyfit(x, my_pat_fft_suby, 2)
    # calculate center of each parabola
    x0 = -xfit[1] / (2 * xfit[0])
    y0 = -yfit[1] / (2 * yfit[0])
    # NOTE: comments below may be useful later.
    # save fits as poly functions
    # ypoly = np.poly1d(yfit)
    # xpoly = np.poly1d(xfit)
    # peak_value = ypoly(y0) / ypoly(0) * xpoly(x0)
    # #
    # assert np.isclose(peak_value,
    #                   xpoly(x0) / xpoly(0) * ypoly(y0))
    # return center
    return y0, x0


def pattern_params(my_pat, size=2):
    '''
    Find stuff
    '''
    # REAL FFT!
    # note the limited shifting, we don't want to shift the last axis
    my_pat_fft = ifftshift(rfftn(fftshift(my_pat)),
                           axes=tuple(range(my_pat.ndim))[:-1])
    my_abs_pat_fft = abs(my_pat_fft)
    # find dc loc, center of FFT after shifting
    sizeky, sizekx = my_abs_pat_fft.shape
    # remember we didn't shift the last axis!
    dc_loc = (sizeky // 2, 0)
    # mask data and find next biggest peak
    dc_power = my_abs_pat_fft[dc_loc]
    my_abs_pat_fft[dc_loc] = 0
    max_loc = np.unravel_index(my_abs_pat_fft.argmax(), my_abs_pat_fft.shape)
    # pull the 3x3 region around the peak and fit
    max_shift = localize_peak(my_abs_pat_fft[slice_maker(*max_loc, width=3)])
    # calculate precise peak relative to dc
    peak = np.array(max_loc) + np.array(max_shift) - np.array(dc_loc)
    # correct location based on initial data shape
    peak_corr = peak / np.array(my_pat.shape)
    # calc angle
    preciseangle = np.arctan2(*peak_corr)
    # calc period
    precise_period = 1 / norm(peak_corr)
    # calc phase
    phase = np.angle(my_pat_fft[max_loc[0], max_loc[1]])
    # calc modulation depth
    numerator = abs(my_pat_fft[slice_maker(*max_loc, width=size)].sum())
    mod = numerator / dc_power
    return {"period": precise_period,
            "angle": preciseangle,
            "phase": phase,
            "fft": my_pat_fft,
            "mod": mod,
            "max_loc": max_loc}


def opt_period(iperiod, angle, **kwargs):
    """
    Optimize actual period
    """
    def objf_l1(period):
        calc_period = pattern_params(angle, period, **kwargs)['period']
        return abs(calc_period - iperiod)
    return minimize(objf_l1, iperiod, method='Nelder-Mead')['x']


def opt_angle(period, iangle, **kwargs):
    """
    Optimize angle
    """
    def objf_l1(angle):
        calc_angle = pattern_params(angle, period, **kwargs)['angle']
        return abs(calc_angle - iangle)
    return minimize(objf_l1, iangle, method='Nelder-Mead')['x']


def make_angles(init_angle, num_angles):
    """
    Make a list of angles
    """
    thetas = np.arange(0., num_angles) * pi / 1. / num_angles + init_angle
    return thetas


def ideal_period(wavelength, na=0.85):
    '''
    Wavelength is in nm
    '''
    # all other units in mm
    # pixel size in mm for QXGA display (4DD)
    pixel_size = 8.2 / 1000
    # focal length of lens in mm
    fl = 250
    # focal length of the second lens
    fl2 = 300
    # focal length of the tube lens, for Nikon this is 200 mm
    ftube = 200
    # focal length of objective
    fobj = 2
    # wavelength of light
    wl = wavelength / 10**6
    mag = fobj / ftube
    # std dev of gaussian beam in units of pixels at the SLM
    # sigma = np.sqrt(2) * 12 / pixel_size / 4
    # Size of pupil image at first fourier plane
    pupil_diameter = 2 * na * mag * fl2
    # this is the limit of hole size
    # hole_radius = 2 * wl * fl / (2 * pi * sigma * np.sqrt(2) * pixel_size)
    # hole_radius = 0.1/2# this is more reasonable (50 um)
    # period = wl * fl * (1/(pupil_diameter/2 - hole_radius))/ pixel_size
    period = wl * fl / (pupil_diameter / 2) / pixel_size
    return period


class SIMRepertoire(object):
    """
    A class that takes care of the actual generation of images

    This is _not_ a subclass of slm.Repertoire, but does hold an instance
    """

    blank_bitplane = BitPlane(np.zeros((1536, 2048)), "Blank")

    def __init__(self, name, wls, nas, orders, norientations, seq,
                 SIM_2D=True):
        """
        Parameters
        ----------
        name : string
            name of the repertoire
        wls : numeric or tuple
            wavelengths to generate patterns for
        nas : numeric or tuple
            NAs to generate patterns for

        """
        # we have one sequence for now
        self.seq = seq
        # make new internal Repertoire to hold everything.
        self.rep = Repertoire(name)
        # what wavelengths to try
        self.wls = tuplify(wls)
        # what na's to try
        self.nas = tuplify(nas)
        # what non-linear orders to try
        self.orders = tuplify(orders)
        if min(self.orders) < 1:
            raise RuntimeError("Orders less than 0")
        # for now hard code, we have to double the phases
        # so we can do 90 deg phase stepping.
        if SIM_2D:
            # Then we only want to take steps of 2pi/n in illumination which
            # means pi/n at the SLM
            phase_step = 2
        else:
            phase_step = 1
        # calculate the number of phases needed
        self.nphases = np.prod(np.array(orders) * 2 + 1)
        # calculate actual phases
        self.phases = [(n / self.nphases / phase_step) * (2 * pi)
                       for n in range(self.nphases)]
        # if we're doing nonlinear, add flipped patterns
        if max(orders) > 1:
            self.phases += list(np.array(self.phases) + pi / 2)
        # make sure the proposed repertoire will fit
        if np.prod((len(self.nas), len(self.wls),
                    norientations + 1, len(self.phases))) + 1 > 1024:
            raise RuntimeError("These settings will generate too many")
        # For the current microscope we can only have one set of orientation
        if norientations == 3:
            init_angle = np.deg2rad(11.6)
        elif norientations == 5:
            init_angle = np.deg2rad(9.0)
        elif norientations == 7:
            init_angle = np.deg2rad(0.9)
        else:
            raise RuntimeError("number of orientations not valid, ",
                               norientations)
        self.angles = make_angles(init_angle, norientations)
        # once we have all this info we can start making bitplanes
        self.make_bitplanes()

    def clear_rep(self):
        # make new internal Repertoire to hold everything.
        self.rep = Repertoire(self.name)

    def make_bitplanes(self):
        """
        Function that returns a dictionary of dictionary of bitplanes
        of BitPlanes
        """
        # first level is wl
        self.bitplanes = {wl: {
            na: [
                [BitPlane(pattern(ang, ideal_period(wl, na), phi),
                          gen_name(ang, wl, na, n))
                 for n, phi in enumerate(self.phases)] for ang in self.angles]
            for na in self.nas
        } for wl in self.wls}

    def make_ROs(self):
        for wl, na_dict in self.bitplanes.items():
            for na, angle_list in na_dict.items():
                self.gen_sims(wl, na, angle_list)
                self.gen_all_angles(wl, na, angle_list)

    def gen_sims(self, wl, na, angle_list):
        """
        Do linear and nonlinear SIMs here
        """
        name_str = ("{} nm ".format(wl) +
                    "{} phases " +
                    "{:.2f} NA ".format(na) +
                    "{} SIM")
        for order in self.orders:
            num_phases = order * 2 + 1
            d = self.nphases // num_phases
            if order == 1:
                # linear SIM case
                frames = [Frame(self.seq, phase_bp, looped, True)
                          for phase_list in angle_list
                          for phase_bp in phase_list[:self.nphases:d]
                          for looped in (False, True)]
                RO_name = name_str.format(num_phases, "Linear")
            else:
                # non-linear SIM case
                frames = []
                for phase_list in angle_list:
                    off_phases = phase_list[self.nphases::d]
                    on_phases = phase_list[:self.nphases:d]
                    assert len(off_phases) == len(on_phases)
                    reactivation = [self.blank_bitplane] * len(off_phases)
                    frames.extend([Frame(self.seq, phase_bp, looped, True)
                                   for series in zip(reactivation, off_phases,
                                                     on_phases)
                                   for phase_bp in series
                                   for looped in (False, True)])
                RO_name = name_str.format(num_phases, "Non-Linear")
            RO = RunningOrder(RO_name, frames)
            self.rep.addRO(RO)
        # make the super sim here
        super_frames = [Frame(self.seq, phase_bp, looped, True)
                        for phase_list in angle_list
                        for phase_bp in phase_list[:self.nphases]
                        for looped in (False, True)]
        RO_super_name = name_str.format(self.nphases, "Super")
        RO_super = RunningOrder(RO_super_name, super_frames)
        self.rep.addRO(RO_super)
        # make single orientations
        for angle, phase_list in zip(self.angles, angle_list):
            RO_name = name_str.format(
                1, "{:.1f} angle".format(np.rad2deg(angle)))
            self.rep.addRO(RunningOrder(
                RO_name, Frame(self.seq, phase_list[0], True, False)))

    def gen_all_angles(self, wl, na, angle_list):
        """
        Makes a RunningOrder to display all angles at once
        """
        # make an array of the first phase of the data
        data_array = np.array([
            ang[0].image for ang in angle_list
        ])
        # take the mean and threshold at 0.5 (binarize)
        bitplane = data_array.mean(0) > 0.5
        # generate names
        bp_name = "AllAngles-{}nm-{:.2f}NA".format(wl, na)
        RO_name = "All Angles {} nm {:.2f} NA".format(wl, na)
        # generate bitplane
        all_angles_bitplane = BitPlane(bitplane, bp_name)
        # generate Frame
        frame = Frame(self.seq, all_angles_bitplane, True, False)
        # make RO and add to list
        self.rep.addRO(RunningOrder(RO_name, frame))

    @property
    def name(self):
        return self.rep.name

    def write(self, path=""):
        """
        Write the internal repertoire to a repz11 file
        """
        self.rep.write_repz11(path)

    def make_sim_frame_list(self, phase_list):
        return ((self.seq, phase_bp, looped, True)
                for series in tuplify(phase_list)
                for phase_bp in series
                for looped in (False, True))


def gen_name(angle, wl, na, n):
    """
    Generate a unique name for a BitPlane
    """
    degree = np.rad2deg(angle)
    my_per = ideal_period(wl, na)
    name = 'pat-{}nm-{:.2f}NA{:+.1f}deg-{:02d}ph-{:.4f}pix-{:.2f}DC'
    return name.format(wl, na, degree, n, my_per, 0.5)
