# -*- coding: utf-8 -*-
"""
Created on 5/10/2016

@author: david-hoffman
@copyright : David Hoffman

Package holding the necessary functions for generating SIM patterns for
the QXGA-3DM writing repz11 files and ini files for labVIEW
"""

import os
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
import numexpr as ne
try:
    from pyfftw.interfaces.numpy_fft import (fftshift, ifftshift,
                                             rfftn, irfftn)
    import pyfftw
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import fftshift, ifftshift, rfftn, irfftn
from dphutils import slice_maker
from .slm import (Sequence, Repertoire, RunningOrder, Frame,
                  BitPlane, BitPlane24, tuplify)
# define pi for later use
pi = np.pi

# define QXGA shape
QXGA_shape = (1536, 2048)

# define local sequences, for easy acces
seq_home = os.path.join(os.path.dirname(__file__), "HHMI_R11_Seq", "")
seq_10ms = Sequence(seq_home + "48070 HHMI 10ms.seq11")
seq_50ms = Sequence(seq_home + "48071 HHMI 50ms.seq11")
seq_5ms = Sequence(seq_home + "48075 HHMI 5ms.seq11")
seq_20ms = Sequence(seq_home + "48076 HHMI 20ms.seq11")
seq_24_50ms = Sequence(seq_home + "48077 HHMI 24 50ms.seq11")
seq_24_1ms = Sequence(seq_home + "48078 HHMI 24 1ms.seq11")
seq_1ms_LB = Sequence(seq_home + "48083 HHMI 1ms 1-bit Lit Balanced.seq11")
seq_2ms_LB = Sequence(seq_home + "48084 HHMI 2ms 1-bit Lit Balanced.seq11")


mcf_dict = {
    405: {
        "frequency": 151.479,
        "maxpower": 22.5
    },
    445: {
        "frequency": 151.980,
        "maxpower": 22.4
    },
    488: {
        "frequency": 130.202,
        "maxpower": 16.7
    },
    532: {
        "frequency": 115.488,
        "maxpower": 18.0
    },
    561: {
        "frequency": 108.000,
        "maxpower": 20.0
    },
    642: {
        "frequency": 90.985,
        "maxpower": 20.5
    }
}


def pattern(angle, period, phi=0, onfrac=0.5, shape=QXGA_shape):
    """Generates a binary SLM pattern for SIM by generating a 2D sine wave
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
    """
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
    """Small utility function to localize a peak center. Assumes passed data has
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
    """Find stuff"""
    # REAL FFT!
    # note the limited shifting, we don't want to shift the last axis
    my_pat_fft = fftshift(rfftn(ifftshift(my_pat)),
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
    max_shift = localize_peak(my_abs_pat_fft[slice_maker(max_loc, 3)])
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
    numerator = abs(my_pat_fft[slice_maker(max_loc, size)].sum())
    mod = numerator / dc_power
    return {"period": precise_period,
            "angle": preciseangle,
            "phase": phase,
            "fft": my_pat_fft,
            "mod": mod,
            "max_loc": max_loc}


def opt_period(iperiod, angle, **kwargs):
    """Optimize actual period"""
    def objf_l1(period):
        calc_period = pattern_params(angle, period, **kwargs)['period']
        return abs(calc_period - iperiod)
    return minimize(objf_l1, iperiod, method='Nelder-Mead')['x']


def opt_angle(period, iangle, **kwargs):
    """Optimize angle"""
    def objf_l1(angle):
        calc_angle = pattern_params(angle, period, **kwargs)['angle']
        return abs(calc_angle - iangle)
    return minimize(objf_l1, iangle, method='Nelder-Mead')['x']


def make_angles(init_angle, num_angles):
    """Make a list of angles"""
    thetas = np.arange(0., num_angles) * pi / 1. / num_angles + init_angle
    return thetas


def ideal_period(wavelength, na=0.85):
    """Wavelength is in nm"""
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


class ExptRepertoire(object):
    """A class that takes care of the actual generation of images

    This is _not_ a subclass of slm.Repertoire, but does hold an instance
    """

    # this one is for NL SIM
    blank_bitplane = BitPlane(np.zeros(QXGA_shape, dtype=bool), "Blank")
    blank_bitplane24 = BitPlane24(np.zeros((24,) + QXGA_shape, dtype=bool), "Blank24")

    blank_RO = RunningOrder(
            "Blank",
            Frame(seq_24_50ms, blank_bitplane24, True, False, False)
        )

    blank_RO_triggered = RunningOrder(
            "Blank Triggered",
            [Frame(seq_24_50ms, blank_bitplane24, False, True, False),
             Frame(seq_24_50ms, blank_bitplane24, True, True, False)]
        )

    def __init__(self, name, wls, nas, phases, norientations, seq, onfrac=0.5):
        """
        Parameters
        ----------
        name : string
            name of the repertoire
        wls : numeric or tuple
            wavelengths to generate patterns for
        nas : numeric or tuple
            NAs to generate patterns for
        orders : numeric or tuple
            Number of orders to generate patterns for, needed for NL
        norientations : int
            number of pattern orientations
        seq : slm.Sequence object
            The sequence to use for this repertoire.
        onfrac : float
            The fraction of "on" pixels. onfrac = 0.5 + DC / 2
            where DC is the DC offset of the pattern
        """
        # we have one sequence for now
        self.seq = seq
        # save ndirs
        self.ndirs = norientations
        # make new internal Repertoire to hold everything.
        self.rep = Repertoire(name)
        # what wavelengths to try
        self.wls = tuplify(wls)
        # what na's to try
        self.nas = tuplify(nas)
        self.onfrac = onfrac
        # calculate actual phases
        self.phases = phases
        # if we're doing nonlinear, add flipped patterns
        # make sure the proposed repertoire will fit
        num_bitplanes = len(self.nas)
        num_bitplanes *= len(self.wls)
        num_bitplanes *= norientations
        # all phases will be used for super-NL running order
        num_bitplanes *= len(self.phases)
        num_bitplanes += len(self.wls) + 1
        if num_bitplanes > 1024:
            raise RuntimeError(
                ("These settings will generate {} "
                 "bitplanes which is more than 1024").format(num_bitplanes)
            )
        else:
            print("Generating {} bitplanes".format(num_bitplanes))
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
        # add a blank bit plane
        self.rep.addRO(self.blank_RO)
        self.rep.addRO(self.blank_RO_triggered)

    def make_bitplanes(self):
        """Function that returns a dictionary of dictionary of bitplanes
        of BitPlanes
        """
        # first level is wl
        self.bitplanes = {wl: {
            na: [
                [BitPlane(pattern(ang, ideal_period(wl, na), phi, onfrac=self.onfrac),
                          _gen_name(ang, wl, na, n, self.onfrac))
                 for n, phi in enumerate(self.phases)] for ang in self.angles]
            for na in self.nas
        } for wl in self.wls}

    def make_ROs(self):
        raise NotImplementedError

    @property
    def name(self):
        return self.rep.name

    def _write_all(self, path=""):
        """Write the internal repertoire to a repz11 file"""
        print("Writing .repz11")
        self.rep.write_repz11(path)

    def __call__(self, path=""):
        """Do everything and write it out"""
        self.make_ROs()
        self._write_all(path)


class SIMRepertoire(ExptRepertoire):
    """A class that takes care of the actual generation of images

    This is _not_ a subclass of slm.Repertoire, but does hold an instance
    """

    linear_str = "\n".join([
        '[Sequence setpoints:{ROname}]',
        'Running Order = {RO_num:d}',
        'N Phases = {nphases:d}',
        'Detection Mode = "Camera 1"',
        'Pixel size (um) = "0.13, 0.13"',
        'Filter Wheel 1 = "Filter {filter:d}"',
        'Filter Wheel 2 = "Filter 0"',
        r'File Index = "0,-1\0D\0A"',
        'Setpoints:Galvo = "{galvo:}"',
        ('Setpoints:Step 1 = "Laser[{wl:d} nm]; LC [{lc:}];'
         ' Delay [0]; Camera[0]; Imaging[TRUE];'
         ' AOTF[ 100 {wl:d} nm(X);   0 532 nm(X);   0 560 nm(X);  0 405 nm(X)];'
         ' BeamBlock[No change]"')
    ])

    nonlinear_str = "\n".join([
        '[Sequence setpoints:{ROname}]',
        'Running Order = {RO_num:d}',
        'N Phases = {nphases:d}',
        'Detection Mode = "Camera 1"',
        'Pixel size (um) = "0.13, 0.13"',
        'Filter Wheel 1 = "Filter {filter:d}"',
        'Filter Wheel 2 = "Filter 0"',
        r'File Index = "0,-1\0D\0A1,-1\0D\0A2,-1\0D\0A"',
        'Setpoints:Galvo = "{galvo:}"',
        ('Setpoints:Step 1 = "Laser[405 nm]; LC [{galvo:}];'
         ' Delay [100]; Camera[0]; Imaging[FALSE];'
         ' AOTF[   0 {wl:d} nm(X);   0 488 nm(X);   0 560 nm(X); 100 405 nm(X)];'
         ' BeamBlock[Out]"'),
        ('Setpoints:Step 2 = "Laser[{wl:d} nm]; LC [{lc:}];'
         ' Delay [100]; Camera[0]; Imaging[FALSE];'
         ' AOTF[ 100 {wl:d} nm(X);   0 532 nm(X);   0 560 nm(X);  0 405 nm(X)];'
         ' BeamBlock[In]"'),
        ('Setpoints:Step 3 = "Laser[{wl:d} nm]; LC [{lc:}];'
         ' Delay [0]; Camera[0]; Imaging[TRUE];'
         ' AOTF[ 100 {wl:d} nm(X);   0 532 nm(X);   0 560 nm(X);  0 405 nm(X)];'
         ' BeamBlock[In]"')
    ])

    react_all_str = "\n".join([
        '[Sequence setpoints:{ROname}]',
        'Running Order = {RO_num:d}',
        'N Phases = {nphases:d}',
        'Detection Mode = "Camera 1"',
        'Pixel size (um) = "0.13, 0.13"',
        'Filter Wheel 1 = "Filter {filter:d}"',
        'Filter Wheel 2 = "Filter 0"',
        r'File Index = "0,-1\0D\0A1,-1\0D\0A2,-1\0D\0A"',
        'Setpoints:Galvo = "{galvo:}"',
        ('Setpoints:Step 1 = "Laser[405 nm]; LC [{galvo:}];'
         ' Delay [100]; Camera[0]; Imaging[FALSE];'
         ' AOTF[   0 {wl:d} nm(X);   0 488 nm(X);   0 560 nm(X); 100 405 nm(X)];'
         ' BeamBlock[No change]"'),
        ('Setpoints:Step 2 = "Laser[{wl:d} nm]; LC [{lc:}];'
         ' Delay [100]; Camera[0]; Imaging[True];'
         ' AOTF[ 100 {wl:d} nm(X);   0 532 nm(X);   0 560 nm(X);  0 405 nm(X)];'
         ' BeamBlock[No change]"')
    ])

    linear_w_react_str = "\n".join([
            '[Multiscan:{ROname}]',
            'Scan Def = "{ROname_sub},405 nm WF\\0D\\0A"',
            'Running Order = "{RO_num:d}"',
            r'N Times = "0,0\0D\0A"',
            r'Nth Cycle = "1,1\0D\0A"',
            'Interleave = TRUE'
        ])

    def __init__(self, name, wls, nas, orders, norientations, seq, onfrac=0.5,
                 super_repeats=1, SIM_2D=True, doNL=True, super_mult=0):
        """
        Parameters
        ----------
        name : string
            name of the repertoire
        wls : numeric or tuple
            wavelengths to generate patterns for
        nas : numeric or tuple
            NAs to generate patterns for
        orders : numeric or tuple
            Number of orders to generate patterns for, needed for NL
        norientations : int
            number of pattern orientations
        seq : slm.Sequence object
            The sequence to use for this repertoire.
        onfrac : float
            The fraction of "on" pixels. onfrac = 0.5 + DC / 2
            where DC is the DC offset of the pattern
        super_repeats : int
            Number of times to repeat the superSIM pattern
        SIM_2D : bool
            Are these 2D patterns?
        doNL : bool
            Are these nonlinear patterns?
        super_mult : int
            How many extra phases do you want? As a multiple of the phases
            you need. Useful for when extra orders aren't required (No NL)
            or when you want to restrict the number of extra phases.
        """
        # what non-linear orders to try
        self.orders = tuplify(orders)
        if min(self.orders) < 1:
            raise RuntimeError("Orders less than 0")
        self.repeats = super_repeats
        self.do_nl = doNL
        if SIM_2D:
            # Then we only want to take steps of 2pi/n in illumination which
            # means pi/n at the SLM
            phase_step = 2
            # extra orders is for how many orders we have to use
            extra_orders = 1
            self.np_base = 1
        else:
            phase_step = 1
            extra_orders = 3
            self.np_base = 3
        # calculate the number of phases needed
        self.nphases = np.prod(np.array(orders) * 2 + extra_orders)
        # expand for phase fitting if wanted
        if super_mult:
            self.nphases *= super_mult
        print("number of phases =", self.nphases)
        # calculate actual phases
        phases = [(n / self.nphases / phase_step) * (2 * pi)
                       for n in range(self.nphases)]
        # if we're doing nonlinear, add flipped patterns
        if max(self.orders) > 1 and self.do_nl:
            print("generating non-linear phases")
            phases += list(np.array(phases) + pi / 2)
        # initialize the parent class
        super().__init__(name, wls, nas, phases, norientations, seq, onfrac)
        # add bitplane with timing
        self.blankwtiming = [
            Frame(self.seq, self.blank_bitplane, False, True, False),
            Frame(self.seq, self.blank_bitplane, True, False, True)
        ]

        RO405 = RunningOrder("405 nm WF", self.blankwtiming)
        RO405.wl = 405
        RO405.nphases = 1
        RO405.linear = True
        self.rep.addRO(RO405)

    def make_ROs(self):
        """Sub-method that makes the running orders and adds them to the
        Repertoire.
        """
        for wl, na_dict in self.bitplanes.items():
            for na, angle_list in na_dict.items():
                self.gen_sims(wl, na, angle_list)
                self.gen_all_angles(wl, na, angle_list)

    def gen_sims(self, wl, na, angle_list):
        """Do linear and nonlinear SIMs here"""
        # define the naming convention for ROs
        name_str = ("{} nm ".format(wl) +
                    "{} phases " +
                    "{:.2f} NA ".format(na) +
                    "{} SIM")
        # loop through the orders
        for order in self.orders:
            # calculate the number of phases
            # Calculate the offset of bitplanes
            for reps, num_phases, ext_str, orients in zip(
                    (1, 1, 1, 1, self.repeats),
                    # here we need to add the order
                    (self.np_base + order * 2, self.np_base + order * 2, self.np_base + order * 2, self.np_base + order * 2, self.nphases),
                    ("", "React ", "React_All ", "Single Orientation ", "Super "),
                    (slice(None), slice(None), slice(None), slice(1), slice(None))):
                delta = self.nphases // num_phases
                if order == 1:
                    # linear sim case, super and regular
                    series_list = [
                        phase_bp
                        for phase_list in angle_list[orients]
                        for phase_bp in phase_list[:self.nphases:delta] * reps
                    ]
                    if ext_str == "React_All ":
                        series_list = [(self.blank_bitplane, phase_bp) for phase_bp in series_list]
                    else:
                        series_list = [(phase_bp, ) for phase_bp in series_list]
                    RO_name = name_str.format(num_phases, ext_str + "Linear")
                elif self.do_nl:
                    # non-linear SIM case
                    series_list = []
                    for phase_list in angle_list:
                        off_phases = phase_list[self.nphases::delta] * reps
                        on_phases = phase_list[:self.nphases:delta] * reps
                        assert len(off_phases) == len(on_phases)
                        reactivation = [self.blank_bitplane] * len(off_phases)
                        series_list.extend([
                            phase_bp
                            for series in zip(reactivation, off_phases, on_phases)
                            for phase_bp in series
                        ])
                    RO_name = name_str.format(num_phases, ext_str +
                                              "Non-Linear")
                else:
                    # order greater than 1 and no nonlinear requested
                    continue
                print('Generating "' + RO_name + '"')
                frames = [
                    Frame(self.seq, phase_bp, looped, triggered, finish)
                    for series in series_list
                    for phase_bp in series
                    for looped, triggered, finish in zip((False, True), (True, False), (False, True))
                ]
                if ext_str == "React ":
                    frames += self.blankwtiming
                # the number of frames is double what one would expect because
                # there's one frame to trigger the start and one to loop until
                # triggered to finish
                RO = RunningOrder(RO_name, frames)
                # tag the RO for writing the INI file later
                if order == 1:
                    RO.linear = True
                else:
                    RO.linear = False
                RO.nphases = num_phases
                RO.wl = wl
                # add the RO to the rep
                self.rep.addRO(RO)
        ###################################################################
        # make single orientations
        ###################################################################
        for angle, phase_list in zip(self.angles, angle_list):
            RO_name = name_str.format(
                1, "{:.1f} angle".format(np.rad2deg(angle)))
            self.rep.addRO(RunningOrder(
                RO_name, Frame(self.seq, phase_list[0], True, False, False)))
            print('Generating "' + RO_name + '"')

    def gen_all_angles(self, wl, na, angle_list):
        """Makes a RunningOrder to display all angles at once"""
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
        frame = Frame(self.seq, all_angles_bitplane, True, False, False)
        # make RO and add to list
        allro = RunningOrder(RO_name, frame)
        allro.wl = wl
        self.rep.addRO(allro)
        print('Generating "' + RO_name + '"')

    def _write_all(self, path=""):
        """Write the internal repertoire to a repz11 file"""
        super()._write_all(path)
        self._write_ini(path)
        self._write_mcf_and_tsv(path)

    def _write_ini(self, path):
        """Method to write the INI file for LabVIEW"""
        # pull rep out for ease of use
        print("Writing .ini")
        rep = self.rep
        filename = os.path.join(path, rep.name + ".ini")
        # open file
        with open(filename, "w") as file:
            for i, RO in enumerate(rep):
                try:
                    if "Single Orientation" in RO.name or "WF" in RO.name:
                        ndirs = 1
                    else:
                        ndirs = self.ndirs
                    if RO.linear:
                        if "React " in RO.name:
                            str2write = self.linear_w_react_str
                        elif "React_All " in RO.name:
                            str2write = self.react_all_str
                        else:
                            str2write = self.linear_str
                    else:
                        str2write = self.nonlinear_str
                    # now format
                    if RO.wl == 488:
                        lc_start = 1
                        filter_num = 1
                    elif RO.wl == 405:
                        lc_start = 0
                        filter_num = 1
                    elif RO.wl == 560:
                        lc_start = 1 + self.ndirs
                        filter_num = 0
                    elif RO.wl == 532:
                        lc_start = 1 + 2 * self.ndirs
                        filter_num = 2
                    elif RO.wl == 642:
                        lc_start = 1 + 3 * self.ndirs
                        filter_num = 3
                    else:
                        lc_start = -self.ndirs
                    lc = ",".join([
                        str(i)
                        for i in range(lc_start, lc_start + ndirs)
                    ])
                    str_dict = dict(
                        wl=RO.wl,
                        nphases=RO.nphases,
                        ROname=RO.name,
                        ROname_sub=RO.name.replace("React ", ""),
                        RO_num=i,
                        filter=filter_num,
                        lc=lc,
                        galvo=",".join("0" * ndirs)
                    )
                    file.write(str2write.format(
                        **str_dict
                    ))
                    file.write("\n\n")
                except AttributeError:
                    pass

    def _write_mcf_and_tsv(self, path=""):
        """dfds"""

        print("Writing .mcf and .tsv")
        mcf_entry = ("[Line{num:}]\n"
                     "Frequency={frequency:3f}\n"
                     "WaveLength={wl:}\n"
                     "Power={maxpower:.1f}\n"
                     "MaxPower={maxpower:.1f}\n"
                     "Title={title:}\n")

        # iterate through ROs
        mcfname = os.path.join(path, self.rep.name + ".mcf")
        tsvname = os.path.join(path, self.rep.name + ".tsv")
        abspath = os.path.abspath("")
        with open(mcfname, 'w') as mcf, open(tsvname, 'w') as tsv:
            for i, ro in enumerate(self.rep):
                if "NA Linear SIM" in ro.name:
                    # reorder frames into orientations
                    frame_names = np.array([frame.bitplanes[0].name for frame in ro])[::2].reshape(-1, 5)
                    for j, orient in enumerate(frame_names):
                        # there should be 3 of these
                        # write a line in the MCF file
                        tsvline = "\t".join([os.path.join(abspath, self.rep.name, bp + ".bmp") for bp in orient])
                        tsv.write(tsvline + "\n")
                        title = ro.name + " {}".format(j)
                        mcfline = mcf_entry.format(num="{}{}".format(i,j), wl=ro.wl, title=title, **mcf_dict[ro.wl])
                        mcf.write(mcfline)
                elif "All Angles" in ro.name:
                    # Here we make blank an dall
                    bmp1 = ro.frames[0].bitplanes[0].name + ".bmp"
                    bmp2 = self.blank_bitplane.name + ".bmp"
                    for name, filename in zip(("All", "Blank"), (bmp1, bmp2)):
                        tsvline = "\t".join([os.path.join(abspath, self.rep.name, filename)] * 5)
                        tsv.write(tsvline + "\n")
                        title = "{} nm {}".format(ro.wl, name)
                        mcfline = mcf_entry.format(num=i, wl=ro.wl, title=title, **mcf_dict[ro.wl])
                        mcf.write(mcfline)

    def make_sim_frame_list(self, series):
        """Utility function that interleaves a list of bitplanes
        such that there's one single triggered version followed
        by a looped version that has a triggered ending.
        """
        return [(self.seq, phase_bp, looped, triggered, finish)
                for phase_list in tuplify(series)
                for phase_bp in tuplify(phase_list)
                for looped, triggered, finish in zip((False, True), (True, False), (False, True))]


class PALMRepertoire(ExptRepertoire):
    """Subclass of ExptRepertoire to generate reps for PALM

    The problem we're having is that we're burning the objectives
    by spinning things around in the back pupil or spreading the
    energy out can we avoid this ..."""

    def __init__(self, name, wls, nas, seq):
        """
        Parameters
        ----------
        name : string
            name of the repertoire
        wls : numeric or tuple
            wavelengths to generate patterns for
        nas : numeric or tuple
            NAs to generate patterns for
        seq1bit : slm.Sequence object
            the sequence for the 1 bit images
        seq24bit : slm.Sequence object
            the sequence for the 24 bit images
        """
        if not (seq is seq_24_1ms or seq is seq_24_50ms):
            raise ValueError("Must use a 24 bit sequence")
        # for now we're going to assume we're using a 3 phase mask (makes for even divsion)
        # assuming that we're only doing two beams
        phase_step = 2
        nphases = 8
        onfrac = 0.5
        # we want 8 phases
        phases = [(n / nphases / phase_step) * (2 * pi) for n in range(8)]
        super().__init__(name, wls, nas, phases, 3, seq, onfrac)

    def make_ROs(self):
        """Sub-method that makes the running orders and adds them to the
        Repertoire.
        """
        for wl, na_dict in sorted(self.bitplanes.items()):
            for na, angle_list in sorted(na_dict.items()):
                self.gen_palms_2(wl, na, angle_list)
                self.gen_palms_6(wl, na, angle_list)

    def gen_palms_2(self, wl, na, angle_list):
        """Makes a RunningOrder to display all angles at once"""
        # make an array of the first phase of the data
        RO_name = ("{} nm ".format(wl) +
            "{:.2f} NA ".format(na) +
            "2 Beam PALM")
        print('Generating "' + RO_name + '"')
        # expand bitplanes into stack
        data_stack = np.array([phase_bp.image
            for angle in angle_list
            for phase_bp in angle])
        # make a 24-bit bitplane and single frame
        bp24 = BitPlane24(data_stack, RO_name.replace(" ", "-"))
        # looping without triggering
        frame = Frame(self.seq, bp24, True, False, False)
        # make and add the RO
        RO = RunningOrder(RO_name, frame)
        self.rep.addRO(RO)
        RO = RunningOrder(RO_name + " triggered",  [Frame(self.seq, bp24, False, True, False),
            Frame(self.seq, bp24, True, True, False)])
        self.rep.addRO(RO)

    def gen_palms_6(self, wl, na, angle_list):
        """Makes a RunningOrder to display all angles at once"""
        # make an array of the first phase of the data
        RO_name = (
            "{} nm ".format(wl) +
            "{:.2f} NA ".format(na) +
            "6 Beam PALM"
        )
        print('Generating "' + RO_name + '"')
        # make an array of all the bitplanes (ordered as angle x phase)
        data_array = np.array([
            [phase_bp.image for phase_bp in angle]
            for angle in angle_list
        ])
        # set up container
        angle_bp_list = []
        # loop through angle indexes
        angle_idxs = list(range(data_array.shape[0]))
        for i in angle_idxs:
            # make a slice
            s = angle_idxs[:i] + angle_idxs[i + 1:]
            # average the first phase of two angles and all the phases
            # of the other angles
            d = (data_array[s, :1].sum(0) + data_array[i]) / len(angle_idxs)
            # digitize bitplanes for one angle phase stepping
            angle_bp_list.append(d > 0.5)
        # make a new 24-bit plane
        bp24 = BitPlane24(np.concatenate(angle_bp_list), RO_name.replace(" ", "-"))
        # looping without triggering
        frame = Frame(self.seq, bp24, True, False, False)
        # make and add the RO
        RO = RunningOrder(RO_name, frame)
        self.rep.addRO(RO)
        RO = RunningOrder(RO_name + " triggered", [Frame(self.seq, bp24, False, True, False),
            Frame(self.seq, bp24, True, True, False)])
        self.rep.addRO(RO)


def _gen_name(angle, wl, na, n, onfrac):
    """Generate a unique name for a BitPlane"""
    degree = np.rad2deg(angle)
    my_per = ideal_period(wl, na)
    name = 'pat-{}nm-{:.2f}NA{:+.1f}deg-{:02d}ph-{:.4f}pix-{:.2f}DC'
    return name.format(wl, na, degree, n, my_per, onfrac)
