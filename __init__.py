
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
from skimage.draw import circle
from dphutils import slice_maker
from .slm import RunningOrder, BitPlane, Repertoire, Sequence
# define pi for later use
pi = np.pi


# newest one
def pattern(angle, period, onfrac=0.5, phase_idx=0., phase_offset=0.,
            nphases=5, sizex=2048, sizey=1536, SIM_2D=True):
    '''
    Generates a binary SLM pattern for SIM
    Generates a sine wave and then binarizes it
    designed for my orientation
    Parameters
    ----------
    x : array_like
        the $\vec{a}$ from [1]. Defines the pattern orientation
    period : float
        Defines the period of the pattern
    onfrac : float
        The fraction of on pixels in the pattern
    phase_idx : int
        The phase of the pattern (see `nphases` for more info)
    phase_offset : float
        The offset in phase, mostly used for aligning patterns of different
        colors
    nphases : int
        the number of phases
    sizex : int
        size of the pattern
    sizey : int
        size of the pattern

    Returns
    -------
    pattern : ndarray
        A binary array representing a single bitplane to send to the SLM
    '''
    if not 0 < onfrac < 1:
        raise ValueError(('onfrac must have a value between'
                          ' 0 and 1. onfrac = {}').format(onfrac))
    if not 2 <= period <= max(sizex, sizey):
        raise ValueError(('period must be larger than 2 (nyquist limit)'
                          ' and smaller than the size of the array (DC limit)'
                          '. period = {}').format(period))
    if SIM_2D:
        # Then we only want to take steps of 2pi/n in illumination which means
        # pi/n at the SLM
        phase_step = 2
    else:
        phase_step = 1
    # generate grids
    yy, xx = np.indices((sizey, sizex))
    # here's the pattern frequency
    freq = 2 * pi / period
    # calculate phase
    phi = (phase_idx / nphases / phase_step) * (2 * pi) + phase_offset
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
    return minimize(objf_l1, iperiod, method='Nelder-Mead')['x']


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


def tuplify(arg):
    """
    A utility function to convert args to tuples
    """
    # strings need special handling
    if isinstance(arg, str):
        return (arg, )
    # other wise try and make it a tuple
    try:
        # turn passed arg to list
        return tuple(arg)
    except TypeError:
        # if not iterable, then make list
        return (arg, )


class SIMRepertoire(object):
    """
    A class that takes care of the actual generation of images

    This is _not_ a subclass of slm.Repertoire, but does hold an instance
    """

    blank = BitPlane(np.zeros((1536, 2048)), "Blank")

    def __init__(self, name, wls, nas, orders, norientations, seq):
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
        # for now hard code, we have to double the phases
        # so we can do 90 deg phase stepping.
        self.nphases = prod(orders) * 2
        if prod(len(self.nas), len(self.wls),
                norientations + 1, self.nphases) + 1 > 1024:
            raise RuntimeError("These settings will generate too many")
        # For the current microscope we can only have one set of orientation
        if norientations == 3:
            init_angle = 11.6
        elif norientations == 5:
            init_angle = 9.0
        elif norientations == 7:
            init_angle = 0.9
        else:
            raise RuntimeError("number of orientations not valid, ",
                               norientations)
        self.angles = make_angles(norientations, init_angle)

        # once we have all this info we can start making bitplanes
        self.make_bitplanes()

    def make_bitplanes(self):
        """
        Function that returns a dictionary of dictionary of bitplanes
        of BitPlanes
        """
        # first level is wl
        self.bitplanes = {wl: {
            na: [
                [BitPlane(pattern(ang, ideal_period(wl, na), phase_idx=n,
                                  nphases=self.nphases),
                          gen_name(ang, wl, na, n))
                 for n in range(self.nphases)] for ang in self.angles]
            for na in self.nas
        } for wl in self.wls}

    def make_ROs(self):
        for wl, na_dict in self.bitplanes.items():
            for na, angle_list in na_dict.items():
                self.gen_fast_sims(wl, na, angle_list)
                self.gen_super_sims(wl, na, angle_list)
                self.gen_all_angles(wl, na, angle_list)

    def gen_all_angles(self, wl, na, angle_list):
        data_array = np.array([
            ang[0].image for ang in angle_list
        ])
        bitplane = data_array.mean(0) > 0.5
        name = "AllAngles-{}nm-{:.2f}NA".format(wl, na)
        all_angles_bitplane = BitPlane(bitplane, name)

    @property
    def name(self):
        return self.rep.name

    def write(self, path=""):
        """
        Write the internal repertoire to a repz11 file
        """
        self.rep.write_repz11(path)


def gen_name(angle, wl, na, n):
    """
    Generate a unique name for a BitPlane
    """
    degree = np.rad2deg(angle)
    my_per = ideal_period(wl, na)
    name = 'pat-{}nm-{:.2f}NA{:+.1f}deg-{:02d}ph-{:.4f}pix-{:.2f}DC'
    return name.format(wl, na, degree, n, my_per, 0.5)
