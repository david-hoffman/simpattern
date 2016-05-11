
import numpy as np
from numpy.linalg import norm
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
