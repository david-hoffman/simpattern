# -*- coding: utf-8 -*-
"""
Created on 5/10/2016

@author: david hoffman

Package holding the necessary functions for generating SIM patterns for
the QXGA-3DM writing repz11 files and ini files for labVIEW
"""

import os
import collections
import zipfile
import hashlib
import numpy as np
import numexpr as ne
# for minimizing the difference between the desired frequency and the calculated one
from scipy.optimize import minimize
# PIL allows us to write binary images, though we need a cludge, see 'Writing Binary Files.ipynb'
from PIL import Image
try:
    from pyfftw.interfaces.numpy_fft import (fftn, ifftshift, fftshift, fftfreq)
    import pyfftw
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import (fftn, ifftshift, fftshift, fftfreq)
from skimage.draw import circle


class Repertoire(object):
    """
    """

    def __init__(self, name, runnningorders=None):
        """
        """
        self.name = name
        # define a list of frames
        if runnningorders is None:
            self._ROs = []
            # store sets of bitplanes and sequences
            self.sequences = set()
            self.bitplanes = set()
        else:
            self._ROs = runnningorders
            # build set of sequences in the repertoire
            self.sequences = {seq for RO in self for frame in RO
                              for seq in frame.sequences}
            # build set of bitplanes to store in repertoire
            self.bitplanes = {bp for RO in self for frame in RO
                              for bp in frame.bitplanes}

    @property
    def ROs(self):
        return self._ROs

    def __iter__(self):
        return iter(self.ROs)

    def __len__(self):
        return len(self.ROs)


class RunningOrder(object):
    """
    A Running Order is a user defined list of instructions executed by the
    system. The Running Order determines and controls the display of
    bit-planes/n-bit images on the microdisplay by directing the Display
    Controller to execute Compiled Sequences on selected bit-planes/n-bit
    images.
    """

    def __init__(self, name, frames=None):
        """
        """
        self.name = name
        # define a list of frames
        if frames is None:
            self._frames = []
        else:
            self._frames = frames

    @property
    def frames(self):
        return self._frames

    def __iter__(self):
        return iter(self.frames)


class Frame(object):
    """
    A Frame is an association between a sequence and an image. It indicates
    that a particular image is to be shown using an iteration of a particular
    sequence. A Frame is described by the sequence and image designators in
    a parenthesised pair, for example
    """
    # for now there will be two types of Frames, looped and unlooped

    def __init__(self, sequences, bitplanes, looped, triggered):
        self.looped = looped
        self.triggered = triggered
        self.sequences = sequences
        self.bitplanes = bitplanes


class Sequence(collections.namedtuple("Sequence", ["path"])):
    """
    A class representing a sequence
    """
    __slots__ = ()

    @property
    def name(self):
        # return the filename assosicated with the path.
        return os.path.split(self.path)[-1]

    def __hash__(self):
        return hash(self.path)


class BitPlane(object):
    """
    BitPlanes have data (the image) and names
    """
    # This class should take care of all the lowlevel tasks we may later
    # want to implement, such as loading from disk writing to disk, excetera

    def __init__(self, image=None):
        """
        """
        self.image = image

    def __hash__(self):
        return hash(hashlib.sha1(self.image))

    def __eq__(self, other):
        # we don't care about the specific names
        return np.all(self.image == other.image)



class BlankBitPlane(BitPlane):
    pass


class GridBitPlane(BitPlane):
    pass
