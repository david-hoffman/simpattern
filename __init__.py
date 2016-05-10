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
    A Repertoire declares the Sequences to be used, the images to
    be shown and defines the Running Order(s).

    - Sequences
        Sequence files instruct the system which bit-plane(s) to upload to the
        microdisplay, when to illuminate them, and for how long. Please refer
        to [7], [8] for more information on Sequence files.
    - Images
        Image files contain bit-plane data to be shown on the microdisplay.
        These can be 8-bit and / or 1-bit images. Supported image file formats
        are .BMP, .GIF, and .PNG.
    - Running Orders
        A Running Order is a set of user defined instructions that combine
        images, sequences, delays and triggers which all control display
        operation in the Active Mode. A Repertoire may contain one or more
        Running Orders.
    """

    def __init__(self, name, runnningorders=None):
        """
        Initialize the Repertoire from a given set of running orders.
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
        # protect the interal ROs
        return self._ROs

    def __iter__(self):
        # we want to be able to easily iterate over the internal ROs
        return iter(self.ROs)

    def __len__(self):
        # the length of a Repertoire is really the number of ROs
        return len(self.ROs)

    def __str__(self):
        # initialize the result string we'll write to
        rep = ""
        # make prepare sequences and bitplanes for writing.
        rep += self.prep_bp_for_write()
        rep += self.prep_seq_for_write()
        # loop through internal ROs
        for RO in self:
            rep += str(RO)

    def prep_bp_for_write(self):
        """
        This function prepares the internal set of bitplanes for writing a
        rep file

        It generates the string that will appear at the beginning of the
        rep file and generates a dictionary that remembers the positions of
        the bitplanes in the dictionary.

        Both the dictionary and the str are saved as instance variables.
        """
        self.bp_dict = {}
        bps = ['IMAGES']
        for i, bp in enumerate(sorted(self.bitplanes)):
            # we assume bitplanes have a bit depth of 1 here.
            bps.append('1 "' + bp.name + '"')
            self.bp_dict[bp] = i
        bps.append('IMAGES_END\n\n')
        self.bp_str = "\n".join(bps)
        return self.bp_str

    def prep_seq_for_write(self):
        """
        This function prepares the internal set of sequences for writing a
        rep file

        It generates the string that will appear at the beginning of the
        rep file and generates a dictionary that remembers the characters
        of the sequences in the dictionary.

        Both the dictionary and the str are saved as instance variables.
        """
        self.seq_dict = {}
        seqs = ['SEQUENCES']
        for i, seq in enumerate(sorted(self.sequences)):
            char = chr(65 + i)
            seqs.append(char + ' "' + seq.name + '"')
            self.seq_dict[seq] = char
        seqs.append('SEQUENCES_END\n\n')
        self.seq_str = "\n".join(seqs)
        return self.seq_str


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
        # protect the internal frames
        return self._frames

    def __iter__(self):
        # we want to easily iterate over the internal frames
        return iter(self.frames)

    def __len__(self):
        # the length of a RunningOrder is really the number of Frames it
        # contains
        return len(self.frames)


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
        # we just want to know if the data is identical
        # we chould probably just compare the hashes.
        return np.all(self.image == other.image)

    def __lt__(self, other):
        # we don't care about the specific names
        return self.name <= other.name

    @property
    def name(self):
        # we want unique names
        return hex(hash(self))
        return self.name


class BlankBitPlane(BitPlane):
    pass


class GridBitPlane(BitPlane):
    pass
