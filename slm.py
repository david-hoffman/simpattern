# -*- coding: utf-8 -*-
"""
Created on 5/10/2016

@author: david-hoffman
@copyright : David Hoffman

Package holding the necessary functions for generating SIM patterns for
the QXGA-3DM writing repz11 files and ini files for labVIEW
"""

import os
import collections
import zipfile
import hashlib
import numpy as np
# PIL allows us to write binary images, though we need a cludge
# see 'Writing Binary Files.ipynb'
from PIL import Image
from io import BytesIO


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
            self._ROs = list(tuplify(runnningorders))
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

    def addRO(self, RO):
        # add the RO to the internal list
        self._ROs.append(RO)
        # update the internal sets of sequences and bitplanes
        self.sequences.update({seq for frame in RO
                               for seq in frame.sequences})
        self.bitplanes.update({bp for frame in RO
                               for bp in frame.bitplanes})

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
        rep += self.prep_seq_for_write()
        rep += self.prep_bp_for_write()
        # make the first RO "Default"
        rep += "DEFAULT "
        # loop through internal ROs
        for RO in self:
            rep += self.write_RO(RO)
            rep += "\n\n"
        return rep

    def prep_bp_for_write(self):
        """
        This function prepares the internal set of bitplanes for writing a
        rep file

        It generates the string that will appear at the beginning of the
        rep file and generates a dictionary that remembers the positions of
        the bitplanes in the dictionary.

        Both the dictionary and the str are saved as instance variables.
        """
        # initialize dictionary for bitplanes
        self.bp_dict = {}
        # start a list for the final printout
        bps = ['IMAGES']
        # iterate through bitplanes, which will be sorted be name
        for i, bp in enumerate(sorted(self.bitplanes)):
            # we assume bitplanes have a bit depth of 1 here.
            bps.append('1 "' + bp.name + '.bmp"')
            # update dict
            self.bp_dict[bp] = i
        # finish printout
        bps.append('IMAGES_END\n\n')
        # save string internally for later use
        self.bp_str = "\n".join(bps)
        # return right away for ease of use
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
        # initialize dictionary for sequences
        self.seq_dict = {}
        # start a list for final printout
        seqs = ['SEQUENCES']
        # iterate through sequences
        for i, seq in enumerate(sorted(self.sequences)):
            # find right character
            char = chr(65 + i)
            # build printout list
            seqs.append(char + ' "' + seq.name + '"')
            # update dict
            self.seq_dict[seq] = char
        # finish printout
        seqs.append('SEQUENCES_END\n\n')
        # save string for later and return right aways for convenience
        self.seq_str = "\n".join(seqs)
        return self.seq_str

    def write_RO(self, RO):
        """
        Function that writes RO using the internal dictionaries
        """
        # put out name
        result = ['"' + RO.name + '"', '[HWA \n']
        result.extend([self.write_frame(frame) for frame in RO])
        result.append(']')
        return '\n'.join(result)

    def write_frame(self, frame):
        seq_dict = self.seq_dict
        bp_dict = self.bp_dict
        int_result = []
        for seq, bp in zip(frame.sequences, frame.bitplanes):
            int_result.append("({},{}) ".format(seq_dict[seq], bp_dict[bp]))

        if frame.looped:
            result = [" {"]
            if frame.triggered:
                result += ["f "]
            result += int_result + ["}"]
        else:
            result = [" <"]
            if frame.triggered:
                result += ["t"]
            result += int_result + [">"]

        return "".join(result)

    def write_repz11(self, path=""):
        """
        A function for writing a complete repz11 file
        Including a repfile, images and moving sequences

        Parameters
        ----------
        path : path (optional)
            path to place the repz11 file.
        """

        with zipfile.ZipFile(path + self.name + ".repz11", "w",
                             compression=zipfile.ZIP_DEFLATED) as zf:
            # write sequences to zipfile
            for seq in self.sequences:
                zf.write(seq.path, arcname=seq.name)
            # write rep to zipfile
            zf.writestr(self.name + ".rep", str(self).encode())
            # write images to zipfile
            for bp in self.bitplanes:
                # form the 8 bit grayscale image
                bp_img = Image.fromarray(
                    (bp.image * 255).astype('uint8'), mode='L')
                # create an output bytes buffer to save the image to
                output = BytesIO()
                # save the image to the buffer
                bp_img.convert('1').save(output, "BMP")
                # write the buffer to the zipfile
                zf.writestr(bp.name + ".bmp", output.getvalue())

    def write_ini(self):
        raise NotImplementedError


class RunningOrder(object):
    """
    A Running Order is a user defined list of instructions executed by the
    system. The Running Order determines and controls the display of
    bit-planes/n-bit images on the microdisplay by directing the Display
    Controller to execute Compiled Sequences on selected bit-planes/n-bit
    images.
    """
    # NOTE: it might be worthwhile to just subclass a list for this
    # give a list a name attribute

    def __init__(self, name, frames=None):
        """
        """
        self.name = name
        # define a list of frames
        if frames is None:
            self._frames = []
        else:
            self._frames = list(tuplify(frames))

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
        tsequences = tuplify(sequences)
        tbitplanes = tuplify(bitplanes)
        assert len(tsequences) == len(tbitplanes), ("Number of bitplanes does"
                                                    " not equal number of"
                                                    " sequences!")
        self.sequences = tsequences
        self.bitplanes = tbitplanes


class Sequence(object):
    """
    A class representing a sequence
    """

    def __init__(self, path):
        """
        Initialize with path to sequence file.
        """
        self.path = path

    @property
    def name(self):
        # return the filename assosicated with the path.
        return os.path.split(self.path)[-1]

    def __hash__(self):
        return hash(self.path)

    def __eq__(self, other):
        # we don't care about the specific names
        # we just want to know if the data is identical
        # we chould probably just compare the hashes.
        return self.path == other.path

    def __lt__(self, other):
        # we don't care about the specific names
        return self.name <= other.name


class BitPlane(object):
    """
    BitPlanes have data (the image) and names
    """
    # This class should take care of all the lowlevel tasks we may later
    # want to implement, such as loading from disk writing to disk, excetera

    def __init__(self, image, name=None):
        """
        """
        # make a copy so the external array can be used
        self.image = image
        # make it unchangeable
        self.image.flags.writeable = False
        if name is None:
            self._name = hex(hash(self))
        else:
            self._name = name

    def __hash__(self):
        return int.from_bytes(hashlib.md5(self.image.data).digest(),
                              byteorder="big", signed=True)

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
        return self._name
