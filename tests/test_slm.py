# -*- coding: utf-8 -*-
"""
Created on 5/10/2016

@author: david-hoffman
@copyright : David Hoffman

Testing packages
"""

from nose.tools import *
import os
import numpy as np
import unittest

from simpattern.slm import *

# some globals
path_to_seq1 = os.path.join(os.path.dirname(__file__),
                            "..", "HHMI_R11_Seq",
                            "48070 HHMI 10ms.seq11")
path_to_seq2 = os.path.join(os.path.dirname(__file__),
                            "..", "HHMI_R11_Seq",
                            "48071 HHMI 50ms.seq11")
assert os.path.exists(path_to_seq1), os.path.abspath(path_to_seq1)
assert os.path.exists(path_to_seq2), os.path.abspath(path_to_seq2)


def test_tuplify_int():
    """
    Testing that tuplify works, ints
    """
    obj = tuplify(1)
    assert_is_instance(obj, tuple)


def test_tuplify_float():
    """
    Testing that tuplify works, floats
    """
    obj = tuplify(0.1)
    assert_is_instance(obj, tuple)


def test_tuplify_bool():
    """
    Testing that tuplify works, bools
    """
    obj = tuplify(True)
    assert_is_instance(obj, tuple)


def test_tuplify_str():
    """
    Testing that tuplify works, strings
    """
    obj = tuplify("str")
    assert_is_instance(obj, tuple)
    assert_equal(1, len(obj))


def test_tuplify_list():
    """
    Testing that tuplify works, lists
    """
    obj = tuplify([1, 2, 3])
    assert_is_instance(obj, tuple)

class TestBitPlane(unittest.TestCase):
    """
    Testing functionality of BitPlane class
    """

    def test_name(self):
        """
        The name is the hash when no name is passed
        """
        bp = BitPlane(np.random.randint(1, size=(512, 512)))
        assert_equal(hex(hash(bp)), bp.name)

    def test_hash(self):
        """
        Make sure BitPlane's are hashable
        """
        hash(BitPlane(np.random.randint(1, size=(512, 512))))

    def test_eq(self):
        """
        Make sure BitPlane's with the same data are equal
        """
        # make fake data
        data = np.random.randint(1, size=(512, 512))
        # make two bitplanes containing same values but different underlying
        # objects
        bp1 = BitPlane(data)
        bp2 = BitPlane(data.copy())
        # test equality
        assert_equal(bp1, bp2)
        # same thing, but here make two different objects differently
        bp3 = BitPlane(np.ones_like(data))
        bp4 = BitPlane(np.ones_like(data))
        assert_equal(bp3, bp4)
        # make sure there's a hash collision between objects containg same
        # data
        assert_equal(len({bp3, bp4, bp1, bp2}), 2)
        # but show objects are not the same
        assert_is_not(bp1, bp2)
        assert_is_not(bp3, bp4)


class TestSequence(unittest.TestCase):
    """
    Testing functionality of Sequence class
    """
    def setUp(self):
        self.path_to_seq1 = path_to_seq1
        self.path_to_seq2 = path_to_seq2

    def test_hash(self):
        """
        Make sure Sequences's are hashable
        """
        seq = Sequence(self.path_to_seq2)
        hash(seq)

    def test_name(self):
        """
        Make sure sequence name is determined correctly
        """
        seq1 = Sequence(self.path_to_seq1)
        assert_equal(seq1.name, os.path.split(path_to_seq1)[1])


class TestFrame(unittest.TestCase):
    """
    Testing Frame
    """
    def setUp(self):
        """
        Set up some dummy sequences and bitplanes
        """
        # load paths
        self.path_to_seq1 = path_to_seq1
        self.path_to_seq2 = path_to_seq2
        # make sure they're valid
        assert_true(os.path.exists(self.path_to_seq1))
        assert_true(os.path.exists(self.path_to_seq2))
        self.seq1 = Sequence(self.path_to_seq1)
        self.seq2 = Sequence(self.path_to_seq2)
        self.data1 = np.random.randint(2, size=(512, 512))
        self.data2 = np.random.randint(2, size=(512, 512))
        self.bp1 = BitPlane(self.data1)
        self.bp2 = BitPlane(self.data2)

    def test_seq_bp(self):
        "Assertion error thrown for unequal number of sequences and bitplanes"
        assert_raises(AssertionError, Frame, (self.seq1, ),
                      (self.bp1, self.bp2), True, False)


class TestRepertoire(unittest.TestCase):
    """
    Testing functionality of BitPlane class
    """

    def setUp(self):
        """
        Set up some dummy sequences and bitplanes
        """
        self.path_to_seq1 = path_to_seq1
        self.path_to_seq2 = path_to_seq2
        self.seq1 = Sequence(self.path_to_seq1)
        self.seq2 = Sequence(self.path_to_seq2)
        self.data1 = np.random.randint(2, size=(512, 512))
        self.data2 = np.random.randint(2, size=(512, 512))
        self.bp1 = BitPlane(self.data1)
        self.bp2 = BitPlane(self.data2)
        self.frame1 = Frame((self.seq1, self.seq2), (self.bp1, self.bp2),
                       True, False)
        self.frame2 = Frame((self.seq1, ), (self.bp1, ), True, False)
        self.frame3 = Frame((self.seq1, ), (self.bp1, ), False, False)
        self.RO1 = RunningOrder("Dummy Frame", (self.frame1, self.frame2, self.frame3))
        self.RO2 = RunningOrder("Dummy Frame2", (self.frame1, ))
        self.rep = Repertoire("Dummy", (self.RO1, self.RO2))

    def test_bps(self):
        """
        We initialized two bitplanes with different data, they should be different
        """
        assert_not_equal(self.bp1, self.bp2)

    def test_addRO(self):
        """
        Testing that adding a new RO updates internals right
        """
        seq3_path = os.path.join(os.path.dirname(__file__),
                                 "..", "HHMI_R11_Seq",
                                 "48075 HHMI 5ms.seq11")
        seq3 = Sequence(seq3_path)
        data3 = np.random.randint(2, size=(512, 512))
        bp3 = BitPlane(data3)
        frame = Frame((seq3, ), (bp3, ), True, False)
        RO3 = RunningOrder("Testy", (frame, ))
        self.rep.addRO(RO3)
        assert_in(bp3, self.rep.bitplanes)
        assert_in(seq3, self.rep.sequences)

    def test_rep_duplicates(self):
        """
        Test for duplicates
        """
        assert_equal(len(self.rep), 2)
        assert_equal(self.rep.sequences, {self.seq1, self.seq2})

    def test_rep_bp_write(self):
        """
        Make sure the returned string is of the correct format
        """
        test_str = self.rep.prep_bp_for_write()
        bp_str = ["IMAGES"]
        flip = {v: k for k, v in self.rep.bp_dict.items()}
        bp_str.extend(['1 "' + bp.name + '.bmp"'
                       for k, bp in sorted(flip.items())])
        bp_str.append("IMAGES_END\n\n")
        assert_equal(test_str, "\n".join(bp_str))

    def test_rep_seq_write(self):
        """
        Make sure the returned string is of the correct format
        """
        test_str = self.rep.prep_seq_for_write()
        seq_str = ["SEQUENCES"]
        flip = {v: k for k, v in self.rep.seq_dict.items()}
        seq_str.extend([k + ' "' + seq.name + '"'
                        for k, seq in sorted(flip.items())])
        seq_str.append("SEQUENCES_END\n\n")
        assert_equal(test_str, "\n".join(seq_str))

    def test_write_frame(self):
        """
        Testing the frames are put out right
        """
        frame = Frame((self.seq1, self.seq1), (self.bp1, self.bp1), True, True)
        RO = RunningOrder("test_write_frame", (frame, ))
        rep = Repertoire("Dummy rep", (RO,))
        rep.prep_bp_for_write()
        rep.prep_seq_for_write()
        test_str = " {f (A,0) (A,0) }"
        assert_equals(test_str, rep.write_frame(frame))

    def test_write_RO(self):
        """
        Testing the frames are put out right
        """
        frame = Frame((self.seq1, self.seq1), (self.bp1, self.bp1), True, True)
        RO = RunningOrder("test_write_frame", (frame, ))
        rep = Repertoire("Dummy rep", (RO,))
        rep.prep_bp_for_write()
        rep.prep_seq_for_write()
        test_str = '"test_write_frame"\n[HWA \n\n {f (A,0) (A,0) }\n]'
        assert_equals(test_str, rep.write_RO(RO))

    def test_write_rep(self):
        """
        Testing the rep is put out right
        """
        frame = Frame((self.seq1, self.seq1), (self.bp1, self.bp1), True, True)
        RO = RunningOrder("test_write_frame", (frame, ))
        rep = Repertoire("Dummy rep", (RO,))
        test_str = 'SEQUENCES\nA "' + self.seq1.name + '"\nSEQUENCES_END\n\n'
        test_str += 'IMAGES\n1 "' + self.bp1.name + '.bmp"\nIMAGES_END\n\n'
        test_str += 'DEFAULT "test_write_frame"\n[HWA \n\n {f (A,0) (A,0) }\n]\n\n'
        assert_equals(test_str, str(rep))


class TestRepertoire2(unittest.TestCase):
    """
    Testing functionality of BitPlane class
    """

    maxDiff = None

    def setUp(self):
        """
        Set up some dummy sequences and bitplanes
        """
        with open(os.path.join(os.path.dirname(__file__), "test_rep.txt")) as fn:
            str_list = fn.readlines()
        self.strcmp = "".join(str_list)

    def test_rep(self):
        """
        Do we match the test file?
        """
        seqA = Sequence(path_to_seq1)
        seqB = Sequence(path_to_seq2)
        bp_names = [
            "pat-6.92003pixel-0.5DC-Ang0Ph0",
            "pat-6.92003pixel-0.5DC-Ang0Ph1",
            "pat-6.92003pixel-0.5DC-Ang0Ph2",
            "pat-6.92003pixel-0.5DC-Ang0Ph3",
            "pat-6.92003pixel-0.5DC-Ang0Ph4",
            "pat-6.92929pixel-0.5DC-Ang1Ph0",
            "pat-6.92929pixel-0.5DC-Ang1Ph1",
            "pat-6.92929pixel-0.5DC-Ang1Ph2",
            "pat-6.92929pixel-0.5DC-Ang1Ph3",
            "pat-6.92929pixel-0.5DC-Ang1Ph4",
            "pat-6.93262pixel-0.5DC-Ang2Ph0"
        ]
        bp_list = [BitPlane(np.random.randint(2, size=(512, 512)), name)
                   for name in bp_names]

        RO0_frames = [Frame((seqA,), (bp, ), looped, True)
                      for bp in bp_list[:5]
                      for looped in (False, True)]
        RO1_frames = [Frame((seqA,), (bp, ), looped, True)
                      for bp in bp_list[5:-1]
                      for looped in (False, True)]
        RO2_frames = [Frame((seqB, seqB, seqB), bp_list[::5], True, False)]
        RO0 = RunningOrder("NA 0.85 5 phases 1 angle", RO0_frames)
        RO1 = RunningOrder("NA 0.80 5 phases 1 angle", RO1_frames)
        RO2 = RunningOrder("NA 0.85 3 angles no trig", RO2_frames)
        rep = Repertoire("test_rep", (RO0, RO1, RO2))
        assert_equals(str(rep), self.strcmp)
