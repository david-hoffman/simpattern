from nose.tools import *
import os
import numpy as np
import unittest

from simpattern import BitPlane, Sequence, Frame, RunningOrder, Repertoire


class TestBitPlane(unittest.TestCase):
    """
    Testing functionality of BitPlane class
    """

    def test_hash(self):
        """
        Make sure BitPlane aren't hashable without data
        """
        bp = BitPlane()
        assert_raises(TypeError, hash, bp)

    def test_hash_wdata(self):
        """
        Make sure BitPlane's are hashable
        """
        bp = BitPlane(np.random.randint(1, size=(512, 512)))
        hash(bp)

    def test_eq(self):
        """
        Make sure BitPlane's are equal
        """
        data = np.random.randint(1, size=(512, 512))
        bp1 = BitPlane(data)
        bp2 = BitPlane(data)
        assert bp1 == bp2, "bp1 and bp2 are not equal!"


class TestSequence(unittest.TestCase):
    """
    Testing functionality of Sequence class
    """

    def test_hash(self):
        """
        Make sure Sequences's are hashable
        """
        seq = Sequence("path")
        hash(seq)


class TestRepertoire(unittest.TestCase):
    """
    Testing functionality of BitPlane class
    """

    def setUp(self):
        """
        Set up some dummy sequences and bitplanes
        """
        self.seq1_path = "path1"
        self.seq2_path = "path2"
        self.seq1 = Sequence(self.seq1_path)
        self.seq2 = Sequence(self.seq2_path)
        self.data1 = np.random.randint(1, size=(512, 512))
        self.data2 = np.random.randint(1, size=(512, 512))
        self.bp1 = BitPlane(self.data1)
        self.bp2 = BitPlane(self.data2)

    def test_rep_duplicates(self):
        """
        Test for duplicates
        """
        frame1 = Frame((self.seq1, self.seq2), (self.bp1, self.bp2),
                       True, False)
        frame2 = Frame((self.seq1, ), (self.bp1, ), True, False)
        RO1 = RunningOrder("Dummy Frame", (frame1, frame2))
        RO2 = RunningOrder("Dummy Frame2", (frame1, ))
        rep = Repertoire("Dummy", (RO1, RO2))
        assert len(rep) == 2
        assert rep.sequences == {self.seq1, self.seq2}
