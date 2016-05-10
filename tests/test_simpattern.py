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

    def test_name(self):
        """
        The name the hash
        """
        bp = BitPlane(np.random.randint(1, size=(512, 512)))
        assert hex(hash(bp)) == bp.name

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

    def test_name(self):
        """
        Make sure sequence name is determined correctly
        """
        seq1 = Sequence("path")
        assert seq1.name == "path"
        seq2 = Sequence(os.path.join("junk", "junk2", "seq.sq"))
        assert seq2.name == "seq.sq"


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
        self.data1 = np.random.randint(2, size=(512, 512))
        self.data2 = np.random.randint(2, size=(512, 512))
        self.bp1 = BitPlane(self.data1)
        self.bp2 = BitPlane(self.data2)
        self.frame1 = Frame((self.seq1, self.seq2), (self.bp1, self.bp2),
                       True, False)
        self.frame2 = Frame((self.seq1, ), (self.bp1, ), True, False)
        self.RO1 = RunningOrder("Dummy Frame", (self.frame1, self.frame2))
        self.RO2 = RunningOrder("Dummy Frame2", (self.frame1, ))
        self.rep = Repertoire("Dummy", (self.RO1, self.RO2))

    def test_bps(self):
        """
        We initialized two bitplanes with different data, they should be different
        """
        assert_not_equal(self.bp1, self.bp2)

    def test_rep_duplicates(self):
        """
        Test for duplicates
        """
        assert len(self.rep) == 2
        assert self.rep.sequences == {self.seq1, self.seq2}

    def test_rep_bp_write(self):
        """
        Make sure the returned string is of the correct format
        """
        test_str = self.rep.prep_bp_for_write()
        bp_str = ["IMAGES"]
        flip = {v: k for k, v in self.rep.bp_dict.items()}
        bp_str.extend(['1 "' + bp.name + '"'
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
