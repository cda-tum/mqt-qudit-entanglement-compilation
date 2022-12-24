from unittest import TestCase

import numpy as np

import global_vars, customize_vars
from decomposer.cex import Cex
from decomposer.entangled_qr import EntangledQR
from gates.entangling_gates import csum
from utils.qudit_circ_utils import gate_expand_to_circuit


class TestEntangledQR(TestCase):
    def setUp(self) -> None:
        global_vars.SINGLE_DIM = 4
        self.test_unitary = csum(global_vars.SINGLE_DIM)

    def test_virtual_qr_givens_only(self):
        decomposer = EntangledQR(self.test_unitary)
        sequence = decomposer.virtual_qr_givens_only()
        outcome = decomposer.basic_verify(self.test_unitary, sequence)
        self.assertTrue(outcome)

    def test_entangling_qr(self):
        decomposer = EntangledQR(self.test_unitary)
        sequence, num_crot, num_pswaps = decomposer.entangling_qr()
        outcome = decomposer.basic_verify(self.test_unitary, sequence)
        self.assertTrue(outcome)
        self.assertTrue(num_pswaps == 36)
        self.assertTrue(num_crot == 92)

    def test_cex_sequence(self):
        global_vars.SINGLE_DIM = 4
        dim = global_vars.SINGLE_DIM

        identity = gate_expand_to_circuit(np.identity(dim, dtype=complex), n=2, target=0, dim=dim)
        customize_vars.CEX_SEQUENCE = [identity, Cex().cex_101(dim), identity]

        decomposer = EntangledQR(self.test_unitary)
        sequence, num_crot, num_pswaps = decomposer.entangling_qr()
        outcome = decomposer.basic_verify(self.test_unitary, sequence)
        self.assertTrue(outcome)
        self.assertTrue(num_pswaps == 36)
        self.assertTrue(num_crot == 92)
