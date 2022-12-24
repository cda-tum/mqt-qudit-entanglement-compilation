import numpy as np

from src.customize_vars import CEX_SEQUENCE
from src.decomposer.cex import Cex
from src.gates.pauli import ZditR
from src.gates.rotations import R
from src.utils.qudit_circ_utils import calculate_q0_q1, on0, on1
from src.utils.rotation_utils import matmul

from math import floor


class CRotGen:

    @staticmethod
    def levels_setter(la, lb, dimension):
        if la == lb:
            raise Exception
        if la < 0:
            la = dimension + la
        if lb < 0:
            lb = dimension + lb
        if la < lb:
            return la, lb
        else:
            return lb, la

    def __init__(self):
        pass

    def crot_form(self, theta, phi, o_lev_a, o_lev_b, dimension):
        self.original_lev_a = o_lev_a
        self.original_lev_b = o_lev_b

        self.lev_a, self.lev_b = self.levels_setter(o_lev_a, o_lev_b, dimension)

        self.lev_q0a_entangling, self.lev_q1a_entangling = calculate_q0_q1(self.lev_a, dimension)
        self.lev_q0b_entangling, self.lev_q1b_entangling = calculate_q0_q1(self.lev_b, dimension)

        self.theta = theta
        self.phi = phi

        self.dimension = dimension

        identity = np.identity(dimension ** 2, dtype='complex')

        identity[self.lev_a, self.lev_a] = np.cos(theta / 2) * identity[self.lev_a, self.lev_a]
        identity[self.lev_b, self.lev_b] = np.cos(theta / 2) * identity[self.lev_b, self.lev_b]
        identity[self.lev_b, self.lev_a] = np.sin(theta / 2) * (-1j * np.cos(phi) + np.sin(phi))
        identity[self.lev_a, self.lev_b] = np.sin(theta / 2) * (-1j * np.cos(phi) - np.sin(phi))

        self.matrix = identity

        self.shape = self.matrix.shape

        return identity

    @property
    def dag(self):
        return self.matrix.conj().T.copy()

    def crot_101_as_list(self, teta, phi, d):

        frame_back = on1(R(-np.pi / 2, -phi - np.pi / 2, 0, 1, d).matrix, d)

        tminus = on1(ZditR(-teta / 2, 0, 1, d).matrix, d)

        tplus = on1(ZditR(teta / 2, 0, 1, d).matrix, d)

        frame_there = on1(R(np.pi / 2, -phi - np.pi / 2, 0, 1, d).matrix, d)

        if CEX_SEQUENCE is None:
            cex = Cex().cex_101(d, 0)
        else:
            cex = CEX_SEQUENCE

        #############

        compose = [frame_there]

        if CEX_SEQUENCE is None:
            compose.append(cex)
        else:
            compose = compose + cex

        compose.append(tminus)

        if CEX_SEQUENCE is None:
            compose.append(cex)
        else:
            compose = compose + cex

        compose.append(tplus)

        compose.append(frame_back)

        return compose

    def permute_crot_101_as_list(self, i, theta, phase, d):

        q0_i = floor(i / d)
        q1_i = i - (d * q0_i)

        rot_there = []
        rot_back = []

        rotation = self.crot_101_as_list(theta, phase, d)

        if q0_i == 1 and q1_i == 0:
            return rotation

        if q1_i != 0:
            permute_there_10 = on1(R(np.pi, -np.pi / 2, 0, q1_i, d).matrix, d)
            permute_there_11 = on1(R(-np.pi, np.pi / 2, 1, q1_i + 1, d).matrix, d)

            perm = matmul(permute_there_10, permute_there_11)
            permb = perm.conj().T

            rot_there.append(perm)
            rot_back.append(permb)

        if q0_i != 1:
            permute_there_00 = on0(R(np.pi, -np.pi / 2, 1, q0_i, d).matrix, d)
            permute_back_00 = on0(R(np.pi, np.pi / 2, 1, q0_i, d).matrix, d)

            rot_there.append(permute_there_00)
            rot_back.insert(0, permute_back_00)

        return rot_there + rotation + rot_back

    def permute_doubled_crot_101_as_list(self, i, theta, phase, d):

        q0_i = floor(i / d)
        q1_i = i - (d * q0_i)

        rot_there = []
        rot_back = []

        rotation = self.crot_101_as_list(-theta / 2, -phase, d)
        rotation.reverse()

        if q0_i == 1 and q1_i == 0:
            return rotation + rotation

        if q1_i != 0:
            permute_there_10 = on1(R(np.pi, -np.pi / 2, 0, q1_i, d).matrix, d)
            permute_there_11 = on1(R(-np.pi, np.pi / 2, 1, q1_i + 1, d).matrix, d)

            perm = matmul(permute_there_10, permute_there_11)
            permb = perm.conj().T

            rot_there.append(perm)
            rot_back.append(permb)

        if q0_i != 1:
            permute_there_00 = on0(R(np.pi, -np.pi / 2, 1, q0_i, d).matrix, d)
            permute_back_00 = on0(R(np.pi, np.pi / 2, 1, q0_i, d).matrix, d)

            rot_there.append(permute_there_00)
            rot_back.insert(0, permute_back_00)

        rot_back.reverse()
        rot_there.reverse()

        return rot_back + rotation + rotation + rot_there

    def z_from_crot_101_list(self, i, phase, dimension_single):

        pi_there = self.permute_doubled_crot_101_as_list(i, np.pi / 2, 0.0, dimension_single)
        rotate = self.permute_doubled_crot_101_as_list(i, phase, np.pi / 2, dimension_single)
        pi_back = self.permute_doubled_crot_101_as_list(i, -np.pi / 2, 0.0, dimension_single)

        return pi_back + rotate + pi_there
