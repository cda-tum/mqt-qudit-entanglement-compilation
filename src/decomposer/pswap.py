import numpy as np

from math import floor

from src.customize_vars import CEX_SEQUENCE
from src.decomposer.cex import Cex
from src.gates.pauli import HditR, ZditR
from src.gates.rotations import R
from src.utils.qudit_circ_utils import calculate_q0_q1, on0, on1
from src.utils.rotation_utils import matmul


class PSwapGen:
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

    def pswap_form(self, theta, phi, o_lev_a, o_lev_b, dimension):
        self.original_lev_a = o_lev_a
        self.original_lev_b = o_lev_b

        self.lev_a, self.lev_b = self.levels_setter(o_lev_a, o_lev_b, dimension)

        self.is_pswap()

        self.lev_q0a_entangling, self.lev_q1a_entangling = calculate_q0_q1(self.lev_a, dimension)
        self.lev_q0b_entangling, self.lev_q1b_entangling = calculate_q0_q1(self.lev_b, dimension)

        self.theta = theta
        self.phi = phi

        self.dimension = dimension

        identity = np.identity(dimension, dtype='complex')

        identity[self.lev_a, self.lev_a] = np.cos(theta / 2) * identity[self.lev_a, self.lev_a]
        identity[self.lev_b, self.lev_b] = np.cos(theta / 2) * identity[self.lev_b, self.lev_b]
        identity[self.lev_b, self.lev_a] = np.sin(theta / 2) * (-1j * np.cos(phi) + np.sin(phi))
        identity[self.lev_a, self.lev_b] = np.sin(theta / 2) * (-1j * np.cos(phi) - np.sin(phi))

        self.matrix = identity

        self.shape = self.matrix.shape

    @property
    def dag(self):
        return self.matrix.conj().T.copy()

    def pswap_101_as_list(self, teta, phi, d):

        h_ = HditR(0, 1, d).matrix

        zpiov2 = ZditR(np.pi / 2, 0, 1, d).matrix

        zp = ZditR(np.pi, 0, 1, d).matrix

        rphi_there = R(np.pi / 2, -phi - np.pi / 2, 0, 1, d).matrix

        rphi_back = R(-np.pi / 2, -phi - np.pi / 2, 0, 1, d).matrix

        if CEX_SEQUENCE is None:
            cex = Cex().cex_101(d, 0)
        else:
            cex = CEX_SEQUENCE

        ph1 = -1 * np.identity(d, dtype='complex')

        ph1[0][0] = 1
        ph1[1][1] = 1

        ##############################################################################

        compose = [on0(ph1, d), on0(h_, d)]

        #################################

        if d != 2:
            compose.append(on1(R(np.pi, np.pi / 2, 1, d - 1, d).matrix, d))

        compose.append(on1(h_, d))

        compose.append(on0(zpiov2, d))

        if CEX_SEQUENCE is None:
            compose.append(cex)
        else:
            compose = compose + cex

        compose.append(on0(h_, d))

        compose.append(on1(h_, d))

        ###############################################################
        compose.append(on0(zp, d))

        compose.append(on1(rphi_there, d))  # ----------

        ##################################

        if CEX_SEQUENCE is None:
            compose.append(cex)
        else:
            compose = compose + cex

        compose.append(on1(ZditR(teta / 2, 0, 1, d).matrix, d))

        if CEX_SEQUENCE is None:
            compose.append(cex)
        else:
            compose = compose + cex

        compose.append(on1(ZditR(-teta / 2, 0, 1, d).matrix, d))

        ##################################

        compose.append(on1(rphi_back, d))

        ##################################

        compose.append(on0(h_, d))

        compose.append(on1(h_, d))

        compose.append(on0(zpiov2, d))

        if CEX_SEQUENCE is None:
            compose.append(cex)
        else:
            compose = compose + cex

        compose.append(on0(h_, d))

        compose.append(on1(h_, d))

        ##########################
        if d != 2:
            compose.append(on1(R(-np.pi, np.pi / 2, 1, d - 1, d).matrix, d))

        return compose

    def permute_pswap_101_as_list(self, pos, theta, phase, d):

        j = floor(pos / d)

        rotation = self.pswap_101_as_list(theta, phase, d)

        if j != 0:
            permute_there_00 = on0(R(np.pi, -np.pi / 2, 0, j, d).matrix, d)
            permute_there_01 = on0(R(-np.pi, np.pi / 2, 1, j + 1, d).matrix, d)

            perm = matmul(permute_there_00, permute_there_01)
            permb = perm.conj().T

            return [perm] + rotation + [permb]
        else:
            return rotation

    def permute_quad_pswap_101_as_list(self, pos, theta, phase, d):

        j = floor(pos / d)

        rotation = self.pswap_101_as_list(theta / 4, phase, d)
        rotation.reverse()

        if j != 0:
            permute_there_00 = on0(R(np.pi, -np.pi / 2, 0, j, d).matrix, d)
            permute_there_01 = on0(R(-np.pi, np.pi / 2, 1, j + 1, d).matrix, d)
            perm = matmul(permute_there_00, permute_there_01)
            permb = perm.conj().T
            return [permb] + rotation + rotation + rotation + rotation + [perm]
        else:
            return rotation + rotation + rotation + rotation

    def z_pswap_101_as_list(self, i, phase, dimension_single):

        pi_there = self.permute_quad_pswap_101_as_list(i, np.pi / 2, 0.0, dimension_single)
        rotate = self.permute_quad_pswap_101_as_list(i, phase, np.pi / 2, dimension_single)
        pi_back = self.permute_quad_pswap_101_as_list(i, -np.pi / 2, 0.0, dimension_single)

        return pi_back + rotate + pi_there
