import numpy as np
from numpy.linalg import solve
from numpy.linalg import det
from numpy import matmul as mml

from src.decomposer.crotgen import CRotGen
from src.decomposer.pswap import PSwapGen
from src.gates.rotations import R
from src.utils.rotation_utils import matmul, pi_mod


class EntangledQR:

    def __init__(self, gate):

        self.u = gate
        dimension = self.u.shape[0]
        self.dimension = dimension

        self.decomposition = None
        self.decomp_indexes = []

    def virtual_qr_givens_only(self):
        decomp = []
        u_ = self.u
        dimension = self.u.shape[0]
        self.dimension = dimension

        index_iterator = list(range(self.u.shape[0]))
        index_iterator.reverse()
        dimension_single = int(np.sqrt(self.dimension))
        for c in range(self.u.shape[1]):

            diag_index = index_iterator.index(c)

            for r in index_iterator[:diag_index]:

                if abs(u_[r, c]) > 1.0e-8:
                    coef_r1 = u_[r - 1, c].round(14)
                    coef_r = u_[r, c].round(14)

                    theta = 2 * np.arctan2(abs(coef_r), abs(coef_r1))

                    phi = -(np.pi / 2 + np.angle(coef_r1) - np.angle(coef_r))

                    phi = pi_mod(phi)
                    rotation_involved = R(theta, phi, r - 1, r, dimension).matrix

                    u_ = matmul(rotation_involved, u_)

                    decomp.append(rotation_involved)

        diag_u = np.diag(u_)
        args_of_diag = []
        for i in range(dimension):
            args_of_diag.append(round(np.angle(diag_u[i]), 6))

        phase_equations = np.zeros((dimension, dimension - 1))

        last_1 = -1
        for i in range(dimension):
            if last_1 + 1 < dimension - 1:
                phase_equations[i, last_1 + 1] = 1

            if last_1 > -1:
                phase_equations[i, last_1] = -1

            last_1 = i

        phases_t = phase_equations.conj().T
        pseudo_inv = mml(phases_t, phase_equations)
        pseudo_diag = mml(phases_t, np.array(args_of_diag))

        if det(pseudo_inv) == 0:
            raise Exception

        phases = solve(pseudo_inv, pseudo_diag)

        for i, phase in enumerate(phases):
            if abs(phase * 2) > 1.0e-4:
                r1 = R(-np.pi / 2, 0, i, i + 1, dimension)
                r2 = R(phase * 2, np.pi / 2, i, i + 1, dimension)
                r3 = R(np.pi / 2, 0, i, i + 1, dimension)
                decomp.append(r1.matrix)
                decomp.append(r2.matrix)
                decomp.append(r3.matrix)
                sub_seq = [r1.matrix, r2.matrix, r3.matrix]

                rotation_inv_mat = np.identity(dimension_single ** 2, dtype='complex')
                for r___ in sub_seq:
                    rotation_inv_mat = matmul(rotation_inv_mat, r___)

                u_ = matmul(rotation_inv_mat, u_)

        self.decomposition = decomp
        return decomp

    def entangling_qr(self):
        crot_counter = 0
        pswap_counter = 0

        pswap_gen = PSwapGen()
        crot_gen = CRotGen()

        decomp = []

        u_ = self.u
        dimension = self.u.shape[0]
        self.dimension = dimension

        dimension_single = int(np.sqrt(self.dimension))

        index_iterator = list(range(self.u.shape[0]))
        index_iterator.reverse()

        for c in range(self.u.shape[1]):

            diag_index = index_iterator.index(c)

            for r in index_iterator[:diag_index]:

                if abs(u_[r, c]) > 1.0e-8:

                    coef_r1 = u_[r - 1, c].round(15)
                    coef_r = u_[r, c].round(15)

                    theta = 2 * np.arctan2(abs(coef_r), abs(coef_r1))

                    phi = -(np.pi / 2 + np.angle(coef_r1) - np.angle(coef_r))

                    phi = pi_mod(phi)
                    #######################
                    if (r - 1) != 0 and np.mod(r, dimension_single) == 0:
                        rotation_involved = pswap_gen.permute_quad_pswap_101_as_list(r - 1, theta, phi,
                                                                                     dimension_single)
                        pswap_counter += 4
                    else:
                        rotation_involved = crot_gen.permute_doubled_crot_101_as_list(r - 1, theta, phi,
                                                                                      dimension_single)
                        crot_counter += 2
                    ######################

                    for r___ in rotation_involved:
                        u_ = matmul(r___, u_)

                    decomp = decomp + rotation_involved

        diag_u = np.diag(u_)
        args_of_diag = []

        for i in range(dimension):
            args_of_diag.append(round(np.angle(diag_u[i]), 6))

        phase_equations = np.zeros((dimension, dimension - 1))

        last_1 = -1
        for i in range(dimension):
            if last_1 + 1 < dimension - 1:
                phase_equations[i, last_1 + 1] = 1

            if last_1 > -1:
                phase_equations[i, last_1] = -1

            last_1 = i

        phases_t = phase_equations.conj().T
        pseudo_inv = mml(phases_t, phase_equations)
        pseudo_diag = mml(phases_t, np.array(args_of_diag))

        if det(pseudo_inv) == 0:
            raise Exception

        phases = solve(pseudo_inv, pseudo_diag)

        for i, phase in enumerate(phases):

            if abs(phase * 2) > 1.0e-4:

                if i != 0 and np.mod(i + 1, dimension_single) == 0:

                    rotation_involved = pswap_gen.z_pswap_101_as_list(i, phase * 2, dimension_single)
                    pswap_counter += 12
                else:
                    rotation_involved = crot_gen.z_from_crot_101_list(i, phase * 2, dimension_single)
                    crot_counter += 6

                ######################
                for r___ in rotation_involved:
                    u_ = matmul(r___, u_)
                #######################

                decomp = decomp + rotation_involved

                print()

        self.decomposition = decomp
        return decomp, crot_counter, pswap_counter

    def basic_verify(self, target, sequence):
        target = target.copy()
        gates = sequence

        for rotation in gates:
            target = matmul(rotation.round(14), target)

        target = target / target[0][0]

        res = (abs(target - np.identity(self.dimension, dtype='complex')) < 10e-5).all()

        return res
