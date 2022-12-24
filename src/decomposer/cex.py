import numpy as np

from src.customize_vars import CEX_SEQUENCE
from src.gates.entangling_gates import insert_at
from src.layered.compile_pkg.opt.distance_measures import from_dirac_to_basis


class Cex:

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

    def __init__(self, phi=None, o_lev_a=None, o_lev_b=None, dimension=None):
        if phi is not None:
            self.original_lev_a = o_lev_a
            self.original_lev_b = o_lev_b

            self.lev_a, self.lev_b = self.levels_setter(o_lev_a, o_lev_b, dimension)

            self.phi = phi

            self.dimension = dimension

            identity = np.identity(dimension, dtype='complex')

            identity[self.lev_b, self.lev_a] = -1j * np.cos(phi) - np.sin(phi)
            identity[self.lev_a, self.lev_b] = -1j * np.cos(phi) + np.sin(phi)
            identity[self.lev_b, self.lev_b] = 0
            identity[self.lev_a, self.lev_a] = 0

            self.matrix = identity

            self.shape = self.matrix.shape
        else:
            self.matrix = None

            self.shape = None

    def cex_101(self, dimension, ang=0):
        result = np.zeros(dimension ** 2, dtype='complex')

        for i in range(dimension):
            temp = np.zeros(dimension, dtype='complex')
            mapmat = temp + np.outer(np.array(from_dirac_to_basis([i], dimension)),
                                     np.array(from_dirac_to_basis([i], dimension)))

            if i == 1:  # apply control on 1 rotation on levels 01

                opmat = np.array([[0, -1j * np.cos(ang) - np.sin(ang)], [-1j * np.cos(ang) + np.sin(ang), 0]])
                embedded_op = np.identity(dimension, dtype='complex')
                embedded_op = insert_at(embedded_op, (0, 0), opmat)

            else:
                embedded_op = np.identity(dimension, dtype='complex')

            result = result + np.kron(mapmat, embedded_op)

        self.matrix = result

        self.shape = self.matrix.shape

        return result

    def cex_101_from_sequence(self):

        self.matrix = CEX_SEQUENCE
        self.shape = self.matrix[0].shape

        return self.matrix

    @property
    def dag(self):
        return self.matrix.conj().T.copy()
