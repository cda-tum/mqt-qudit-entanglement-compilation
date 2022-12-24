import itertools

import numpy as np

from src.gates.rotations import CustomUnitary, R
from src.utils.rotation_utils import matmul


class H:

    def __init__(self, dimension):
        index_iter = [list(range(dimension)), list(range(dimension))]

        ret = np.outer([0 for _ in range(dimension)], [0 for _ in range(dimension)])

        for e0, e1 in itertools.product(*index_iter):
            omega = np.mod(2 / dimension * (e0 * e1), 2)
            omega = omega * np.pi * 1j
            omega = np.e ** omega

            l1 = [0 for _ in range(dimension)]
            l2 = [0 for _ in range(dimension)]
            l1[e0] = 1
            l2[e1] = 1

            array1 = np.array(l1, dtype='complex')
            array2 = np.array(l2, dtype='complex')

            result = omega * np.outer(array1, array2)

            ret = ret + result

        ret = (1 / np.sqrt(dimension)) * ret

        self.matrix = ret


class Z:
    def __init__(self, dimension):
        index_iter = list(range(dimension))

        ret = np.outer([0 for _ in range(dimension)], [0 for _ in range(dimension)])

        for el in index_iter:
            omega = np.mod(2 * el / dimension, 2)
            omega = omega * np.pi * 1j
            omega = np.e ** omega

            l1 = [0 for _ in range(dimension)]
            l2 = [0 for _ in range(dimension)]
            l1[el] = 1
            l2[el] = 1

            array1 = np.array(l1, dtype='complex')
            array2 = np.array(l2, dtype='complex')

            result = omega * np.outer(array1, array2)

            ret = ret + result

        self.matrix = ret


class X:
    def __init__(self, dimension):
        index_iter = list(range(dimension))

        ret = np.outer([0 for _ in range(dimension)], [0 for _ in range(dimension)])

        for el in index_iter:
            i = el
            i_plus_1 = np.mod(i + 1, dimension)

            l1 = [0 for _ in range(dimension)]
            l2 = [0 for _ in range(dimension)]
            l1[i_plus_1] = 1
            l2[i] = 1

            array1 = np.array(l1, dtype='complex')
            array2 = np.array(l2, dtype='complex')

            result = np.outer(array1, array2)

            ret = ret + result

        self.matrix = ret


class S:
    def __init__(self, dimension):
        index_iter = list(range(dimension))

        ret = np.outer([0 for _ in range(dimension)], [0 for _ in range(dimension)])

        for el in index_iter:
            omega = np.mod(2 / dimension * el * (el + 1) / 2, 2)
            omega = omega * np.pi * 1j
            omega = np.e ** omega

            l1 = [0 for _ in range(dimension)]
            l2 = [0 for _ in range(dimension)]
            l1[el] = 1
            l2[el] = 1

            array1 = np.array(l1, dtype='complex')
            array2 = np.array(l2, dtype='complex')

            result = omega * np.outer(array1, array2)

            ret = ret + result

        self.matrix = ret


class ZditR:
    def __init__(self, teta, l1, l2, d):
        self.matrix = (R(np.pi / 2, 0, l1, l2, d) * R(teta, np.pi / 2, l1, l2, d) * R(-np.pi / 2, 0, l1, l2,
                                                                                      d)).matrix
        self.dimension = d

    def __mul__(self, x):
        return CustomUnitary(matmul(self.matrix, x.matrix), self.dimension)


class HditR:
    def __init__(self, l1, l2, d):
        self.matrix = (R(-np.pi, 0, l1, l2, d) * R(np.pi / 2, np.pi / 2, l1, l2, d)).matrix
        self.dimension = d

    def __mul__(self, x):
        return CustomUnitary(matmul(self.matrix, x.matrix), self.dimension)
