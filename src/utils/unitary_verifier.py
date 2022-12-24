import numpy as np
from numpy.linalg import inv

from src.utils.rotation_utils import matmul


class UnitaryVerifier:

    def __init__(self, sequence, target, dimension, nodes=None, initial_map=None, final_map=None):
        self.decomposition = sequence
        self.target = target.copy()
        self.dimension = dimension

        if nodes is not None and initial_map is not None and final_map is not None:
            self.permutation_matrix_initial = self.get_perm_matrix(nodes, initial_map)
            self.permutation_matrix_final = self.get_perm_matrix(nodes, final_map)
            self.target = matmul(self.permutation_matrix_initial, self.target)
        else:
            self.permutation_matrix_initial = None
            self.permutation_matrix_final = None

    def get_perm_matrix(self, nodes, mapping):
        # sum ( |phy> <log| )
        perm = np.zeros((self.dimension, self.dimension))

        for i in range(self.dimension):
            a = [0 for i in range(self.dimension)]
            b = [0 for i in range(self.dimension)]
            a[nodes[i]] = 1
            b[mapping[i]] = 1
            narr = np.array(a)
            marr = np.array(b)
            perm = perm + np.outer(marr, narr)

        return perm

    def verify(self):
        target = self.target.copy()

        for rotation in self.decomposition:
            target = matmul(rotation, target)

        if self.permutation_matrix_final is not None:
            target = matmul(inv(self.permutation_matrix_final), target)

        target = target / target[0][0]

        res = (abs(target - np.identity(self.dimension, dtype='complex')) < 1e-4).all()

        return res
