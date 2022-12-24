from __future__ import division
import numpy as np
import typing

from src.utils.rotation_utils import matmul

"""FROM An alternative quantum fidelity for mixed states of qudits Xiaoguang Wang, 1, 2, ∗ Chang-Shui Yu, 3 and x. x. 
Yi 3 """


def from_dirac_to_basis(vec, d):  # |00> -> [1,0,0,0]
    basis_vecs = []
    for basis in vec:
        temp = [0] * d
        temp[basis] = 1
        basis_vecs.append(temp)

    ret = basis_vecs[0]
    for e_i in range(1, len(basis_vecs)):
        ret = np.kron(np.array(ret), np.array(basis_vecs[e_i]))

    return ret


def size_check(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape == b.shape:
        if a.shape[0] == a.shape[1]:
            return True

    return False


def fidelity_on_operator(a: np.ndarray, b: np.ndarray) -> float:
    if not size_check(a, b):
        raise Exception

    adag = a.T.conj().copy()
    bdag = b.T.conj().copy()
    numerator = np.abs(np.trace(matmul(adag, b)))
    denominator = np.sqrt(np.trace(matmul(a, adag)) * np.trace(matmul(b, bdag)))

    return typing.cast(float, (numerator / denominator))


def fidelity_on_unitares(a: np.ndarray, b: np.ndarray) -> float:
    if not size_check(a, b):
        raise Exception

    dimension = a.shape[0]

    return typing.cast(float, np.abs(np.trace(matmul(a.T.conj(), b))) / dimension)


def fidelity_on_density_operator(a: np.ndarray, b: np.ndarray) -> float:
    if not size_check(a, b):
        raise Exception

    numerator = np.abs(np.trace(matmul(a, b)))
    denominator = np.sqrt(np.trace(matmul(a, a)) * np.trace(matmul(b, b)))

    return typing.cast(float, (numerator / denominator))


def density_operator(state_vector) -> np.ndarray:
    if isinstance(state_vector, list):
        state_vector = np.array(state_vector)

    return np.outer(state_vector, state_vector.conj())


def frobenius_dist(x, y):
    a = x - y
    return np.sqrt(np.trace(np.abs(matmul(a.T.conj(), a))))
