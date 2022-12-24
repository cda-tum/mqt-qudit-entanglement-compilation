import random

import numpy as np
from scipy.linalg import expm
from src.layered.compile_pkg.opt.distance_measures import from_dirac_to_basis
from src.gates.gellman import GellMann
from src.layered.compile_pkg.ansatz.parametrize import generic_sud, params_splitter, reindex
from src.gates.pauli import X
from src.utils.qudit_circ_utils import gate_expand_to_circuit
from src.utils.rotation_utils import matmul


def ms_gate(theta, d):
    return expm(-1j * theta * (matmul(np.outer(np.identity(d, dtype='complex'), GellMann(0, 1, 's', d).matrix)
                                      + np.outer(GellMann(0, 1, 's', d).matrix, np.identity(d, dtype='complex'))
                                      , np.outer(np.identity(d, dtype='complex'), GellMann(0, 1, 's', d).matrix)
                                      + np.outer(GellMann(0, 1, 's', d).matrix, np.identity(d, dtype='complex'))
                                      )) / 4)


#  Made as : Exp[-i \[Theta] sum (| ii > < ii |)]
def ls_gate(theta, d):
    exp_matrix = np.outer(np.array([0] * d ** 2), np.array([0] * d ** 2))

    for i in range(d):
        exp_matrix += np.outer(np.array(from_dirac_to_basis([i, i], d)), np.array(from_dirac_to_basis([i, i], d)))

    return expm(-1j * theta * exp_matrix)


def csum(dimension):
    result = np.zeros(dimension ** 2, dtype='complex')

    for i in range(dimension):
        temp = np.zeros(dimension, dtype='complex')
        mapmat = temp + np.outer(np.array(from_dirac_to_basis([i], dimension)),
                                 np.array(from_dirac_to_basis([i], dimension)))

        xmat = X(dimension).matrix
        xmat_i = np.linalg.matrix_power(xmat, i)

        result = result + (np.kron(mapmat, xmat_i))

    return result


def insert_at(big_arr, pos, to_insert_arr):
    x1 = pos[0]
    y1 = pos[1]
    x2 = x1 + to_insert_arr.shape[0]
    y2 = y1 + to_insert_arr.shape[1]

    assert x2 <= big_arr.shape[0], "the position will make the small matrix exceed the boundaries at x"
    assert y2 <= big_arr.shape[1], "the position will make the small matrix exceed the boundaries at y"

    big_arr[x1:x2, y1:y2] = to_insert_arr

    return big_arr


# TODO FIX
def random_uniform_assigner(b1, b2, b3, num_params_single, d, number_unitaries):
    set_of_ass = []

    for i in range(number_unitaries):
        assignment = [None] * (num_params_single + 1)

        for m in range(0, d):
            for n in range(0, d):
                if m == n:
                    assignment[reindex(m, n, d)] = b3[1] * random.uniform(0, 1)
                elif m > n:
                    assignment[reindex(m, n, d)] = b1[1] * random.uniform(0, 1)
                else:
                    assignment[reindex(m, n, d)] = b2[1] * random.uniform(0, 1)

        set_of_ass = set_of_ass + assignment[:-1]  # no global phase

    return set_of_ass


def custom_random_entangling(dimension):
    number_unitaries = 4
    num_params_single_unitary = -1 + dimension ** 2
    dim = dimension

    bound1 = [0, np.pi]
    bound2 = [0, np.pi / 2]
    bound3 = [0, 2 * np.pi]
    x_rand = random_uniform_assigner(bound1, bound2, bound3, num_params_single_unitary, dimension, number_unitaries)

    params = params_splitter(x_rand, dim)

    u0 = gate_expand_to_circuit(generic_sud(params[0], dim), n=2, target=0, dim=dim)
    u1 = gate_expand_to_circuit(generic_sud(params[1], dim), n=2, target=1, dim=dim)

    u2 = gate_expand_to_circuit(generic_sud(params[2], dim), n=2, target=0, dim=dim)
    u3 = gate_expand_to_circuit(generic_sud(params[3], dim), n=2, target=1, dim=dim)

    ms = ms_gate(np.pi / 2, dim)

    instructions = [u0, u1, ms, u2, u3]

    unitary = instructions[0]

    for i in range(1, len(instructions)):
        unitary = matmul(unitary, instructions[i])

    return unitary




