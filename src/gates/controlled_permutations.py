import itertools
import random
import numpy as np

from src.layered.compile_pkg.opt.distance_measures import size_check
from src.utils.rotation_utils import matmul


def permutation_matrix(dimension, indexes):
    results = np.zeros((dimension, dimension), dtype='complex')
    for i, idx in enumerate(indexes):
        results[i][idx] = 1

    return results


def add_phase(matrix, phases):
    results = matrix.copy()

    for i, p in enumerate(phases):
        results[i] = p * results[i]

    return results


def permutation_in_space(matrices, starting_indexes):
    if not type(matrices) == 'list':
        matrices = [matrices]
        starting_indexes = [starting_indexes]

    space = None
    for counter, m in enumerate(matrices):
        starting_index = starting_indexes[counter]

        if not size_check(m, m.T):
            raise Exception

        dim = m.shape[0]

        space = np.identity(dim ** 2, dtype='complex')
        for i in range(dim):
            for j in range(dim):
                space[starting_index + i][starting_index + j] = m[i][j]

    return space


def generate_indexes(dimension):
    positions = list(range(dimension))

    perms = list(itertools.permutations(positions))

    return perms


def random_controlled_perm_gate(dimension, phases=None):
    perms = generate_indexes(dimension)

    select_perm = random.randint(0, len(perms) - 1)

    select_indx = random.randint(0, dimension - 1)
    select_indx = select_indx * dimension

    select = perms[select_perm]
    matrix = permutation_matrix(dimension, select)

    if phases:
        matrix = add_phase(matrix, phases)

    embedded = permutation_in_space(matrix, select_indx)

    return embedded


def multiple_rand_controlled_perm_gate(dimension, phases=None):
    space = np.identity(dimension ** 2, dtype='complex')

    perms = generate_indexes(dimension)

    for i in range(dimension):
        select_perm = random.randint(0, len(perms) - 1)

        do_it = random.randint(0, 1)

        if do_it == 1:
            select_indx = i * dimension

            select = perms[select_perm]
            matrix = permutation_matrix(dimension, select)
            embedded = permutation_in_space(matrix, select_indx)
            space = matmul(space, embedded)

    if phases:
        space = add_phase(space, phases)

    return space
