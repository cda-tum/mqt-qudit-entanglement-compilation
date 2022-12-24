import numpy as np
from scipy.linalg import expm
from src.layered.compile_pkg.opt.distance_measures import from_dirac_to_basis
from src.utils.rotation_utils import matmul


def params_splitter(params, dim):
    n = (-1 + dim ** 2)
    ret = [params[i:i + n] for i in range(0, len(params), n)]
    return ret


def reindex(ir, jc, num_col):
    return ir * num_col + jc


bound_1 = [0, np.pi]
bound_2 = [0, np.pi / 2]
bound_3 = [0, 2 * np.pi]


def generic_sud(params, d) -> np.ndarray:  # required well-structured d2 -1 params

    c_unitary = np.identity(d, dtype="complex")

    for diag_index in range(0, d - 1):
        l_vec = from_dirac_to_basis([diag_index], d)
        d_vec = from_dirac_to_basis([d - 1], d)

        zld = np.outer(np.array(l_vec), np.array(l_vec).T.conj()) - np.outer(np.array(d_vec), np.array(d_vec).T.conj())
        c_unitary = matmul(c_unitary, expm(1j * params[reindex(diag_index, diag_index, d)] * zld))
        # print(c_unitary)

    for m in range(0, d - 1):
        for n in range(m + 1, d):
            m_vec = from_dirac_to_basis([m], d)
            n_vec = from_dirac_to_basis([n], d)

            zmn = np.outer(np.array(m_vec), np.array(m_vec).T.conj()) - np.outer(np.array(n_vec),
                                                                                 np.array(n_vec).T.conj())

            ymn = -1j * np.outer(np.array(m_vec), np.array(n_vec).T.conj()) + 1j * np.outer(np.array(n_vec),
                                                                                            np.array(m_vec).T.conj())

            c_unitary = matmul(c_unitary, expm(1j * params[reindex(n, m, d)] * zmn))

            c_unitary = matmul(c_unitary, expm(1j * params[reindex(m, n, d)] * ymn))

    return c_unitary
