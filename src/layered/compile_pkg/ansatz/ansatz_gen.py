import numpy as np

from src.gates.entangling_gates import ms_gate, ls_gate
from src.layered.compile_pkg.ansatz.parametrize import params_splitter, generic_sud
from src import global_vars, customize_vars
from src.utils.qudit_circ_utils import gate_expand_to_circuit
from src.utils.rotation_utils import matmul


def cu_ansatz(P, dim):
    params = params_splitter(P, dim)

    cu = customize_vars.CUSTOM_PRIMITIVE

    counter = 0

    unitary = gate_expand_to_circuit(np.identity(dim, dtype=complex), n=2, target=0, dim=dim)

    for i in range(len(params)):

        if counter == 2:
            counter = 0
            unitary = matmul(unitary, cu)

        unitary = matmul(unitary, gate_expand_to_circuit(generic_sud(params[i], dim), n=2, target=counter, dim=dim))
        counter += 1

    return unitary


def ms_ansatz(P, dim):
    params = params_splitter(P, dim)
    ms = ms_gate(np.pi / 2, dim)

    counter = 0

    unitary = gate_expand_to_circuit(np.identity(dim, dtype=complex), n=2, target=0, dim=dim)

    for i in range(len(params)):

        if counter == 2:
            counter = 0
            unitary = matmul(unitary, ms)

        unitary = matmul(unitary, gate_expand_to_circuit(generic_sud(params[i], dim), n=2, target=counter, dim=dim))
        counter += 1

    return unitary


def ls_ansatz(P, dim):
    params = params_splitter(P, dim)

    if dim == 2:
        theta = np.pi / 2
    elif dim == 3:
        theta = 2 * np.pi / 3
    else:
        theta = np.pi

    ls = ls_gate(theta, dim)

    counter = 0

    unitary = gate_expand_to_circuit(np.identity(dim, dtype=complex), n=2, target=0, dim=dim)

    for i in range(len(params)):

        if counter == 2:
            counter = 0
            unitary = matmul(unitary, ls)

        unitary = matmul(unitary, gate_expand_to_circuit(generic_sud(params[i], dim), n=2, target=counter, dim=dim))
        counter += 1

    return unitary
