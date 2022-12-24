import numpy as np

from src.gates.entangling_gates import ms_gate, ls_gate
from src.layered.compile_pkg.ansatz.parametrize import params_splitter, generic_sud
from src import global_vars, customize_vars
from src.utils.qudit_circ_utils import gate_expand_to_circuit


def create_cu_instance(P, dim):
    decomposition = []

    params = params_splitter(P, dim)

    cu = customize_vars.CUSTOM_PRIMITIVE

    counter = 0

    for i in range(len(params)):

        if counter == 2:
            counter = 0
            decomposition.append(cu)

        decomposition.append(gate_expand_to_circuit(generic_sud(params[i], dim), n=2, target=counter, dim=dim))

        counter += 1

    return decomposition


def create_ms_instance(P, dim):
    decomposition = []

    params = params_splitter(P, dim)
    ms = ms_gate(np.pi / 2, dim)

    counter = 0

    for i in range(len(params)):

        if counter == 2:
            counter = 0
            decomposition.append(ms)

        decomposition.append(gate_expand_to_circuit(generic_sud(params[i], dim), n=2, target=counter, dim=dim))

        counter += 1

    return decomposition


def create_ls_instance(P, dim):
    decomposition = []

    params = params_splitter(P, dim)

    if dim == 2:
        theta = np.pi / 2
    elif dim == 3:
        theta = 2 * np.pi / 3
    else:
        theta = np.pi

    ls = ls_gate(theta, dim)

    counter = 0

    for i in range(len(params)):

        if counter == 2:
            counter = 0
            decomposition.append(ls)

        decomposition.append(gate_expand_to_circuit(generic_sud(params[i], dim), n=2, target=counter, dim=dim))
        counter += 1

    return decomposition
