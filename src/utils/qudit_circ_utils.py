import numpy as np


def calculate_q0_q1(lev, dim):
    q1 = (lev - (lev // dim) * dim)
    q0 = ((lev - (lev - (lev // dim) * dim)) // dim)

    return q0, q1


def on1(gate, d):
    return np.kron(np.identity(d, dtype='complex'), gate)


def on0(gate, d):
    return np.kron(gate, np.identity(d, dtype='complex'))


def gate_expand_to_circuit(gate, n, target, dim=2):
    if n < 1:
        raise ValueError("integer n must be larger or equal to 1")

    if target >= n:
        raise ValueError("target must be integer < integer n")

    upper = [np.identity(dim, dtype='complex') for _ in range((n - target - 1))]
    lower = [np.identity(dim, dtype='complex') for _ in range(target)]
    circ = upper + [gate] + lower
    res = circ[-1]

    for i in reversed(list(range(1, len(circ)))):
        res = np.kron(circ[i - 1], res)

    return res


def apply_gate_to_tlines(gate, n=2, targets=None, dim=2):
    if targets is None:
        targets = range(n)

    if isinstance(targets, int):
        targets = [targets]

    subset_gate = 0
    for i in targets:
        subset_gate += gate_expand_to_circuit(gate, n, i, dim)
    return subset_gate
