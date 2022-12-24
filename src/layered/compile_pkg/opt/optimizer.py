from src.layered.compile_pkg.exception import FidelityReachException
from src import global_vars
from scipy.optimize import dual_annealing

from src.layered.compile_pkg.ansatz.ansatz_gen import ls_ansatz, ms_ansatz, cu_ansatz
from src.layered.compile_pkg.opt.distance_measures import fidelity_on_unitares
from src.layered.compile_pkg.ansatz.parametrize import reindex


def bounds_assigner(b1, b2, b3, num_params_single, d):
    assignment = [None] * (num_params_single + 1)

    for m in range(0, d):
        for n in range(0, d):
            if m == n:
                assignment[reindex(m, n, d)] = b3
            elif m > n:
                assignment[reindex(m, n, d)] = b1
            else:
                assignment[reindex(m, n, d)] = b2

    return assignment[:-1]  # dont return last eleement which is just a global phase


def objective_fnc_ms(lambdas):
    ansatz = ms_ansatz(lambdas, global_vars.SINGLE_DIM)

    print(1 - fidelity_on_unitares(ansatz, global_vars.TARGET_GATE))

    if (1 - fidelity_on_unitares(ansatz, global_vars.TARGET_GATE)) < global_vars.OBJ_FIDELITY:
        global_vars.X_SOLUTION = lambdas

        global_vars.FUN_SOLUTION = 1 - fidelity_on_unitares(ansatz, global_vars.TARGET_GATE)

        raise FidelityReachException
    if global_vars.timer_var:
        raise TimeoutError

    return 1 - fidelity_on_unitares(ansatz, global_vars.TARGET_GATE)


def objective_fnc_ls(lambdas):
    ansatz = ls_ansatz(lambdas, global_vars.SINGLE_DIM)

    print(1 - fidelity_on_unitares(ansatz, global_vars.TARGET_GATE))

    if (1 - fidelity_on_unitares(ansatz, global_vars.TARGET_GATE)) < global_vars.OBJ_FIDELITY:
        global_vars.X_SOLUTION = lambdas
        global_vars.FUN_SOLUTION = 1 - fidelity_on_unitares(ansatz, global_vars.TARGET_GATE)

        raise FidelityReachException
    if global_vars.timer_var:
        raise TimeoutError

    return 1 - fidelity_on_unitares(ansatz, global_vars.TARGET_GATE)


def objective_fnc_cu(lambdas):
    ansatz = cu_ansatz(lambdas, global_vars.SINGLE_DIM)

    print(1 - fidelity_on_unitares(ansatz, global_vars.TARGET_GATE))

    if (1 - fidelity_on_unitares(ansatz, global_vars.TARGET_GATE)) < global_vars.OBJ_FIDELITY:
        global_vars.X_SOLUTION = lambdas
        global_vars.FUN_SOLUTION = 1 - fidelity_on_unitares(ansatz, global_vars.TARGET_GATE)

        raise FidelityReachException
    if global_vars.timer_var:
        raise TimeoutError

    return 1 - fidelity_on_unitares(ansatz, global_vars.TARGET_GATE)


def solve_anneal(bounds, ansatz_type, result_queue):
    try:
        if ansatz_type == 'MS':  # MS is 0
            opt = dual_annealing(objective_fnc_ms, bounds=bounds)
        elif ansatz_type == 'LS':  # LS is 1
            opt = dual_annealing(objective_fnc_ls, bounds=bounds)
        elif ansatz_type == 'CU':
            opt = dual_annealing(objective_fnc_cu, bounds=bounds)
        else:
            opt = None

        x = opt.x
        fun = opt.fun

        result_queue.put((fun, x))

    except FidelityReachException as e:
        print("FidelityReachException ", e)
        result_queue.put((global_vars.FUN_SOLUTION, global_vars.X_SOLUTION))

    except TimeoutError as e:
        print("Execution Time Out", e)
        result_queue.put((global_vars.FUN_SOLUTION, global_vars.X_SOLUTION))
