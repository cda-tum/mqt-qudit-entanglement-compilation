import numpy as np
import threading
import queue

from src.layered.compile_pkg.opt.optimizer import bounds_assigner, solve_anneal
from src.layered.compile_pkg.ansatz.parametrize import bound_1, bound_2, bound_3
from src import global_vars


def interrupt_function():
    global_vars.timer_var = True


def binary_search_compile(dimension, max_num_layer, ansatz_type):
    if max_num_layer < 0:
        raise Exception
    counter = 0
    low = 0
    high = max_num_layer

    tol = global_vars.OBJ_FIDELITY

    best_layer, best_error, best_xi = (low + (high - low) // 2, np.inf, [])
    mid, error, xi = (low + (high - low) // 2, np.inf, [])

    # Repeat until the pointers low and high meet each other
    while low <= high:
        print("counter :", counter)
        mid = low + (high - low) // 2
        print("number layer: ", mid)

        error, xi = run(dimension, mid, ansatz_type)

        if error > tol:
            low = mid + 1
        else:
            high = mid - 1
            best_layer, best_error, best_xi = (mid, error, xi)

        counter += 1

    return best_layer, best_error, best_xi


def run(dimension, num_layer, ansatz_type):
    number_unitaries = 2 * (num_layer + 1)
    num_params_single_unitary = -1 + dimension ** 2
    bounds = bounds_assigner(bound_1, bound_2, bound_3, num_params_single_unitary, dimension) * number_unitaries

    duration = 3600 * (global_vars.SINGLE_DIM ** 2 / 4)

    result_queue = queue.Queue()

    thread = threading.Thread(target=solve_anneal, args=(bounds, ansatz_type, result_queue))
    thread.start()

    timer = threading.Timer(duration, interrupt_function)
    timer.start()

    thread.join()
    f, x = result_queue.get()
    # f, x = solve_anneal(bounds, ansatz_type)

    return f, x
