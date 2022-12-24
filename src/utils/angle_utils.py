import numpy as np


def theta_corrector(angle):
    theta_in_units_of_pi = np.mod(abs(angle / np.pi), 4)
    if angle < 0:
        theta_in_units_of_pi = theta_in_units_of_pi * -1
    if abs(theta_in_units_of_pi) < 0.2:
        theta_in_units_of_pi += 4.0
    return theta_in_units_of_pi * np.pi


def phi_cost(theta):
    theta_on_units = theta / np.pi
    err = abs(theta_on_units) * 1e-04
    return err


def theta_cost(theta):
    theta_on_units = theta / np.pi
    err = (4 * abs(theta_on_units) + abs(np.mod(abs(theta_on_units) + 0.25, 0.5) - 0.25)) * 1e-04
    return err
