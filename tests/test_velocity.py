import numpy as np
import triplepy.sharp_interface_solution as si


def test_si_velocity():
    driving_forces = np.array([0.1, 1.0, 10.0, 100.0])
    driving_forces = np.concatenate((-np.flip(driving_forces), [0.0], driving_forces))
    # Solutions taken from Wolfram Mathematica
    velocity_solutions_neg = [-115.4700538379255, -11.79727890422015, -2.098316790238965, -1.152126956202861]
    velocity_solution_0 = [-1.047197551196597]
    velocity_solutions_pos = [-0.9423073427275777, 0.0, 9.297707785163341, 99.83162536999436]
    solutions = np.concatenate((velocity_solutions_neg, velocity_solution_0, velocity_solutions_pos))
    calculator = si.GB_VelocityCalculator(si.calculate_slope(1.0, 1.0))
    assert np.allclose(calculator.calculate_velocity(driving_forces), solutions)
