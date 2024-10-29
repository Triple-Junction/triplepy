import numpy as np
import triplepy.sharp_interface_solution as si


def test_stationary_gb():
    # we check 4 different aspect ratios
    gamma_ratios = np.array([0.5, 1.0, 1.5, 1.8])
    # corresponding triple-junction slopes
    tj_slopes = si.calculate_slope(1.0, gamma_ratios)
    # corresponding dihedral angles
    tj_angles = si.calculate_dihedral_angle(1.0, gamma_ratios)
    # the balancing driving force
    driving_forces = 2.0 * tj_slopes / np.sqrt(1.0 + tj_slopes**2)
    # the height-to-width ratio for the circular arc passing through the given triple junctions
    hw_ratios_analytic = (1.0 - np.sin(0.5 * tj_angles)) / (2.0*np.cos(0.5 * tj_angles))
    hw_ratios = []
    rtol = 1.0e-8
    for i, m in enumerate(tj_slopes):
        solver = si.GB_GeometrySolver(m, driving_forces[i])
        geometry = solver.calc_dimensionless_geometry(kind="double", relative_l2_tolerance=rtol)
        width = np.max(geometry["x"]) - np.min(geometry["x"])
        height = np.max(geometry["y"]) - np.min(geometry["y"])
        hw_ratios.append(height / width)
    # we check 1 digit less due to rounding
    assert np.allclose(hw_ratios_analytic, hw_ratios, rtol=10.0 * rtol)


def test_capillary_only_gb():
    # we check 4 different aspect ratios
    gamma_ratios = np.array([0.5, 1.0, 1.5, 1.8])
    # corresponding triple-junction slopes
    tj_slopes = si.calculate_slope(1.0, gamma_ratios)
    # corresponding dihedral angles
    tj_angles = si.calculate_dihedral_angle(1.0, gamma_ratios)
    # the height-to-width ratio according to Eq.~(42) in https://doi.org/10.1016/j.commatsci.2022.111995
    hw_ratios_analytic = np.log(np.sin(0.5 * tj_angles)) / (tj_angles - np.pi)
    hw_ratios = []
    rtol = 1.0e-8
    for m in tj_slopes:
        solver = si.GB_GeometrySolver(m, 0.0)
        geometry = solver.calc_dimensionless_geometry(kind="double", relative_l2_tolerance=rtol)
        width = np.max(geometry["x"]) - np.min(geometry["x"])
        height = np.max(geometry["y"]) - np.min(geometry["y"])
        hw_ratios.append(height / width)
    # we check 1 digit less due to rounding
    assert np.allclose(hw_ratios_analytic, hw_ratios, rtol=10.0 * rtol)


def calc_differential_eq(y_prime, x_spacing, p, v):
    # we test the ODE in the following form which allows to use a relative error on unity
    # y''(x) / (v (1 + (y'(x))^2) - p (1 + (y'(x))^2)^(3/2)) = 1
    y_2prime = np.gradient(y_prime, x_spacing, edge_order=2)
    sqrt_term = np.sqrt(1.0 + y_prime**2)
    return y_2prime / (v*(sqrt_term**2) - p*sqrt_term**3)


def test_general_gbs():
    # we check 4 different aspect ratios
    gamma_ratios = np.array([0.5, 1.0, 1.5, 1.8])
    # corresponding triple-junction slopes
    tj_slopes = si.calculate_slope(1.0, gamma_ratios)

    for m in tj_slopes:
        for p in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
            calc = si.GB_VelocityCalculator(m)
            v = calc.calculate_velocity(p)
            solver = si.GB_GeometrySolver(m, p)
            geometry = solver.calc_dimensionless_geometry(kind="half", relative_l2_tolerance=1e-8)

            # check that the derivative is correct (three digits)
            assert np.allclose(np.gradient(geometry["y"], geometry["x"], edge_order=2), geometry["derivative"], rtol=1e-3)

            # check that ODE is satisfied
            x_min = np.min(geometry["x"])
            x_max = np.max(geometry["x"])
            n_points = len(geometry["x"]) // 2

            x_spacing = (x_max - x_min) / n_points
            x_equidistant = np.linspace(x_min, x_max, n_points)
            yprime = np.interp(x_equidistant, geometry["x"], geometry["derivative"])
            values = calc_differential_eq(yprime, x_spacing, p, v)

            # the accuracy with interpolation is limited to three digits
            assert np.allclose(values, 1.0, rtol=1e-3)
