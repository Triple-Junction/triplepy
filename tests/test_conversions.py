import numpy as np
import triplepy.sharp_interface_solution as si


def test_angle_slope():
    theta = np.deg2rad([45.0, 90.0, 110.0, 145.0])
    m = si.calculate_slope_from_dihedral_angle(theta)
    assert np.allclose(si.calculate_dihedral_angle_from_slope(m), theta)


def test_ratio_angle():
    gamma_ratio = np.array([0.5, 1.0, 1.5, 1.8])
    angle = si.calculate_dihedral_angle(1.0, gamma_ratio)
    assert np.allclose(si.calculate_gb_energy_ratio_from_dihedral_angle(angle), gamma_ratio)


def test_ratio_slope():
    gamma_ratio = np.array([0.5, 1.0, 1.5, 1.8])
    m = si.calculate_slope(1.0, gamma_ratio)
    angle = si.calculate_dihedral_angle_from_slope(m)
    assert np.allclose(si.calculate_gb_energy_ratio_from_dihedral_angle(angle), gamma_ratio)
