import numpy as np
import triplepy.sharp_interface_solution as analytic
from triplepy.numerical_evaluation import write_dict_to_csv

angles_deg = np.array([60.0, 90.0, 120.0, 150.0])
slopes = analytic.calculate_slope_from_dihedral_angle(np.deg2rad(angles_deg))
ratios = analytic.calculate_gb_energy_ratio_from_dihedral_angle(np.deg2rad(angles_deg))

for i, m in enumerate(slopes):
    # Compute slope from Young's law
    print("Dihedral angle = %.6g" % angles_deg[i])
    print("TJ slope = %.6g" % m)
    print("Energy ratio = %.6g" % ratios[i])

    # Compute non-dimensional driving force
    p = np.linspace(-10, 10, num=501)

    # Compute (non-dimensional) analytical velocity
    gb = analytic.GB_VelocityCalculator(m)
    v_analytic = gb.calculate_velocity(p)

    result_dict = {"dimless_drivingforce": p, "v_analytic": v_analytic, "sqrt1m2_p": np.sqrt(1.0 + m*m) * p}
    write_dict_to_csv(result_dict, "v_analytic_theta%.10g.csv" % angles_deg[i])
    print("------------------------")
