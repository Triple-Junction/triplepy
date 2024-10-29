import matplotlib.pyplot as plt
import numpy as np
import triplepy.sharp_interface_solution as si
from triplepy.numerical_evaluation import write_dict_to_csv

angle_deg = 120.0
m = si.calculate_slope_from_dihedral_angle(np.deg2rad(angle_deg))

driving_forces = [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0]
result_dict = {}
result_dict["x"] = np.linspace(0.0, 1.0, 1001)

fig, ax1 = plt.subplots()
for p in driving_forces:
    solver = si.GB_GeometrySolver(m, p)
    geometry = solver.calc_dimensionless_geometry(kind="double", relative_l2_tolerance=1e-8)
    key = "y_p%s" % (str(p))
    result_dict[key] = np.interp(result_dict["x"], geometry["x"], geometry["y"])
    ax1.plot(result_dict["x"], result_dict[key], "-", label="p = %.3f" % (p), lw=0.25)


ax1.set_aspect(1)
ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(ax1.get_ylim()[0], 0.0)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=4)
plt.savefig("plot_geometries_theta_%g.pdf" % angle_deg, bbox_inches="tight")

write_dict_to_csv(result_dict, "analytical_geometries_theta_%g.csv" % angle_deg)
