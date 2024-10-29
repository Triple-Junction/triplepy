import triplepy.sharp_interface_solution as si

calc = si.GB_GeometrySolver(3.0, 5.0)
# This is our reference solution
geometry = calc.calc_dimensionless_geometry(relative_l2_tolerance=1.0e-8)
print("Reference resolution = %i" % len(geometry["x"]))

for tolerance in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    print("-------------------------")
    coarse_geometry = calc.calc_dimensionless_geometry(relative_l2_tolerance=tolerance)
    l2_error = si.calc_l2_norm_rel_error(coarse_geometry["x"], coarse_geometry["y"], geometry["x"], geometry["y"])
    print("Resolution = %i, Tolerance = %.2e, L2_error = %.2e, error_ratio = %.2g" % (len(coarse_geometry["x"]), tolerance, l2_error, l2_error / tolerance))
    print("Accuracy goal passed? %s" % "YES" if l2_error < tolerance else "NO")
