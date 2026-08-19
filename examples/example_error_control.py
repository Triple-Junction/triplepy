import triplepy.sharp_interface_solution as si

calc = si.GB_GeometrySolver(3.0, 5.0)
# This is our reference solution
geometry = calc.calc_dimensionless_geometry(relative_l2_tolerance=1.0e-8)
print(f"Reference resolution = {len(geometry['x']):d}")

for tolerance in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    print("-------------------------")
    coarse_geometry = calc.calc_dimensionless_geometry(relative_l2_tolerance=tolerance)
    l2_error = si.calc_l2_norm_rel_error(coarse_geometry["x"], coarse_geometry["y"], geometry["x"], geometry["y"])
    print(f"Resolution = {len(coarse_geometry['x']):d}, Tolerance = {tolerance:.2e}, L2_error = {l2_error:.2e}, error_ratio = {l2_error / tolerance:.2g}")
    print(f"Accuracy goal passed? {'YES'}" if l2_error < tolerance else "NO")
