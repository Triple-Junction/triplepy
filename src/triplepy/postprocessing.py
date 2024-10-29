import numpy as np

import triplepy.numerical_evaluation as numerics
import triplepy.sharp_interface_solution as analytic


def calc_p(sim_params):
    p = (
        sim_params["driving_force"]
        * sim_params["width"]
        / sim_params["gb_energy_horizontal"]
    )
    return float(p)


def calc_m(sim_params):
    m = analytic.calculate_slope(
        sim_params["gb_energy_horizontal"], sim_params["gb_energy_vertical"]
    )
    return float(m)


def calc_p_pf(sim_params):
    delta_x = sim_params["width"] / sim_params["horizontal_resolution"]
    eta = delta_x * sim_params["n_cells_interface"]
    p_pf = sim_params["driving_force"] * eta / sim_params["gb_energy_horizontal"]
    return round(float(p_pf), 10)


def calc_transient_time(sim_params):
    transient_time = (sim_params["width"]) ** 2 / (
        sim_params["gb_mobility"] * sim_params["gb_energy_horizontal"]
    )
    return transient_time


def calc_velocity_dimensional(v_dimless, sim_params):
    mobility = sim_params["gb_mobility"]
    gb_energy = sim_params["gb_energy_horizontal"]
    width = sim_params["width"]
    velocity = (v_dimless * mobility * gb_energy / width).to_base_units()
    return velocity


def calc_analytic_velocity_nondimensional(m, p):
    # Compute (non-dimensional) analytical velocity
    gb = analytic.GB_VelocityCalculator(m)
    v_analytic = gb.calculate_velocity(p)
    return v_analytic


def calc_dimless_velocity_in_frames(
    simdata, first_frame: int = 1, last_frame: int = None
):
    v_left = []
    v_right = []
    v_triple = []
    v_phase = []
    if last_frame is None:
        last_frame = simdata["phia"].shape[0] - 1

    # Note that in the micress vtk data, the first frame are at time 0
    # thus we start with frame 1
    lst_frame_idx = []
    for frame in range(first_frame + 1, last_frame + 1):
        v = numerics.calculate_velocity_nondimensional(simdata, frame - 1, frame)
        v_left.append(v["left_boundary"])
        v_right.append(v["right_boundary"])
        v_triple.append(v["triple_point"])
        v_phase.append(v["phase_fraction"])
        lst_frame_idx.append(frame)
    v_dimless = {}
    v_dimless["frame_indices"] = np.array(lst_frame_idx)
    v_dimless["v_left"] = np.array(v_left)
    v_dimless["v_right"] = np.array(v_right)
    v_dimless["v_triple"] = np.array(v_triple)
    v_dimless["v_phase"] = np.array(v_phase)
    return v_dimless


def calc_analytic_grain_boundary(m, p):
    solver = analytic.GB_GeometrySolver(m, p)
    GB_analytic = solver.calc_dimensionless_geometry(
        kind="double", relative_l2_tolerance=1e-7
    )

    GB_x_analytic = GB_analytic["x"] - np.min(GB_analytic["x"])
    GB_y_analytic = GB_analytic["y"] - np.max(GB_analytic["y"])
    GB_height_analytic = np.max(GB_y_analytic) - np.min(GB_y_analytic)
    return GB_x_analytic, GB_y_analytic, GB_height_analytic


def calc_simulated_gb_profiles_in_frames(
    simdata, first_frame: int = 1, last_frame: int = None
):
    if last_frame is None:
        last_frame = simdata["phia"].shape[0] - 1

    # Compute analytical GB profile
    m = calc_m(simdata["input_params"])
    p = calc_p(simdata["input_params"])
    GB_x_analytic, GB_y_analytic, GB_height_analytic = calc_analytic_grain_boundary(
        m, p
    )

    # Evaluate simulated profile in frames
    GB_sim_in_frames = []
    for i, frame in enumerate(range(first_frame + 1, last_frame + 1)):
        if i == 0:
            _, GB_y_sim_old = numerics.extract_grainboundary_profile(simdata, frame - 1)
            height_old = np.max(GB_y_sim_old) - np.min(GB_y_sim_old)
        else:
            height_old = GB_sim_in_frames[i - 1]["height"]
        GB_x_sim, GB_y_sim = numerics.extract_grainboundary_profile(simdata, frame)
        GB_sim = {}
        GB_sim["frame_idx"] = frame
        GB_sim["x"] = GB_x_sim
        GB_sim["y_sim"] = GB_y_sim
        GB_sim["height"] = np.max(GB_y_sim) - np.min(GB_y_sim)
        # Quantify convergence through change in GB profile height
        GB_sim["rel_height_change"] = (
            GB_sim["height"] - height_old
        ) / GB_height_analytic
        # Quantify relative error in GB profile height
        GB_sim["rel_height_error"] = (
            GB_sim["height"] - GB_height_analytic
        ) / GB_height_analytic
        # Quantify deviation of GB profile in terms of L2-Norm
        if GB_x_sim.size > 1:
            GB_y_analytic_interp = np.interp(GB_x_sim, GB_x_analytic, GB_y_analytic)
            diff2 = (GB_y_sim - GB_y_analytic_interp) ** 2
            GB_sim["L2"] = np.sqrt(
                np.trapz(diff2, x=GB_x_sim)
                / np.trapz(GB_y_analytic_interp**2, x=GB_x_sim)
            )
        else:
            GB_y_analytic_interp = np.nan
            GB_sim["L2"] = np.nan
        GB_sim["y_analytic_interp"] = GB_y_analytic_interp
        GB_sim_in_frames.append(GB_sim)
    return GB_sim_in_frames


def postprocess_simulation(
    simdata,
    first_frame: int = None,
    last_frame: int = None,
    display_message: bool = True,
):
    """
    Postprocesses simulation data to evaluate and summarize simulation results.

    Args:
        simdata (dict): Simulation data containing grid info, input parameters, 2D field values, and time series.
            The structure of simdata is as follows:
            - "grid": dict with grid information like dx, dy, and origin of first grid point/cell center.
            - "input_params": dict of physical simulation input parameters; can be read from a JSON file.
            - "phia", "phib", "phic": time-series of 2D field values saved as numpy arrays
              with dimensions (number of frames, points in x, points in y).
            - "time": list of timeframes.
        first_frame (int, optional): Index of the first frame to consider for analysis. Defaults to None.
        last_frame (int, optional): Index of the last frame to consider for analysis. Defaults to None.
        display_message (bool, optional): Flag to control the display of messages. Defaults to True.

    Raises:
        ValueError: If the first_frame is greater than or equal to the last_frame.

    Returns:
        dict: A dictionary containing geometry data and scalar results.
    """
    # determine evaluation frame range
    # From dimensionality analysis we compute the following transient time
    transient_time = calc_transient_time(simdata["input_params"])

    # The simulation should at least be run the transient time
    # to ensure convergence of the system.
    dimless_times = np.array([float(t / transient_time) for t in simdata["time"]])
    if (first_frame is None) or (last_frame is None):
        frame_start = np.where(dimless_times >= 0.5)[0][0]
        frame_end = np.where(dimless_times == 1.0)[0][0]
    elif first_frame >= last_frame:
        raise ValueError(
            "The first frame number should be smaller than the last frame number."
        )
    else:
        frame_start = first_frame
        frame_end = last_frame

    # Compute analytical velocity
    # Compute slope from Young's law
    m = calc_m(simdata["input_params"])
    # Compute non-dimensional driving force
    p = calc_p(simdata["input_params"])
    v_analytic = calc_analytic_velocity_nondimensional(m, p)
    if display_message:
        print(f"TJ slope = {m:g}")
        print(f"Dimensionless driving force = {p:g}")
        print(f"Analytic velocity = {v_analytic:g}")
        print("\n")

    # Evaluate velocities in frames
    v_dimless = calc_dimless_velocity_in_frames(
        simdata=simdata, first_frame=frame_start, last_frame=frame_end
    )
    v_left = v_dimless["v_left"]
    v_phase = v_dimless["v_phase"]

    # Evaluation of gb profiles
    GB_sim_in_frames = calc_simulated_gb_profiles_in_frames(
        simdata=simdata, first_frame=frame_start, last_frame=frame_end
    )
    # Quantify deviations
    lst_L2 = [GB_sim["L2"] for GB_sim in GB_sim_in_frames]
    lst_L2 = np.array(lst_L2)
    lst_rel_height_change = np.array(
        [GB_sim["rel_height_change"] for GB_sim in GB_sim_in_frames]
    )
    lst_rel_height_error = np.array(
        [GB_sim["rel_height_error"] for GB_sim in GB_sim_in_frames]
    )

    # final geometry
    GB_sim_final = GB_sim_in_frames[-1]
    geometry_dict = {}
    geometry_dict["x"] = GB_sim_final["x"]
    geometry_dict["y_sim"] = GB_sim_final["y_sim"]
    geometry_dict["y_analytic"] = GB_sim_final["y_analytic_interp"]

    results = {
        "geometries": geometry_dict,
        "scalars": {
            "p": p,
            "v_analytic": v_analytic,
            "v_mean_left": np.mean(v_left),
            "v_std_left": np.std(v_left),
            "v_mean_phase": np.mean(v_phase),
            "v_std_phase": np.std(v_phase),
            "height_change_mean": np.mean(lst_rel_height_change),
            "height_change_std": np.std(lst_rel_height_change),
            "height_error_mean": np.mean(lst_rel_height_error),
            "height_error_std": np.std(lst_rel_height_error),
            "L2_mean": np.mean(lst_L2),
            "L2_std": np.std(lst_L2),
        },
    }
    return results


def make_results_table(simulations, keys):
    table = {key: [] for key in keys}
    for sim in simulations:
        for key in keys:
            table[key].append(sim[key])
    return table
