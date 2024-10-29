import pytest
import numpy as np
import os

import triplepy.vtk_io as vtk
import triplepy.postprocessing as pp

from triplepy.set_input_parameters import load_sim_params_from_json


@pytest.fixture
def root_folder():
    return os.path.join(os.path.dirname(__file__), "../data")


def test_import_vtk(root_folder):
    vtkdata = vtk.import_vtkdata(
        os.path.join(root_folder, "example_vtk_simulation/phia.vtk.series"),
        os.path.join(root_folder, "example_vtk_simulation/phib.vtk.series"),
        os.path.join(root_folder, "example_vtk_simulation/phic.vtk.series"),
    )
    simdata = vtk.load_simdata(
        vtkdata=vtkdata,
        simparams_json=os.path.join(
            root_folder, "example_vtk_simulation/simparams.json"
        ),
    )
    print(simdata)
    result = pp.postprocess_simulation(simdata)
    assert np.isclose(result["scalars"]["v_mean_phase"], -2.130934307)
    assert np.isclose(result["scalars"]["L2_mean"], 0.0434029331)

    lst_GB_sim = pp.calc_simulated_gb_profiles_in_frames(simdata=simdata, first_frame=39, last_frame=40)
    GB_sim = lst_GB_sim[0]
    assert GB_sim["frame_idx"] == 40
    assert np.isclose(GB_sim["L2"], 0.04371279725609258)


def test_import_pvd(root_folder):
    vtkdata = vtk.import_vtkdata(
        os.path.join(root_folder, "example_pvd_simulation/phia.pvd"),
        os.path.join(root_folder, "example_pvd_simulation/phib.pvd"),
        os.path.join(root_folder, "example_pvd_simulation/phic.pvd"),
    )
    simdata = vtk.load_simdata(
        vtkdata=vtkdata,
        simparams_json=os.path.join(
            root_folder, "example_pvd_simulation/simparams.json"
        ),
    )
    result = pp.postprocess_simulation(simdata)
    assert np.isclose(result["scalars"]["v_mean_phase"], -2.130934628)
    assert np.isclose(result["scalars"]["L2_mean"], 0.04387948137)

    lst_GB_sim = pp.calc_simulated_gb_profiles_in_frames(simdata=simdata, first_frame=39, last_frame=40)
    GB_sim = lst_GB_sim[0]
    assert GB_sim["frame_idx"] == 40
    assert np.isclose(GB_sim["L2"], 0.04341047094847948)


def import_vtkdata_micress(res_dir_path, basename):
    # Define dictionary for phase fields
    vtkdata = {"phia": [], "phib": [], "phic": [], "time": []}
    timetable_path = os.path.join(res_dir_path, f"{basename}_VTK_Time.txt")

    timetable = np.loadtxt(timetable_path)
    index_list = timetable[:, 0].astype(int)
    time_list = timetable[:, 1]

    first_file = os.path.join(res_dir_path, f"{basename}_{index_list[0]}.vtk")
    header = vtk.read_vtk_header(first_file)

    vtkdata["grid"] = {
        "delta_x": header["spacing"][0],
        "delta_y": header["spacing"][1],
        "origin_x": header["spacing"][0] * 0.5,
        "origin_y": header["spacing"][1] * 0.5,
    }

    for i, index in enumerate(index_list):
        vtk_file_path = os.path.join(res_dir_path, f"{basename}_{index}.vtk")
        field = vtk.read_vtk_field_data(vtk_file_path)
        # frac1 --> phia
        # frac2 --> phib
        # frac0 --> phic
        vtkdata["phia"].append(field["point_data"]["frac1"])
        vtkdata["phib"].append(field["point_data"]["frac2"])
        vtkdata["phic"].append(field["point_data"]["frac0"])
        vtkdata["time"].append(float(time_list[i]))

    vtkdata["phia"] = np.array(vtkdata["phia"])
    vtkdata["phib"] = np.array(vtkdata["phib"])
    vtkdata["phic"] = np.array(vtkdata["phic"])
    return vtkdata


def test_postprocess_micress_example(root_folder):
    res_dir_path = os.path.join(root_folder, "example_micress_simulation/Results")
    simparams_json = os.path.join(
        root_folder, "example_micress_simulation/simparams_0.json"
    )
    vtkdata = import_vtkdata_micress(
        res_dir_path=res_dir_path, basename="tripleBench_0"
    )
    simdata = vtk.load_simdata(vtkdata=vtkdata, simparams_json=simparams_json)
    results = pp.postprocess_simulation(simdata)
    # assert results["scalars"]["p"] == simdata["input_params"]["p"]
    assert np.isclose(results["scalars"]["v_analytic"], -11.79727890422015)
    assert np.isclose(results["scalars"]["v_mean_phase"], -11.822548437499998)
    assert np.isclose(results["scalars"]["L2_mean"], 0.008544205521747878)

    lst_GB_sim = pp.calc_simulated_gb_profiles_in_frames(simdata=simdata, first_frame=39, last_frame=40)
    GB_sim = lst_GB_sim[0]
    assert GB_sim["frame_idx"] == 40
    assert np.isclose(GB_sim["L2"], 0.008567135279800039)


def test_calc_p(root_folder):
    sim_params = load_sim_params_from_json(
        os.path.join(root_folder, "example_micress_simulation/simparams_0.json")
    )
    p = pp.calc_p(sim_params)
    assert p == sim_params["p"]


def test_calc_p_pf(root_folder):
    sim_params = load_sim_params_from_json(
        os.path.join(root_folder, "example_micress_simulation/simparams_0.json")
    )
    p_pf = pp.calc_p_pf(sim_params)
    assert p_pf == sim_params["p_pf"]
