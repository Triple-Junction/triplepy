import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import json
from xml.dom import minidom
import numpy as np
import warnings
from triplepy.set_input_parameters import load_sim_params_from_json


def read_vtk_header(filename):
    header = {}

    if filename.endswith(".vtk"):
        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(filename)
        reader.ReadAllScalarsOn()
    else:
        # vti format
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(filename)

    reader.Update()
    data = reader.GetOutput()

    header["origin"] = data.GetOrigin()
    header["spacing"] = data.GetSpacing()

    return header


def read_vtk_field_data(filename):
    fields = {"point_data": {}, "cell_data": {}}

    if filename.endswith(".vtk"):
        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(filename)
        reader.ReadAllScalarsOn()
    else:
        # vti format
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(filename)

    reader.Update()
    data = reader.GetOutput()

    # read all point data
    for i in range(data.GetPointData().GetNumberOfArrays()):
        array = data.GetPointData().GetAbstractArray(i)
        fields["point_data"][array.GetName()] = vtk_to_numpy(array).reshape(data.GetDimensions(), order="F").squeeze()

    # read all cell data
    for i in range(data.GetCellData().GetNumberOfArrays()):
        cell_dimensions = [n - 1 if n != 1 else n for n in data.GetDimensions()]
        array = data.GetCellData().GetAbstractArray(i)
        fields["cell_data"][array.GetName()] = vtk_to_numpy(array).reshape(cell_dimensions, order="F").squeeze()

    return fields


def read_vtk_file(filename):
    print("Reading file \"%s\"" % filename)
    result_dict = {}
    header_dict = read_vtk_header(filename)
    fields_dict = read_vtk_field_data(filename)
    result_dict = {**header_dict, **fields_dict}

    return result_dict


def read_vtk_time_series(series_filename):
    simpath = os.path.dirname(series_filename)

    with open(series_filename) as json_file:
        vtk_series = json.load(json_file)

    frames = {"time": [], "data": []}
    for vtk_file in vtk_series["files"]:
        filename = vtk_file["name"]
        if not os.path.isabs(filename):
            filename = os.path.join(simpath, filename)

        vtk_data = read_vtk_file(filename)
        frames["data"].append(vtk_data)
        frames["time"].append(float(vtk_file["time"]))
    return frames


def read_pvd_time_series(pvd_filename):
    simpath = os.path.dirname(pvd_filename)

    doc = minidom.parse(pvd_filename)
    vtkfile = doc.getElementsByTagName("VTKFile")[0]
    collection = vtkfile.getElementsByTagName("Collection")[0]
    datasets = collection.getElementsByTagName("DataSet")

    frames = {"time": [], "data": []}
    for dataset in datasets:
        if not dataset.hasAttribute("timestep"):
            raise IOError("File \"%s\" has a missing timestep" % pvd_filename)

        if not dataset.hasAttribute("file"):
            raise IOError("File \"%s\" has a missing file" % pvd_filename)

        if not dataset.hasAttribute("part"):
            raise IOError("File \"%s\" has a missing part" % pvd_filename)

        if int(dataset.getAttribute("part")) != 0:
            raise IOError("Only a single part supported in pvd file")

        filename = dataset.getAttribute("file")
        if not os.path.isabs(filename):
            filename = os.path.join(simpath, filename)

        vtk_data = read_vtk_file(filename)
        frames["data"].append(vtk_data)
        frames["time"].append(float(dataset.getAttribute("timestep")))

    return frames


def read_time_series(filename):
    if filename.endswith(".vtk.series"):
        return read_vtk_time_series(filename)
    elif filename.endswith(".pvd"):
        return read_pvd_time_series(filename)
    else:
        raise IOError("Unknown file format of file %s" % filename)


def extract_field_data(vtk_data):
    origin_pointdata = np.array(vtk_data[0]["origin"])
    spacing = np.array(vtk_data[0]["spacing"])
    origin_celldata = origin_pointdata + 0.5 * spacing

    out_data = {}
    for point_data in vtk_data[0]["point_data"]:
        out_data[point_data] = {"origin": origin_pointdata, "spacing": spacing, "data": []}

    for cell_data in vtk_data[0]["cell_data"]:
        out_data[cell_data] = {"origin": origin_celldata, "spacing": spacing, "data": []}

    for data in vtk_data:
        if not np.allclose(data["origin"], origin_pointdata):
            raise IOError("Inconsistent origins within file series")
        if not np.allclose(data["spacing"], spacing):
            raise IOError("Inconsistent spacings within file series")

        for point_data, array in data["point_data"].items():
            out_data[point_data]["data"].append(array)

        for cell_data, array in data["cell_data"].items():
            out_data[cell_data]["data"].append(array)

    return out_data


def get_first_field(field_data):
    if len(list(field_data.keys())) > 1:
        warnings.warn("Results ambiguous since multiple fields inside vtk file")
    return field_data[list(field_data.keys())[0]]


def import_vtkdata(phia_vtk, phib_vtk, phic_vtk):
    phia = read_time_series(phia_vtk)
    phia_field = get_first_field(extract_field_data(phia["data"]))

    origin = phia_field["origin"]
    spacing = phia_field["spacing"]
    vtkdata = {"time": phia["time"], "phia": phia_field["data"]}

    vtks = {"phib": phib_vtk, "phic": phic_vtk}
    for phase, filename in vtks.items():
        time_series = read_time_series(filename)
        if not np.allclose(time_series["time"], vtkdata["time"]):
            raise IOError("Inconsistent times among phases")

        field = get_first_field(extract_field_data(time_series["data"]))

        if not np.allclose(field["origin"], origin):
            raise IOError("Inconsistent origins among phases")

        if not np.allclose(field["spacing"], spacing):
            raise IOError("Inconsistent spacings among phases")

        vtkdata[phase] = field["data"]

    vtkdata["grid"] = {
        "delta_x": spacing[0],
        "delta_y": spacing[1],
        "origin_x": origin[0],
        "origin_y": origin[1],
    }
    return vtkdata


def load_simdata(vtkdata, simparams_json):
    sim_params = load_sim_params_from_json(simparams_json)
    simdata = vtkdata.copy()
    simdata["time"] = [t * sim_params["time_unit"] for t in vtkdata["time"]]
    simdata["grid"] = {
        "delta_x": vtkdata["grid"]["delta_x"] * sim_params["length_unit"],
        "delta_y": vtkdata["grid"]["delta_y"] * sim_params["length_unit"],
        "origin_x": vtkdata["grid"]["origin_x"] * sim_params["length_unit"],
        "origin_y": vtkdata["grid"]["origin_y"] * sim_params["length_unit"],
    }
    simdata["input_params"] = sim_params

    return simdata
