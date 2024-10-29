import json
from triplepy import ureg, Quantity
from pint import PintError
from triplepy.postprocessing import calc_p, calc_p_pf


def load_default_input_params(
    horizontal_resolution: int = 100, n_cells_int: float = 10.0
):
    simulation_params_default = {
        "gb_mobility": ureg("2e-8 m^4/(J*s)"),
        "gb_energy_horizontal": ureg("0.5 J/m^2"),
        "gb_energy_vertical": ureg("0.5 J/m^2"),
        "driving_force": ureg("1.0e5 Pa"),
        "width": ureg("50.0 um"),
        "horizontal_resolution": horizontal_resolution,
        "n_cells_interface": n_cells_int,
        "time_unit": ureg("1.0 s"),
        "length_unit": ureg("1.0 um").to("m")
    }
    return simulation_params_default


def update_driving_force(sim_params, p_value: float):
    out_params = sim_params.copy()
    p = p_value
    width = sim_params["width"]
    gb_energy_0 = sim_params["gb_energy_horizontal"]
    driving_force = (p * gb_energy_0) / width
    out_params["driving_force"] = driving_force.to("Pa")
    return out_params


def extract_numerical_params(params, unit_system):
    numerical_params = {}
    for parameter in params:
        quantity = params[parameter]
        if isinstance(quantity, Quantity):
            numerical_params[parameter] = quantity.to_preferred(unit_system).magnitude
        else:
            numerical_params[parameter] = quantity
    return numerical_params


def write_sim_params_to_json(sim_params, json_file_path: str, precision: int = 10):
    format_str = "%." + str(precision) + "g"

    out_params = sim_params.copy()
    out_params["p"] = calc_p(sim_params)
    out_params["p_pf"] = calc_p_pf(sim_params)

    # get rid of rounding errors from double precision
    for key in out_params:
        value = out_params[key]
        if isinstance(value, float):
            out_params[key] = float(format_str % out_params[key])

    # format pint quantities on the fly
    def json_formatter(obj):
        if isinstance(obj, Quantity):
            preferred_units = [ureg.J, ureg.m, ureg.s]
            obj = obj.to_preferred(preferred_units)
            value_str = format_str % obj.magnitude
            return value_str + f" {obj.units:~}"
        else:
            raise TypeError(
                f"Object of type {obj.__class__.__name__} is not JSON serializable"
            )

    with open(json_file_path, "w") as json_file:
        json.dump(out_params, json_file, default=json_formatter, indent=4)


def _is_float(obj: str) -> bool:
    try:
        float(obj)
        return True
    except ValueError:
        return False


def load_sim_params_from_json(json_file_path: str):
    with open(json_file_path, "r") as json_file:
        content = json.load(json_file)

    sim_params = {}
    for key, value in content.items():
        # try to convert a string value to a pint object
        if isinstance(value, str):
            # check if str starts with a number
            if _is_float(value.strip().split()[0]):
                # convert to pint object if possible
                try:
                    sim_params[key] = ureg(value)
                except PintError:
                    sim_params[key] = value
        else:
            sim_params[key] = value
    return sim_params
