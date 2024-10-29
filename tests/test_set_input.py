import triplepy.set_input_parameters as set_input


def test_updata_p():
    default_params = set_input.load_default_input_params()
    new_params = set_input.update_driving_force(default_params, 80)
    new_driving_force = new_params["driving_force"]
    assert new_driving_force.magnitude == 8.0e5
