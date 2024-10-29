import numpy as np
from skimage import measure
import warnings
import csv


# limiting the precision to 10 digits seems reasonable
# we do this in order to get rid of double-precision artifacts
def _formatter(x):
    if isinstance(x, float):
        return "%.10g" % x
    return x


def write_dict_to_csv(dictionary: dict, filename: str, delimiter: str = ",") -> None:
    """
     Write a dictionary to a text file.

     Parameters
     ----------
         dictionary : dict
            The dictionary to be written.
         filename : str
            The name of the output file.
         delimiter : str
            The delimiter to use between fields (default is ",").

     Returns:
         None
    """

    with open(filename, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[key for key in dictionary], delimiter=delimiter)
        writer.writeheader()
        # use the formatter to convert floating point numbers into a string
        for i in range(len(dictionary[list(dictionary.keys())[0]])):
            writer.writerow({key: _formatter(dictionary[key][i]) for key in dictionary})


def _select_isoline(isoline, search_pos):
    idx = [index for index, array in enumerate(isoline) if np.any((array.T)[0] == search_pos)]
    if len(idx) == 0:
        if len(isoline) < 2:
            isoline = (np.squeeze(isoline)).T
        else:
            isoline = (max(isoline, key=lambda x: len(x))).T
    elif len(idx) == 1:
        # One isoline fullfills the criterion -> we're done
        isoline = isoline[idx[0]].T
    else:
        # Multiple isolines fullfill the criterion -> use largest one
        isoline_red = []
        for index in idx:
            isoline_red.append(isoline[idx[index]])
        isoline = (max(isoline_red, key=lambda x: len(x))).T
    return isoline


def extract_isolines(simdata: dict, frame: int):
    """
     Extracts three distinct isolines for a given frame index

        Parameters
        ----------
        simdata : dict
            Simulation data containing a list of numpy arrays for keys "phia", "phib" and "phic"
        frame : int
            Frame index (zero based)

        Returns
        -------
        iso_ab, iso_ac, iso_bc
            Three isolines phia == phib, phia == phic, phib == phic
    """

    phia = simdata["phia"][frame]
    phib = simdata["phib"][frame]
    phic = simdata["phic"][frame]

    iso_ab = measure.find_contours((phia-phib), 0.0)
    iso_ac = measure.find_contours((phia-phic), 0.0)
    iso_bc = measure.find_contours((phib-phic), 0.0)

    # The marching squares might return a list with multiple disconnected lines
    # and we try to extract the desired one.
    # Isoline ac should have a x=0 entry
    iso_ac = _select_isoline(iso_ac, 0)

    # Isoline bc should have a x=Nx-1 entry
    iso_bc = _select_isoline(iso_bc, phib.shape[0] - 1)

    # For iso_ab we don't have a good criterion and this isoline is less
    # important -> use largest list entry
    if len(iso_ab) < 2:
        iso_ab = (np.squeeze(iso_ab)).T
    else:
        iso_ab = (max(iso_ab, key=lambda x: len(x))).T

    # NOTE: assumes STRUCTURED_POINTS with equidistant spacings
    # offset and delta x in dimensionless units
    x_offset = float(simdata["grid"]["origin_x"] / simdata["input_params"]["width"])
    y_offset = float(simdata["grid"]["origin_y"] / simdata["input_params"]["width"])
    dx = float(simdata["grid"]["delta_x"] / simdata["input_params"]["width"])
    dy = float(simdata["grid"]["delta_y"] / simdata["input_params"]["width"])

    if iso_ab.size != 0:
        iso_ab = np.stack((iso_ab[0] * dx + x_offset, iso_ab[1]*dy+y_offset), axis=1).T
    if iso_ac.size != 0:
        iso_ac = np.stack((iso_ac[0] * dx + x_offset, iso_ac[1] * dy + y_offset), axis=1).T
    if iso_bc.size != 0:
        iso_bc = np.stack((iso_bc[0] * dx + x_offset, iso_bc[1] * dy + y_offset), axis=1).T

    return iso_ab, iso_ac, iso_bc


def _split_half(x, y):
    """ Splits two given one-dimensional arrays of x- and y- coordinates by half"""

    if len(x) != 2:
        x_split = np.array_split(x, 2)
        y_split = np.array_split(y, 2)
        # duplicate node at which the array is split
        x_split[1] = np.concatenate(([x_split[0][-1]], x_split[1]))
        y_split[1] = np.concatenate(([y_split[0][-1]], y_split[1]))
    else:
        x_split = np.array([x])
        y_split = np.array([y])
    return x_split, y_split


def _is_overlap_1d(x1, x2) -> bool:
    """Checks if two one-dimensional arrays have overlapping intervals"""

    min_x1, min_x2 = np.min(x1), np.min(x2)
    max_x1, max_x2 = np.max(x1), np.max(x2)
    return not (max_x1 < min_x2 or max_x2 < min_x1)


def _is_overlap(x1, y1, x2, y2) -> bool:
    """Checks if two contours have overlapping intervals"""

    return (_is_overlap_1d(x1, x2) and _is_overlap_1d(y1, y2))


def _solve_intersect(segment1, segment2):
    """
    Calculates the intersection between two straight line-segments (each given by a start and end point)

        Parameters
        ----------
        segment1 : Two dimensional array
            Containg start (0-index) and end (1-index) points of first segment
        segment2 : Two dimensional array
            Containg start (0-index) and end (1-index) points of second segment

        Returns
        -------
        numpy array
            Point of intersection
    """

    x0 = np.array(segment1[0])
    t0 = np.array(segment1[1]) - np.array(segment1[0])
    # normalize
    length_0 = np.linalg.norm(t0)
    t0 = t0 / length_0

    x1 = np.array(segment2[0])
    t1 = np.array(segment2[1]) - np.array(segment2[0])
    # normalize
    length_1 = np.linalg.norm(t1)
    t1 = t1 / length_1
    # rhs
    r0 = np.dot(t0, x1-x0)
    r1 = np.dot(t1, x1-x0)

    t0t1 = np.dot(t0, t1)

    det_M = 1.0 - t0t1*t0t1
    # both segments are colinear? - intersection ill defined
    if abs(det_M) < 1.0e-12:
        return None

    lambda_0 = (r0 - r1*t0t1) / det_M
    lambda_1 = (r0*t0t1 - r1) / det_M

    # check if intersection is within the bounds of the sections -> none if outside
    if (lambda_0 < 0.0 or lambda_0 > length_0):
        return None
    if (lambda_1 < 0.0 or lambda_1 > length_1):
        return None

    # interpolate between the two nodes
    return x0 + lambda_0*t0


def _calc_intersection(x1, y1, x2, y2, intersections) -> None:
    """Recursively calculates intersections for two given contours"""

    if len(x1) == 2 and len(x2) == 2:
        segment1 = [[x1[0], y1[0]], [x1[1], y1[1]]]
        segment2 = [[x2[0], y2[0]], [x2[1], y2[1]]]
        intersect = _solve_intersect(segment1, segment2)
        if intersect is not None:
            if len(intersections) != 0:
                if not np.allclose(intersect, intersections[-1]):
                    intersections.append(intersect)
            else:
                intersections.append(intersect)
    else:
        # one of the curves is not yet a linear segment
        # hence we split them by half
        x1_parts, y1_parts = _split_half(x1, y1)
        x2_parts, y2_parts = _split_half(x2, y2)
        # figure out which of the parts overlap
        for i in range(len(x1_parts)):
            for j in range(len(x2_parts)):
                # if they overlap, there may be intersections
                if _is_overlap(x1_parts[i], y1_parts[i], x2_parts[j], y2_parts[j]):
                    _calc_intersection(x1_parts[i], y1_parts[i], x2_parts[j], y2_parts[j], intersections)


def calc_intersection(x1, y1, x2, y2):
    """
    Calculates intersections between two given contours.

        Parameters
        ----------
        x1 : array_like
            x-coordinates for first contour
        y1 : array_like
            y-coordinates for first contour
        x2 : array_like
            x-coordinates for second contour
        y2 : array_like
            y-coordinates for second contour

        Returns
        -------
        numpy array
            Two-dimensional array containing x- (0-index) and y-coordinates (1-index)
    """
    if len(y1) != len(x1):
        raise ValueError("Size of x1 unequal to y1")
    if len(x2) != len(y2):
        raise ValueError("Size of x2 unequal to y2")

    intersections = []
    _calc_intersection(x1, y1, x2, y2, intersections)
    return np.array([p[0] for p in intersections]), np.array([p[1] for p in intersections])


def _phase_fraction(phase_field):
    return np.sum(phase_field)/(phase_field.size)


def calculate_velocity_nondimensional(simdata: dict, frame1: int, frame2: int):
    """
    Compute velocity of the grain boundary from isolines between phase-fields.
    This function returns non-dimensional velocities.
    Computation is based on backward Euler difference scheme.

    Parameters
    ----------
    simdata : dict
        Simulation data
    frame1 : int
        Start frame index
    frame2 : int
        End frame index

    Returns
    -------
    dict
        4 velocities which are differently computed
        "left_boundary": phia=phic isoline at left boundary,
        "right_boundary": phib=phic isoline at right boundary,
        "triple_point": calculated from intersection of isolines phia==phic and phib==phic,
        "phase_fraction": based on phase evolution of phic.
    """

    if frame1 > frame2:
        print("Frame 1 must be smaller than frame 2 for correct computation of velocity.")
        return None

    times = []
    ymax_left = []
    ymax_right = []
    tp_xy = []
    fraction_c = []

    for frame in [frame1, frame2]:
        iso_ab, iso_ac, iso_bc = extract_isolines(simdata, frame)

        # Check if some isolines are empty sets
        if iso_ac.size != 0:
            ymax_left.append(np.max(iso_ac[1]))
        else:
            ymax_left.append(np.nan)
            warnings.warn("Isoline iso_ac is an empty set!")
        if iso_bc.size != 0:
            ymax_right.append(np.max(iso_bc[1]))
        else:
            ymax_right.append(np.nan)
            warnings.warn("Isoline iso_bc is an empty set!")

        if iso_ac.size != 0 and iso_bc.size != 0:
            # Compute triple point position
            tp_x, tp_y = calc_intersection(iso_ac[0], iso_ac[1], iso_bc[0], iso_bc[1])
            # Check if isolines intersect or yield empty set
            if tp_x.size != 0:
                tp_xy.append([tp_x[0], tp_y[0]])
            else:
                warnings.warn("No triple point found!")
                tp_xy.append([np.nan, np.nan])
        else:
            tp_xy.append([np.nan, np.nan])
        # Compute phase fraction of phic
        fraction_c.append(_phase_fraction(simdata["phic"][frame]))
        times.append(simdata["time"][frame])

    input_params = simdata["input_params"]
    factor = float((input_params["width"]**2 / (input_params["gb_mobility"] * input_params["gb_energy_horizontal"])) / (times[1] - times[0]))

    velocity = {}
    velocity["left_boundary"] = (ymax_left[1] - ymax_left[0]) * factor
    velocity["right_boundary"] = (ymax_right[1] - ymax_right[0]) * factor
    velocity["triple_point"] = (tp_xy[1][1] - tp_xy[0][1]) * factor

    dy = float(simdata["grid"]["delta_y"] / simdata["input_params"]["width"])
    velocity["phase_fraction"] = -(fraction_c[1]-fraction_c[0]) * (simdata["phic"][0].shape[1]*dy) * factor

    return velocity


def extract_grainboundary_profile(simdata: dict, frame: int):
    """
    Compute the grain boundary profile from isolines between phase-fields.

    Parameters
    ----------
    simdata : dict
        Simulation data
    frame : int
        Frame index (0-based)

    Returns
    -------
    x, y
        Two one-dimensional arrays containing the dimensionless x- and y-coordinates of the profile.
    """

    _, iso_ac, iso_bc = extract_isolines(simdata, frame)

    if iso_ac.size != 0 and iso_bc.size != 0:
        # dimensionless delta x
        dx = float(simdata["grid"]["delta_x"] / simdata["input_params"]["width"])

        # Compute triple point position
        tp_x, tp_y = calc_intersection(iso_ac[0], iso_ac[1], iso_bc[0], iso_bc[1])
        # Check if isolines intersect or yield empty set
        if tp_x.size == 0:
            warnings.warn("Triple point is ill-defined!")
            # Assume tp_x is in the middle of simulation domain
            tp_x = 0.5

        xs = np.concatenate((iso_ac[0, iso_ac[0, :] < (tp_x - 0.01*dx)],
                             iso_bc[0, iso_bc[0, :] > (tp_x + 0.01*dx)]), axis=0)
        ys = np.concatenate((iso_ac[1, iso_ac[0, :] < (tp_x - 0.01*dx)],
                             iso_bc[1, iso_bc[0, :] > (tp_x + 0.01*dx)]), axis=0)

        if xs.size != 0:
            # Only return values that are inside the physical domain
            ys = ys[(xs >= 0.0) & (xs <= 1.0)]
            xs = xs[(xs >= 0.0) & (xs <= 1.0)]

            # Shift such that highest point is at y=0
            ys -= np.max(ys)
        else:
            warnings.warn("Something fishy here!")
            xs = np.array([-1])
            ys = [np.nan]

    else:
        warnings.warn("Isoline iso_ac or iso_bc is an empty set!")
        xs = np.array([-1])
        ys = np.nan

    return xs, ys
