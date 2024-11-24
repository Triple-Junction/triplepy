# TriplePy - Triple junction benchmark

## Description
This repo contains the post-processing code for the corresponding publication
_**"Triple junction benchmark for multiphase-field models combining capillary and bulk driving forces"**_
[DOI: 10.1088/1361-651X/ad8d6f](https://doi.org/10.1088/1361-651X/ad8d6f)

## Installation
Run `pip install git+https://github.com/triple-junction/triplepy.git#egg=triplepy` for installation.

## Computing sharp-interface steady-state solutions
Given a triple junction slope `<tj-slope>` and a dimensionless driving force `<dimensionless-driving-force>`, sharp interface solutions can be computed as follows:

### Grain-boundary velocity
```python3
import triplepy.sharp_interface_solution as si

calc = si.GB_VelocityCalculator(<tj-slope>)
v = calc.calculate_velocity(<dimensionless-driving-force>)
```

### Grain-boundary geometry
```python3
import triplepy.sharp_interface_solution as si

solver = si.GB_GeometrySolver(<tj-slope>, <dimensionless-driving-force>)
geometry = solver.calc_dimensionless_geometry(kind="double", relative_l2_tolerance=1e-8)
```
`geometry` is a dictionary containing keys `"x"` `"y"` and `"derivative"` each referring to numpy-arrays of equal size corresponding to dimensionless $x$ and $y$ coordinates as well as the first derivative $y'(x)$.
Take a look at `tests` and `examples` for more examples.

## Postprocessing a simulation
In order to postprocess a simulation simply run
```python3
import triplepy.postprocessing as tri
# simdata = <custom code>
result = tri.postprocess_simulation(simdata)
```
`simdata` is an input dictionary of the following format:
```python3
{
"time": # list of times as pint quantity,
"phia": # list of 2D numpy array containing phase field "a" for each time frame,
"phib": # list of 2D numpy array containing phase field "b" for each time frame,
"phic": # list of 2D numpy array containing phase field "c" for each time frame,
"grid": # dictionary containing grid origin and spacing information,
"input_params": # simulation input parameters (usually parsed from json)
}
```
This is the minimal information necessary to use the subsequent automized simulation evaluation.
The dictionaries must mostly contain `pint` quantities, e.g.:
```python3
{
"time": [<Quantity(0.0, 'second')>, <Quantity(0.00625, 'second')>, ... ,<Quantity(0.25, 'second')>],
"phia": # numpy array with shape (11,100,800),
"phib": # numpy array with shape (11,100,800),
"phic": # numpy array with shape (11,100,800),
"grid": {'delta_x': <Quantity(1e-06, 'meter')>,'delta_y': <Quantity(1e-06, 'meter')>, 'origin_x': <Quantity(-5e-07, 'meter')>, 'origin_y': <Quantity(-5e-07, 'meter')>},
"input_params": {'gb_mobility': <Quantity(2e-08, 'meter ** 4 / joule / second')>, 'gb_energy_horizontal': <Quantity(0.5, 'joule / meter ** 2')>, 'gb_energy_vertical': <Quantity(0.5, 'joule / meter ** 2')>, 'driving_force': <Quantity(-10000.0, 'joule / meter ** 3')>}
}
```
These are the mandatory input keys.

## Simulation file format
The codes that were used for the benchmark publication are based on finite difference stencils on equidistant grids.
Thus, we assume that the three field variables `phia`, `phib` and `phic` (in paper referred to as $\phi_0$) can be extracted as STRUCTURED_POINTS data in a .vtk format or directly as numpy arrays.

### VTK input
Vtk file format is supported, i.e. no extra python code is necessary to analyze such a dataset.
The vtk files for a simulation time series (either in `.pvd` or in `.vtk.series` format) need to be placed together with a `simparams.json` file of the following format.
Each key is parsed using `pint`, i.e. units must be provided within the value string, e.g.:
```json
{
    "gb_mobility": "2e-08 m ** 4 / J / s",
    "gb_energy_horizontal": "0.5 J / m ** 2",
    "gb_energy_vertical": "0.5 J / m ** 2",
    "driving_force": "-10000 J / m ** 3",
    "width": "5e-05 m",
    "time_unit": "0.001 s",
    "length_unit": "1e-06 m"
}
```
`gb_mobility` is the grain-boundary mobility of all GBs, `gb_energy_*` define the energies of the initially vertical and horizontal GBs and `driving_force` is the driving force acting on the horizontal (i.e. moving) GBs.
`width` defines the width of the simulation domain (additional ghost/or padding cells are thus correctly handled as such).
Please make sure that the origin is correctly set in the vtk files. Only the interval $0<x<\text{width}$ is used for extracting the GB geometry.
`time_unit` and `length_unit` define the time and length unit implicitly used within the vtk files.
The standard procedure is as follows:
```python3
import os
import triplepy.vtk_io as vtk
import triplepy.postprocessing as tri

root_folder = <your rootfolder>

vtkdata = vtk.import_vtkdata(
    os.path.join(root_folder, "<your subdir>/phia.vtk.series" #or .pvd),
    os.path.join(root_folder, "<your subdir>/phib.vtk.series" #or .pvd),
    os.path.join(root_folder, "<your subdir>/phic.vtk.series" #or .pvd),
)

simdata = vtk.load_simdata(
    vtkdata=vtkdata,
    simparams_json=os.path.join(
        root_folder, "<your subdir>/simparams.json"
    ),
)

result = tri.postprocess_simulation(simdata)
```


Take a look at `data` folder for example datasets.
### Other file formats
For other file formats you have to make sure to provide valid `simdata` as described above. 
This is the minimal information necessary to use the subsequent automized simulation evaluation.
Additional information can, of course, be added according to personal needs and preferences. The json file which is read as `"input_params"` e.g. might include interfacial resolution ($\eta=10\Delta x$) or model specific details.

## Authors and acknowledgment
The code has been developed by Paul Hoffrogge, Simon Daubner and Bei Zhou.

## License
MIT licence applies.
