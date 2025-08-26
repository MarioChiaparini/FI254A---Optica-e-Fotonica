from collections import OrderedDict
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0, speed_of_light
from scipy.integrate import solve_ivp
from shapely.geometry import Polygon
from skfem import Basis, ElementTriP0, Mesh
from skfem.io import from_meshio

from femwell.maxwell.waveguide import compute_modes
from femwell.mesh import mesh_from_OrderedDict


w_sim = 4
h_clad = 1
h_box = 1
w_core_1 = 0.45
w_core_2 = 0.46
gap = 0.4
h_core = 0.22
offset_heater = 2.2
h_heater = 0.14
w_heater = 2

wavelength = 1.55
k0 = 2 * np.pi / wavelength

polygons = OrderedDict(
    core_1=Polygon(
        [
            (-w_core_1 - gap / 2, 0),
            (-w_core_1 - gap / 2, h_core),
            (-gap / 2, h_core),
            (-gap / 2, 0),
        ]
    ),
    core_2=Polygon(
        [
            (w_core_2 + gap / 2, 0),
            (w_core_2 + gap / 2, h_core),
            (gap / 2, h_core),
            (gap / 2, 0),
        ]
    ),
    clad=Polygon(
        [
            (-w_sim / 2, 0),
            (-w_sim / 2, h_clad),
            (w_sim / 2, h_clad),
            (w_sim / 2, 0),
        ]
    ),
    box=Polygon(
        [
            (-w_sim / 2, 0),
            (-w_sim / 2, -h_box),
            (w_sim / 2, -h_box),
            (w_sim / 2, 0),
        ]
    ),
)

resolutions = dict(
    core_1={"resolution": 0.03, "distance": 1},
    core_2={"resolution": 0.03, "distance": 1},
)

mesh = from_meshio(
    mesh_from_OrderedDict(polygons, resolutions, filename="mesh.msh", default_resolution_max=0.2)
)
mesh.draw().show()