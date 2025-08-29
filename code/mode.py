from collections import OrderedDict
from itertools import chain 
#import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0, speed_of_light
from scipy.integrate import solve_ivp
from shapely.geometry import Polygon 
from skfem import Basis, ElementTriP0, Mesh
from skfem.io import from_meshio
from femwell.maxwell.waveguide import compute_modes
from femwell.mesh import mesh_from_OrderedDict

parts = [("E", "real"), ("E", "imag"), ("H", "real"), ("H", "imag")]

w_sim, w_core_1, w_core_2 = 4, 0.45, 0.46
gap_fixed = 0.4 
h_clad, h_box, h_core = 1, 1, 0.22
offset_heater = 2.2

wavelength = 1.55
k0 = 2 * np.pi / wavelength


x_margin = 1.0
y_top = h_core + 0.6

polygons = OrderedDict(
    core_1=Polygon(
        [
        (-w_core_1 - gap_fixed / 2, 0),
        (-w_core_1 - gap_fixed / 2, h_core),
        (-gap_fixed / 2, h_core),
        (-gap_fixed / 2, 0)
        ]
    ),
    core_2=Polygon(
        [
        (w_core_2 + gap_fixed / 2, 0),
        (w_core_2 + gap_fixed / 2, h_core),
        (gap_fixed / 2, h_core),
        (gap_fixed / 2, 0),
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

resolution = dict(
    core_1={"resolution": 0.03, "distance": 1},
    core_2={"resolution": 0.03, "distance": 1},
)

mesh = from_meshio(
    mesh_from_OrderedDict(polygons, resolution, filename="mesh/mesh.msh", default_resolution_max=0.2)
)

basis0 = Basis(mesh, ElementTriP0(), intorder=4)

epsilon = basis0.zeros() + 1.444**2
epsilon[basis0.get_dofs(elements=("core_1","core_2"))] = 3.476**2
#basis0.plot(epsilon, colorbar=True).show()

# computing the modes
# showing symetric and antisymetric modes

modes_both = compute_modes(basis0, epsilon, wavelength=wavelength, mu_r=1, num_modes=2)

n_modes = len(modes_both)

fig, ax = plt.subplots(figsize=(8, 4))

colors = ["b", "r", "g", "m", "c"]  

for i, mode in enumerate(modes_both):
    field_real = mode.E.real
    field_imag = mode.E.imag
    ax.plot(field_real, label=f"Mode {i}", color=colors[i % len(colors)])
ax.set_xlabel("x-position")
ax.set_ylabel("E real")
ax.legend()
plt.savefig("../modes/x-position/modes_E_real.png", dpi=300, bbox_inches="tight")
plt.show()

n_modes = len(modes_both)
components = ["x", "y", "z"]

fig, ax = plt.subplots()
for i, mode in enumerate(modes_both):
    mode.plot_component("E", component="x", part="real", colorbar=True, ax=ax)
    plt.savefig(f"../modes/real/TE_mode_real_{i}.png", dpi=300, bbox_inches="tight") #symetric mode
    mode.plot_component("H", component="x", part="real", colorbar=True, ax=ax)
    plt.savefig(f"../modes/real/TM_mode_real_{i}.png", dpi=300, bbox_inches="tight") #antisymetric mode
#modes_both[0].plot_component("E", component="x", part="real", colorbar=True, ax=ax)
#fig.savefig("mode2_ex.png", dpi=300, bbox_inches="tight")

for j, mode in enumerate(modes_both):
    mode.plot_component("H", component="x", part="imag", colorbar=True, ax=ax)
    plt.savefig(f"../modes/imag/TM_mode_imag_{j}.png", dpi=300, bbox_inches="tight")


powers_in_waveguide_core_1 = []
powers_in_waveguide_core_2 = []

for mode in modes_both:

    powers_in_waveguide_core_1.append(mode.calculate_power(elements="core_1"))
    powers_in_waveguide_core_2.append(mode.calculate_confinement_factor(elements="core_2"))

print("Powers in waveguide core 1:", powers_in_waveguide_core_1)
print("Powers in waveguide core 2:", powers_in_waveguide_core_2)


# Now oxiding the core of mode 2 only
# cladding in 2 while 1 is still silicon meshs
#-----------------------------------------------------------------------------------------------------
epsilon = basis0.zeros() + 1.444**2
epsilon[basis0.get_dofs(elements="core_1")] = 3.4777**2

modes_1 = compute_modes(basis0, epsilon, wavelength=wavelength, mu_r=1, num_modes=1)
modes_1[0].show(modes_1[0].E.real, direction="x")

epsilon_2 = basis0.zeros() + 1.444**2
epsilon_2[basis0.get_dofs(elements="core_2")] = 3.4777**2

modes_2 = compute_modes(basis0, epsilon_2, wavelength=wavelength, mu_r=1, num_modes=1)
modes_2[0].show(modes_2[0].E.real, direction="x")

