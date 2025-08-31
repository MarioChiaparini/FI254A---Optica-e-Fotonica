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

length = 2500
ts = np.linspace(0, length, 1500)

case = "non_identical"           # non_identical or identical
w = 0.50                     # width microm
t = 0.25                     # thickness miro m

w_sim = 6.0
gap_fixed = 0.4
h_clad, h_box, h_core = 1.0, 1.0, t

w_core_1 = w
w_core_2 = w if case == "identical" else 2*w

wavelength = 1.55
k0 = 2 * np.pi / wavelength

x_margin = 1.0
y_top = h_core + 0.6

polygons = OrderedDict(
    core_1=Polygon([
        (-w_core_1 - gap_fixed / 2, 0),
        (-w_core_1 - gap_fixed / 2, h_core),
        (-gap_fixed / 2, h_core),
        (-gap_fixed / 2, 0)
    ]),
    core_2=Polygon([
        (w_core_2 + gap_fixed / 2, 0),
        (w_core_2 + gap_fixed / 2, h_core),
        (gap_fixed / 2, h_core),
        (gap_fixed / 2, 0),
    ]),
    clad=Polygon([
        (-w_sim / 2, 0),
        (-w_sim / 2, h_clad),
        (w_sim / 2, h_clad),
        (w_sim / 2, 0),
    ]),
    box=Polygon([
        (-w_sim / 2, 0),
        (-w_sim / 2, -h_box),
        (w_sim / 2, -h_box),
        (w_sim / 2, 0),
    ]),
)

resolution = dict(
    core_1={"resolution": 0.03, "distance": 1},
    core_2={"resolution": 0.03, "distance": 1},
)

mesh = from_meshio(
    mesh_from_OrderedDict(polygons, resolution, filename="mesh/mesh.msh", default_resolution_max=0.2)
)
basis0 = Basis(mesh, ElementTriP0(), intorder=4)

# ---- air cladding (n=1.00), Si core (n=3.45) ----
n_clad, n_core = 1.00, 3.45
epsilon = basis0.zeros() + n_clad**2
epsilon[basis0.get_dofs(elements="core_1")] = n_core**2
modes_1 = compute_modes(basis0, epsilon, wavelength=wavelength, mu_r=1, num_modes=1)
modes_1[0].show(modes_1[0].E.real, direction="x")

epsilon_2 = basis0.zeros() + n_clad**2
epsilon_2[basis0.get_dofs(elements="core_2")] = n_core**2
modes_2 = compute_modes(basis0, epsilon_2, wavelength=wavelength, mu_r=1, num_modes=1)
modes_2[0].show(modes_2[0].E.real, direction="x")

epsilons = [epsilon, epsilon_2]

num_modes = len(modes_1) + len(modes_2)
overlap_integrals = np.zeros((num_modes, num_modes), dtype=complex)
for i, mode_i in enumerate(chain(modes_1, modes_2)):
    for j, mode_j in enumerate(chain(modes_1, modes_2)):
        overlap_integrals[i, j] = mode_i.calculate_overlap(mode_j)

coupling_coefficients = np.zeros((num_modes, num_modes), dtype=complex)
for i, mode_i in enumerate(chain(modes_1, modes_2)):
    for j, mode_j in enumerate(chain(modes_1, modes_2)):
        coupling_coefficients[i, j] = (
            k0
            * speed_of_light
            * epsilon_0
            * mode_i.calculate_coupling_coefficient(
                mode_j, epsilons[(j // len(modes_1) + 1) % 2] - n_clad**2 
            )
            * 0.5
        )

print("Overlap integrals")
print(overlap_integrals)
print("Coupling coefficients")
print(coupling_coefficients)

kappas = np.array(
    [[(coupling_coefficients[i, j] - overlap_integrals[i, (i + 1) % 2] * coupling_coefficients[(i + 1) % 2, j]
    / overlap_integrals[(i + 1) % 2, (i + 1) % 2])
    / (1- overlap_integrals[0, 1]* overlap_integrals[1, 0] / (overlap_integrals[0, 0] * overlap_integrals[1, 1]))
    for i in range(2)] for j in range(2)]
)

print(kappas)

delta = 0.5 * (
    np.real(modes_1[0].n_eff) * k0 + kappas[1, 1] - (np.real(modes_2[0].n_eff) * k0 + kappas[0, 0])
)
print(delta, np.real(modes_1[0].n_eff) * k0, kappas[1, 1])

beta_c = (kappas[0, 1] * kappas[1, 0] + delta**2) ** 0.5
print(np.pi / (2 * beta_c))

eta = np.abs(kappas[1, 0] ** 2 / beta_c**2) * np.sin(beta_c * 1e3)

plt.plot(ts, 1 - np.abs(kappas[1, 0] ** 2 / beta_c**2 * np.sin(beta_c * ts) ** 2))
plt.plot(ts, np.abs(kappas[1, 0] ** 2 / beta_c**2 * np.sin(beta_c * ts) ** 2))
plt.xlabel("Length (um)")
plt.ylabel("Power")
plt.title(f"waveguide ({case})")
plt.legend(["waveguide_1", "waveguide_2"])
plt.show()
