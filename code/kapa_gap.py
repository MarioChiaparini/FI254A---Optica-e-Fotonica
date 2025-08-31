# --- kappa_vs_gap_degnerate.py ----------------------------------------------
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from skfem import Basis, ElementTriP0
from skfem.io import from_meshio
from femwell.mesh import mesh_from_OrderedDict
from femwell.maxwell.waveguide import compute_modes

wavelength = 1.55         
n_core, n_clad = 3.45, 1.00

w = 0.50                 
t = 0.25                  

gaps = np.linspace(0.15, 1.50, 100) 

w_sim = 8.0              
h_top, h_box = 2.0, 2.0   
res_core = 0.03            
res_max  = 0.25          

k0 = 2*np.pi/wavelength   

def build_mesh(gap_um: float):

    core_L = Polygon([(-gap_um/2 - w, 0), (-gap_um/2 - w, t),
                      (-gap_um/2, t), (-gap_um/2, 0)])

    core_R = Polygon([( gap_um/2, 0), ( gap_um/2, t),
                      ( gap_um/2 + w, t), ( gap_um/2 + w, 0)])

    clad = Polygon([(-w_sim/2, 0), (-w_sim/2, h_top),
                    ( w_sim/2, h_top), ( w_sim/2, 0)])
    box  = Polygon([(-w_sim/2, 0), (-w_sim/2, -h_box),
                    ( w_sim/2, -h_box), ( w_sim/2, 0)])

    polygons = OrderedDict(
        core_L=core_L,
        core_R=core_R,
        clad=clad,
        box=box,
    )
    resolutions = dict(
        core_L={"resolution": res_core, "distance": 1.0},
        core_R={"resolution": res_core, "distance": 1.0},
    )
    mesh = from_meshio(
        mesh_from_OrderedDict(polygons, resolutions, default_resolution_max=res_max)
    )
    basis0 = Basis(mesh, ElementTriP0(), intorder=4)
    return mesh, basis0

def assign_eps(basis0):

    eps = basis0.zeros() + n_clad**2
    eps[basis0.get_dofs(elements=("core_L","core_R"))] = n_core**2
    return eps

def pick_coupler_supermodes(modes):

    ranked = sorted(
        modes,
        key=lambda m: -(
            m.calculate_power(elements="core_L").real +
            m.calculate_power(elements="core_R").real
        )
    )
    return ranked[0], ranked[1]

kappas = []
Lpi = []
for g in gaps:
    mesh, basis0 = build_mesh(g)
    eps = assign_eps(basis0)

    modes = compute_modes(basis0, eps, wavelength=wavelength, num_modes=6, order=2)

    m_even, m_odd = pick_coupler_supermodes(modes)

    beta_e = np.real(m_even.n_eff) * k0
    beta_o = np.real(m_odd.n_eff) * k0

    kappa = 0.5 * np.abs(beta_e - beta_o)    
    kappas.append(kappa)
    Lpi.append(np.pi / (2*kappa))             

kappas = np.array(kappas)
Lpi = np.array(Lpi)

fig, ax1 = plt.subplots(figsize=(6,4.2))
ax1.plot(gaps*1e3, kappas*1e3, label=r"$\kappa$ (1/mm)")
ax1.set_xlabel("Gap (nm)")
ax1.set_ylabel(r"$\kappa$  [1/mm]")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(gaps*1e3, Lpi, color="tab:orange", label=r"$L_\pi$")
ax2.set_ylabel(r"$L_\pi$  [µm]")

lines, labels = [], []
for ax in (ax1, ax2):
    L, lab = ax.get_legend_handles_labels()
    lines += L; labels += lab
ax1.legend(lines, labels, loc="best")

plt.title("Degenerate coupling: κ and Lπ vs gap")
plt.tight_layout()
plt.show()