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

def build_two_guides(gap_um=0.5, w_um=0.5, t_um=0.25,
                     w_sim=6.0, h_clad=2.0, h_box=2.0,
                     n_core=3.45, n_clad=1.00):
    g = gap_um
    w = w_um
    t = t_um

    polys = OrderedDict(
        coreL=Polygon([(-g/2 - w, 0), (-g/2 - w, t), (-g/2, t), (-g/2, 0)]),
        coreR=Polygon([( g/2, 0), ( g/2, t), ( g/2 + w, t), ( g/2 + w, 0)]),
        clad=Polygon([(-w_sim/2, 0), (-w_sim/2, h_clad), (w_sim/2, h_clad), (w_sim/2, 0)]),
        box=Polygon([(-w_sim/2, 0), (-w_sim/2, -h_box), (w_sim/2, -h_box), (w_sim/2, 0)]),
    )

    resolutions = dict(coreL={"resolution": 0.03, "distance": 1.0},
                       coreR={"resolution": 0.03, "distance": 1.0})

    mesh = from_meshio(
        mesh_from_OrderedDict(polys, resolutions, filename=None, default_resolution_max=0.2)
    )
    basis0 = Basis(mesh, ElementTriP0(), intorder=4)

    eps = basis0.zeros() + (n_clad**2)
    eps[basis0.get_dofs(elements=("coreL",))] = n_core**2
    eps[basis0.get_dofs(elements=("coreR",))] = n_core**2

    return mesh, basis0, eps

def kappa_from_supermodes(epsilon, basis0, wavelength_um=1.55, num_modes=2):

    k0 = 2*np.pi / wavelength_um
    modes = compute_modes(basis0, epsilon, wavelength=wavelength_um, num_modes=num_modes)

    def core_power_balance(mode):
        pL = mode.calculate_power(elements="coreL").real
        pR = mode.calculate_power(elements="coreR").real
        return -abs(pL - pR)  

    modes_sorted = sorted(modes, key=core_power_balance)[:2]
    neffs = [np.real(m.n_eff) for m in modes_sorted]
    idx_e = int(np.argmax(neffs))
    idx_o = 1 - idx_e

    beta_e = k0 * np.real(modes_sorted[idx_e].n_eff)
    beta_o = k0 * np.real(modes_sorted[idx_o].n_eff)

    kappa = 0.5 * (beta_e - beta_o)  # 1/µm
    return kappa, beta_e, beta_o

def beta_isolated(width_left_um=0.50, width_right_um=0.40, gap_um=1.0, t_um=0.25,
                  wavelength_um=1.55, n_core=3.45, n_clad=1.00):

    mesh, basis0, eps = build_two_guides(gap_um=gap_um, w_um=width_left_um, t_um=t_um,
                                         n_core=n_core, n_clad=n_clad)
    eps[basis0.get_dofs(elements=("coreR",))] = n_clad**2  # erase right
    mL = compute_modes(basis0, eps, wavelength=wavelength_um, num_modes=1)[0]
    beta1 = (2*np.pi/wavelength_um) * np.real(mL.n_eff)

    # Right core β2 (erase left)
    mesh, basis0, eps = build_two_guides(gap_um=gap_um, w_um=width_right_um, t_um=t_um,
                                         n_core=n_core, n_clad=n_clad)
    eps[basis0.get_dofs(elements=("coreL",))] = n_clad**2  # erase left
    mR = compute_modes(basis0, eps, wavelength=wavelength_um, num_modes=1)[0]
    
    beta2 = (2*np.pi/wavelength_um) * np.real(mR.n_eff)

    return beta1, beta2

# ---------- Convenience formulas for design lengths ----------
def lengths_from_CMT(kappa, delta_beta=0.0):
    """
    Return L_pi (full power transfer when Δβ=0), and L_50 for 50/50 split if feasible.
    All lengths in µm since kappa and betas are in 1/µm.
    """
    # Degenerate case:
    if abs(delta_beta) < 1e-9:
        L_pi = np.pi / (2*abs(kappa))
        L_50 = np.pi / (4*abs(kappa))
        return L_pi, L_50

    # Non-degenerate: 50/50 is possible only if |Δβ| ≤ 2|κ|
    if abs(delta_beta) > 2*abs(kappa):
        return np.inf, np.inf  # not achievable uniformly

    Omega = np.sqrt(abs(kappa)**2 + (delta_beta/2.0)**2)
    L_50 = (1.0/Omega) * np.arcsin(2*abs(kappa)/Omega)
    # L_pi (100%) doesn’t exist when Δβ≠0; report None for clarity
    return None, L_50