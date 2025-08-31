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



def kappa_ij(mode_i, mode_j, delta_eps, P_i, wavelength_um):
    omega = 2 * np.pi * speed_of_light / (wavelength_um * 1e-6)
    overlap = mode_i.calculate_overlap(mode_j, delta_eps)
    return (omega * epsilon_0 / (4 * P_i)) * overlap

length = 2500
ts = np.linspace(0, length, 1500)
                   
n_clad = 1.00  # Air
n_core = 3.45  # Silicon

t = 0.25                    

w_sim = 6.0
gap_fixed = 0.4
h_clad, h_box, h_core = 1.0, 1.0, t

w_core_1, w_core_2 = 0.50, 0.50

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
resolutions = dict(
    core_1={"resolution": 0.005, "distance": 10e-6},
    core_2={"resolution": 0.005, "distance": 10e-6},
)

mesh = from_meshio(
    mesh_from_OrderedDict(polygons, resolutions, default_resolution_max=0.25)
)
#mesh.draw().show()
basis0 = Basis(mesh, ElementTriP0(), intorder=4)

epsilon_clad = n_clad**2
epsilon_core = n_core**2

epsilon_r = basis0.zeros() + epsilon_clad


epsilon_r[basis0.get_dofs(elements=("core_1", "core_2"))] = epsilon_core

#epsilon_r[basis0.get_dofs(elements=("core_1"))] = epsilon_core


modes_both = compute_modes(
    basis0,
    epsilon_r,
    wavelength=wavelength,
    mu_r=1,
    num_modes=4,
    n_guess=2.4

)

#modes_both[0].show(modes_both[0].E.real, direction="x")
#modes_both[1].show(modes_both[1].E.real, direction="x")

#R = [np.abs(modes_both[0].calc_overlap(i) ** 22) for i in  modes_both]

#for mode in modes_both:
#    mode.show("E", part="real")

#print(f"neff: {np.real(modes_both[0].n_eff)}")
#print(f"neff: {np.real(modes_both[1].n_eff)}")
#print(f"neff: {np.real(modes_both[2].n_eff)}")
#print(f"neff: {np.real(modes_both[3].n_eff)}")

length = 1000
ts = np.linspace(0, length, 1000)

epsilon1 = basis0.zeros() + epsilon_clad
epsilon1[basis0.get_dofs(elements=("core_1",))] = epsilon_core

modes_1 = compute_modes(
    basis0,
    epsilon1,
    wavelength=wavelength,
    mu_r=1,
    num_modes=4,
    n_guess=2.4
    #n_guess=1.16
)
for mode in modes_1:
    mode.show("E", part="real")
for neff in [mode.n_eff for mode in modes_1]:
    print(f"neff: {np.real(neff)}")

epsilon2 = basis0.zeros() + epsilon_clad
epsilon2[basis0.get_dofs(elements=("core_2",))] = epsilon_core

modes_2 = compute_modes(
    basis0,
    epsilon2,
    wavelength=wavelength,
    mu_r=1,
    num_modes=2,
    n_guess=2.43
    #n_guess=1.16
)
for mode in modes_2:
    mode.show("E", part="real")
#epsilons = [epsilon1, epsilon2]
for neff in [mode.n_eff for mode in modes_1]:
    print(f"neff: {np.real(neff)}")

delta_epsilon_1 = epsilon2 - epsilon1
delta_epsilon_2 = epsilon1 - epsilon2

beta1 = k0 * np.real(modes_1[0].n_eff)
beta2 = k0 * np.real(modes_2[0].n_eff)

#num_modes = len(modes_1) + len(modes_2) 
#overlap_integral = np.zeros((num_modes, num_modes), dtype=complex)
#p1, p2 = np.zeros((num_modes, num_modes), dtype=complex), np.zeros((num_modes, num_modes), dtype=complex)

#k11, k12, k21, k22 = np.zeros((num_modes, num_modes), dtype=complex), np.zeros((num_modes, num_modes), dtype=complex), np.zeros((num_modes, num_modes), dtype=complex), np.zeros((num_modes, num_modes), dtype=complex)
#delta_beta_prime = np.zeros((num_modes, num_modes), dtype=complex)
#omega = np.zeros((num_modes, num_modes), dtype=complex)
#power_transfer = np.zeros((num_modes, num_modes, len(ts)), dtype=complex)
#for i, mode_i in enumerate(chain(modes_1, modes_2)):
#    for j, mode_j in enumerate(chain(modes_1, modes_2)):
#        overlap_integral[i, j] = mode_i.calculate_overlap(mode_j)
#        p1[i,j] = 0.25 * np.real(mode_i.calculate_overlap(mode_i))
#        p2[i, j] = 0.25 * np.real(mode_j.calculate_overlap(mode_j))

#        k11[i,j] = kappa_ij(mode_i, mode_i, delta_epsilon_1 , p1[i,j], wavelength)
#        k12[i,j] = kappa_ij(mode_i, mode_j, delta_epsilon_1 , p1[i,j], wavelength)
#        k21[i,j] = kappa_ij(mode_j, mode_i, delta_epsilon_2, p2[i,j], wavelength)
#        k22[i,j] = kappa_ij(mode_j, mode_j, delta_epsilon_2, p2[i,j], wavelength)

#        delta_beta_prime[i,j] = (beta2 + np.real(k22[i,j])) - (beta1 + np.real(k11[i,j]))

#        omega[i,j] = np.sqrt((delta_beta_prime[i,j] / 2)**2 + np.real(k12[i,j]) * np.real(k21[i,j]))
#        Lc = np.pi / np.abs(omega[i,j])
#        print(f"Coupling length Lc between mode {i} and mode {j}: {Lc*1e-6} um")
#        z = np.linspace(0, 2*Lc, 1000)
#        a1 = np.cos(np.abs(omega[i,j]) * z)
#        a2 = np.sin(np.abs(omega[i,j]) * z)
#        power_transfer[i,j] = (np.abs(k21[i,j])**2 / np.abs(omega[i,j])**2) * np.sin(omega[i,j] * z)**2
#        plt.figure()
#        plt.plot(z, 1-power_transfer[i,j], label=f"Power transfer from mode {i} to mode {j}")
#        plt.xlabel("(z) [um]")
#        plt.ylabel("Power")
#        plt.legend()
#        plt.show()


#coupling_coefficients = np.zeros((num_modes, num_modes), dtype=complex)

#for i, mode_i in enumerate(chain(modes_1, modes_2)):
#    for j, mode_j in enumerate(chain(modes_1, modes_2)):
#        if i != j:
#            delta_epsilon = epsilons[j // len(modes_1)] - epsilons[i // len(modes_1)]
#            integrand = (
#                np.conj(mode_i.E) @ delta_epsilon @ mode_j.E
#                - np.conj(mode_i.H) @ (1 - 1) @ mode_j.H
#            )
#            coupling_coefficients[i, j] = (
#                1 / 4
#            ) * (k0**2) * basis0.integrate(integrand)


#kappas = np.array(
#    [[
#        (
#            coupling_coefficients[i, j] / overlap_integral[i, ]
#        )
#    ]]
#)



#k11 = kappa_ij(modes_1[0], modes_1[0], delta_epsilon_1, P1, wavelength)
#k12 = kappa_ij(modes_1[0], modes_2[0], delta_epsilon_1, P1, wavelength)
#k21 = kappa_ij(modes_2[0], modes_1[0], delta_epsilon_2, P2, wavelength)
#k22 = kappa_ij(modes_2[0], modes_2[0], delta_epsilon_2, P2, wavelength) 
#print(f"k11: {k11}, k22: {k22}, k12: {k12}, k21: {k21}")

#beta1 = k0 * np.real(modes_1[0].n_eff)
#beta2 = k0 * np.real(modes_2[0].n_eff)
#delta_beta = (beta2 + np.real(k22)) - (beta1 + np.real(k11))

#omega = np.sqrt((delta_beta / 2)**2 + np.real(k12) * np.real(k21))
#print(f"beta1: {beta1}, beta2: {beta2}")
#print(f"delta_beta: {delta_beta}")

#omega_abs = np.abs(omega)
#print(f"omega: {omega}, |omega|: {omega_abs}")


#L = np.pi / omega_abs
#print(f"Coupling length L: {L*1e-6} um")

#plt.figure()
#plt.plot(ts, np.cos(omega_abs * ts)**2, label="|a1|^2")
#plt.plot(ts, np.sin(omega_abs * ts)**2, label="|a2|^2")
#plt.xlabel("Propagation distance (z) [um]")
#plt.ylabel("Power in each waveguide")
#plt.title("Power exchange between two coupled waveguides")
#plt.axvline(L, color="k", linestyle="--", label="Coupling length L")
#plt.legend()

for i, mode_i in enumerate(modes_1):
    for j, mode_j in enumerate(modes_2):

        m1, m2 = mode_i, mode_j
        beta1 = k0 * np.real(m1.n_eff)
        beta2 = k0 * np.real(m2.n_eff)

        P1 = 0.25 * np.real(m1.calculate_overlap(m1))
        P2 = 0.25 * np.real(m2.calculate_overlap(m2))

        κ11 = kappa_ij(m1, m1, delta_epsilon_1, P1, wavelength)
        κ22 = kappa_ij(m2, m2, delta_epsilon_2, P2, wavelength)
        κ12 = kappa_ij(m1, m2, delta_epsilon_1, P1, wavelength)
        κ21 = kappa_ij(m2, m1, delta_epsilon_2, P2, wavelength)
        Δβp = (beta2 + np.real(κ22)) - (beta1 + np.real(κ11))
        g = np.sqrt(Δβp**2 + 4*np.abs(κ12*κ21)) #mismatch phase
        #g   = np.sqrt((Δβp/2.0)**2 + κ12*κ21) #match case
        L = np.pi/np.abs(g) 
        print(f"Coupling length L between mode {i} and mode {j}: {L*1e-6} um")
        z = np.linspace(0, 2.0*L, 1000)
        zL = z/L
        P2 = (np.abs(κ21)**2 / np.abs(g)**2) * np.sin(np.abs(g)*z)**2
        P1 = 1 - P2
        plt.figure()
        plt.plot(zL, P1, 'b', label=f"W1 mode {i}")
        plt.plot(zL, P2, 'r', label=f"W2 mode {j}")
        plt.xlabel(r"$z/L$")
        plt.ylabel("Power")
        plt.legend()
        plt.show()