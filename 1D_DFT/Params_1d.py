import matplotlib.pyplot as plt
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from jax import partial
from jax.ops import index, index_add, index_update

from scipy.optimize import curve_fit

from dft_1d import *

def create_c_arr(alpha, dist): #u can use beta as well
    return jnp.zeros((len(alpha), len(dist)))

@jit
def H1():
    """
    H1 described by only one basis func
    :return: params for H1
    """
    alpha = jnp.array(1)
    beta = jnp.array(1)
    kohn_sham_orb = jnp.array(1)
    c = create_c_arr(alpha)
    return alpha, beta, kohn_sham_orb, c

@jit
def He():
    """
    He get 2 electrons similar every electron gets a describtion by only one basis func
    as usual the electrons are in the 1s orbitals there has to be a node in the desc of the orbital
    :return: prarams for He
    """
    alpha = jnp.array([[1 ,-1]])
    beta = jnp.array([[1, -1]])
    kohn_sham_orb = jnp.array(1)
    c = create_c_arr(alpha)
    return alpha, beta, kohn_sham_orb, c

@jit
def Li():
    """
     Li gets 3 electrons similar every electron gets a describtion by only one basis func
     as usual the electrons are in the 1s orbital as well as the 2s orbital
     :return: prarams for Li
     """
    alpha = jnp.array([[1, -1], [1, 0]])
    beta = jnp.array([[1, -1], [1, 0]])
    kohn_sham_orb = jnp.array([1,2])
    c = create_c_arr(alpha)
    return alpha, beta, kohn_sham_orb, c


def H2(**kwargs):
    if "alpha" in kwargs:
        alpha = jnp.array(kwargs["alpha"])
    else:
        alpha = jnp.array([1.0, -1.0])

    if "beta" in kwargs:
        beta = jnp.array(kwargs["beta"])
    else:
        beta = jnp.array([1.0, -1.0])

    if "dist" in kwargs:
        dist = jnp.array(kwargs["dist"])
    else:
        dist = jnp.array([0.0, 1.0])
    c = create_c_arr(alpha, dist)
    return alpha, beta, dist, c




def test_LCAO_basis_func(grid_arr, num_electrons):
    """
    to test the DFT Kohn sham approach in 1d_DFT_basis_Sets for the internal coe not relevant
    :param x: grid_arr
    :param dens: local density
    :param orb_arr: orbital array
    :return: wave function
    """
    orb_array = e_conf(num_electrons, len(grid_arr))
    dens = jnp.zeros(len(grid_arr))
    max_iter = 1000
    energy_tolerance = 1e-5

    #add external field

    #pot_ext = well_pot(grid_arr)

    #logging energy diff:

    log = {"energy": [float("inf")], "energy_diff": [float("inf")]}

    def print_log(i, log):
        print(f"step: {i} energy: {round(log['energy'][-1], 3)} energy_diff: {round(log['energy_diff'][-1], 5)}")

    #create fit func
    def gauss_func(x, alpha, beta, pos0):
        """
        func to crate a gauss function
        :param x:  input grid array
        :param coeff: input an 2x1 array where coeff[0] is the prefactor of the exponential func
                      and coeff[0] the factor in the exponential func
        :return: gauss type func
        """
        return alpha * jnp.exp(- beta * (x - pos0) ** 2)

    #solve System in realspace

    for i in range(max_iter):

        if i == 0:
            dens  = dens.squeeze()
            orb_array = orb_array.squeeze()

        energy, psi, dens = calc_raw(grid_arr, dens, orb_array)
        #force = grad(calc_Energy, 3)(grid_arr, dens, orb_array, pot_arr)
        #print(force.shape,force[0,:,:])
        log["energy"].append(energy[0])
        energy_diff = energy[0] - log["energy"][-2]
        log["energy_diff"].append(energy_diff)
        print_log(i, log)

        # convergence
        if np.abs(energy_diff) < energy_tolerance:
            print("converged!")
            break
        else:
            print("not converged")
    #create output arrays
    alpha = np.zeros(num_electrons)
    beta = np.zeros(num_electrons)
    pos = np.zeros(num_electrons)
    psi_new = np.zeros(psi.shape)

    #calculate fit func for the relevant Praticles
    for i in range(num_electrons):

        if i == 0:
            fit_arr = curve_fit(gauss_func, grid_arr, psi[:,0], p0=[1, 1, -1], maxfev=1000000)[0]
        else:
            fit_arr = curve_fit(gauss_func, grid_arr, psi[:,1], maxfev=1000000)[0]


        alpha[i] = fit_arr[0]
        beta[i] = fit_arr[1]
        pos[i] = fit_arr[2]


    # for i in range(3):
    #
    #     plt.plot(grid_arr, psi[:,i],label=f"psi [:,{i}] orginal" )
    #     plt.title("plot psi[:,i](column wise)")
    #     plt.legend()
    # plt.savefig(f"/home/jacob/PycharmProjects/MasterThesis/1D_DFT/Plots/psi_column.png")
    # plt.close()

    # #prove of the fit func for the realspace system
    # for i in range(len(grid_arr)):
    #     if i == 0:
    #         fit_arr = curve_fit(gauss_func, grid_arr, psi[0], p0=[1, 1, 1])[0]
    #     else:
    #         fit_arr = curve_fit(gauss_func, grid_arr, psi[i], maxfev=1000000)[0]
    #     psi_new[i,:] =  gauss_func(grid_arr, fit_arr[0], fit_arr[1] ,fit_arr[2])
    #     print(f"fit of psi of {i} is done!")
    #
    #
    # dens = density(orb_array, psi_new, grid_arr)
    # energy, psi, dens = calc_raw(grid_arr, dens, orb_array)
    # # force = grad(calc_Energy, 3)(grid_arr, dens, orb_array, pot_arr)
    # # print(force.shape,force[0,:,:])
    # log["energy"].append(energy[0])
    # energy_diff = energy[0] - log["energy"][-2]
    # log["energy_diff"].append(energy_diff)
    # print_log(i, log)
    #

    # force = grad(calc_Energy, 3)(grid_arr, dens, orb_array, pot_arr)
    # print(force.shape,force[0,:,:])

    # return jnp.array(alpha), jnp.array(beta), jnp.array(pos), create_c_arr(alpha, pos)

    return psi[:, (0 , 1)]


