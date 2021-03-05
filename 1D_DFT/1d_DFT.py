from jax.config import config
config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.scipy as jsci
from jax import grad, jit, vmap
from jax.ops import index, index_add, index_update


@jit

def diffOP_first(x):
    n_grid_loc = len(x)
    h = x[1] - x[0]

    delta_ip1_j = jnp.diag(jnp.ones(n_grid_loc-1),1) #create the Kronecker i+1, j
        # jnp.diagflat Create a two-dimensional array with the flattened input as a diagonal.
        #    np.diagflat([1,2], 1)
        #    ==>    array([[0, 1, 0],
        #                  [0, 0, 2],
        #                  [0, 0, 0]])'

    delta_ij = jnp.eye(n_grid_loc) #crate the diagonal Elements

        # 1 in diagonal else 0 if k= 0 else k shift used diagonal
        #>>> np.eye(3, k=1)
        #       array([[0.,  1.,  0.],
        #              [0.,  0.,  1.],
        #              [0.,  0.,  0.]])

    D_mat = (delta_ip1_j -delta_ij)/ h  #calc Differential operator D_ij

    return D_mat

@jit
def diffOP_second(x):
    D = diffOP_first(x)
    D2 =(vmap(D.dot)(-D.T[None,:])).squeeze()
    D2_new = index_update(D2, index[-1,-1], D2[0,0])
    return D2_new
#non interacting electrons
@jit
def non_int(x):
   return jsci.linalg.eigh(-diffOP_second(x) / 2)


#harmonic oscilator
@jit
def harm_oscill(x):
    #input: x is 2D Array of shape (n_grid, n_grid)
    #output: x as 2D Array of shape (n_grid, n_grid)
    X = x * x
    return (vmap(jnp.diag, 0)(X[None,:])).squeeze()


#well potential:
@jit
def well_pot(x):
    w_old = jnp.full_like(x, 1.0e10)  # creat array of len of array x
    return index_update(w_old, index[jnp.logical_and(x > -2, x < 2)], 0.)

# integral
@jit
def integral(x,y):
    dx = x[1]- x[0]
    y_new = y[:, None]
    def mult(y):
       return jnp.sum(y * dx)
    return vmap(mult, (0))(y_new)

#Electron configuration array:



#density

def density(orb_array, psi, x):
    # norm the wave function:
    I = integral(x, psi ** 2)
    
    normed_psi = psi / jnp.sqrt(I)
    # follow the Hundschen rules to set up the 2 spins on the orbitals

    fnT =orb_array

    used_wavefunc = normed_psi.T[0:len(fnT), :]
    #def dens(orb, wavefunc):
        #return  orb * (wavefunc ** 2)
    dens = lambda orb, wavefunc : orb * (wavefunc ** 2)
    vdens = vmap(dens, (0,0), (0))(fnT, used_wavefunc)
    sum_ax = lambda x : jnp.sum(x,axis = 0)

    return vmap(sum_ax, (1), 0)(vdens)

#excange Potential
@jit
def get_exchange(nx,x):
    energy=-3./4.*(3./jnp.pi)**(1./3.)*integral(x,nx**(4./3.))
    potential=-(3./jnp.pi)**(1./3.)*nx**(1./3.)
    return energy, potential

#hartree fock
@jit
def get_hatree(nx, x, eps=1e-1):
    h = x[1] - x[0]

    energy = jnp.sum(nx[None, :] * nx[:, None] * h ** 2 / jnp.sqrt((x[None, :] - x[:, None]) ** 2 + eps) / 2)
    prepot = nx[None, :] * h / jnp.sqrt((x[None, :] - x[:, None]) ** 2 + eps)
    potential =lambda x :  jnp.sum(x, axis=-1)

    return energy, vmap(potential, 0, 0 )(prepot)

#prepare actual calc.

def e_conf(num_electrons, ngrid):
    """
    create an array with len of the maximal amount of covert orbitals
    :param num_electrons: describes of number of electrons used for the calculation
    :param ngrid: int, defines the grid size to calculate the maximal posibile number of orbitals
    :return: arraylike, return an 1D array, give the number of electrons in each possible orbital back
    """
    max_orbs = jnp.zeros(int(ngrid/2))
    num_orb = num_electrons // 2
    full_orbs = index_update(max_orbs, index[0:num_orb], 2)
    if jnp.mod(num_electrons, 2) == 1:
        fn = index_update(full_orbs, index[num_orb], 1)
    else:
        fn = full_orbs
    #fnT = fn.reshape((-1, 1))
    return fn

@jit
def hamilton(x,  ext_x):
    ham = lambda x : - diffOP_second(x) / 2
    return ham(x) + ext_x

# def create_calc(ngrid):
#         """
#           func that create the array with x Value for the Hamiltonian
#           :param ngrid: int, defines grid size
#           :param electron_num: describes of number of electrons used for the calculation
#           :param expt: arraylike input for the external potential
#         """
#         x = jnp.linspace(-5, 5, ngrid, dtype=jnp.float64)
#         shape_e_array = jnp.zeros(ngrid//2)
#         def calc(ngrid, orb_arr, pot, dens):
#             ex_energy, ex_potential = get_exchange(dens, x)
#             ha_energy, ha_potential = get_hatree(dens, x)
#             # Hamiltonian
#             pot_ext = ex_potential + ha_potential + pot
#             H = hamilton(x, ext_x= pot_ext)
#             energy, psi = jnp.linalg.eigh(H)
#             dens = density(orb_arr, psi, x)
#             return energy, psi, dens
#         return calc

#@jit
def calc(x, dens, orb_arr, pot):
        ex_energy, ex_potential = get_exchange(dens, x)
        ha_energy, ha_potential = get_hatree(dens, x)
        # Hamiltonian
        pot_ext = ex_potential + ha_potential + pot
        H = hamilton(x, ext_x= pot_ext)
        energy, psi = jnp.linalg.eigh(H)
        dens = 0.9 * dens + 0.1 * density(orb_arr, psi, x)
        return energy, psi, dens




if __name__ == "__main__":
    n_grid = 200
    limit = 5

    grid_arr =jnp.linspace(-limit, limit, n_grid, dtype=jnp.float64)


    num_electron_arr = jnp.array([17,12])
    orb_array = jnp.array([e_conf(num_electron_arr[i], n_grid) for i in range(len(num_electron_arr))])

    pot_arr = jnp.array((harm_oscill(grid_arr), harm_oscill(grid_arr)))
    number_of_samples = len(num_electron_arr)

    dens = jnp.zeros((number_of_samples, n_grid))  # initial dens
    max_iter = 1000
    energy_tolerance = 1e-5

    if number_of_samples > 1:
        log = {"energy": [np.inf for i in range(number_of_samples)] , "energy_diff":[np.inf for i in range(number_of_samples)]}
    else:
        log = {"energy": [float("inf")], "energy_diff": [float("inf")]}

    def print_log(i, log, num_sampl):
        if number_of_samples > 1:
            print(f"step: {i} energy: {[round(float(log['energy'][-1][i]), 3)for i in range(num_sampl) ]}"
                  f" energy_diff: {[round(float(log['energy_diff'][-1][i]), 5) for i in range(num_sampl)]}")
        else:
            print(f"step: {i} energy: {round(log['energy'][-1],3)} energy_diff: {round(log['energy_diff'][-1],5)}")

    #print(grid_arr.shape, dens.shape, num_electron_arr.shape, orb_array.shape)
    for i in range(max_iter):
        if number_of_samples > 1:

            energy, psi, dens = vmap(calc, (None, 0, 0, 0))(grid_arr, dens, orb_array, pot_arr)
            # print(f"energy: {energy.shape},\t psi: {psi.shape},\t dens: {dens.shape}")

            #log
            log["energy"].append([energy[i,0] for i in range(number_of_samples)])

            energy_diff = [energy[i][0] - log["energy"][-2][i] for i in range(number_of_samples)]
            print("Hello Darkness my old friend")
            log["energy_diff"].append(energy_diff)
            print_log(i, log, number_of_samples)

            # convergence
            if np.abs(max(energy_diff)) < energy_tolerance:
                print("converged!")
                break
            else:
                print("not converged")
        else:
            if i == 0:
                dens  = dens.squeeze()
                orb_array = orb_array.squeeze()
                pot_arr = pot_arr.squeeze()
            energy, psi, dens = calc(grid_arr, dens, orb_array, pot_arr)

            log["energy"].append(energy[0])
            energy_diff = energy[0] - log["energy"][-2]
            log["energy_diff"].append(energy_diff)
            print_log(i, log, number_of_samples)

            # convergence
            if np.abs(energy_diff) < energy_tolerance:
                print("converged!")
                break
            else:
                print("not converged")

