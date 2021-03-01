from jax.config import config
config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.scipy as jsci
from jax import grad, jit, vmap
from jax.ops import index, index_add, index_update


x = jnp.linspace(-5, 5, 200, dtype=jnp.float64)

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
    return vmap(jnp.diag, 0)(X[None,:])


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
@jit
def density(orb_array, psi, x):
    # norm the wave function:
    I = integral(x, psi ** 2)
    
    normed_psi = psi / jnp.sqrt(I)
    # follow the Hundschen rules to set up the 2 spins on the orbitals

    fnT =orb_array
    used_wavefunc = normed_psi.T[0:len(fnT[:,0]), :]
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

if __name__ == "__main__":
    n_grid = 200
    x = jnp.linspace(-5, 5, n_grid, dtype=jnp.float64)
    dx = x[1] - x[0]
    num_electrons = 17
    max_iter = 1000
    energy_tolerance = 1e-5
    log = {"energy": [float("inf")], "energy_diff": [float("inf")]}


    def e_conf(num_electrons):
        new_orb = False
        num_orb = num_electrons // 2

        if jnp.mod(num_electrons, 2) == 1:
            new_orb = True
        orbital_number = num_orb + new_orb

        fn_0 = jnp.full(orbital_number, 2)
        if new_orb == True:
            fn = index_update(fn_0, index[-1], 1)

        fnT = fn.reshape((-1, 1))
        return fnT

    orb_array = e_conf(num_electrons)

    def print_log(i, log):
        print(f"step: {i} energy: {round(log['energy'][-1], 5)} energy_diff: {round(log['energy_diff'][-1], 5)}")

    def Energy_min(dens, x):
        ex_energy, ex_potential = get_exchange(dens, x)
        ha_energy, ha_potential = get_hatree(dens, x)

        # Hamiltonian
        pot = ex_potential + ha_potential + x * x
        H = - diffOP_second(x) / 2 + (vmap(jnp.diag, 0)(pot[None, :])).squeeze()
        #test = ex_potential + ha_potential + x * x
        #print(jnp.diagflat(ex_potential + ha_potential + x * x).shape,(vmap(jnp.diag, 0)(test[None, :])).squeeze().shape)
        energy, psi = jnp.linalg.eigh(H)
        return energy, psi

    dens = jnp.zeros(n_grid)

    for i in range(max_iter):

        energy, psi = Energy_min(dens, x)

        # log
        log["energy"].append(energy[0])
        energy_diff = energy[0] - log["energy"][-2]
        log["energy_diff"].append(energy_diff)
        print_log(i, log)
        # convergence

        if np.abs(energy_diff) < energy_tolerance:
            print("converged!")
            break

        # update density
        dens = density(orb_array, psi, x)
    else:
        print("not converged")