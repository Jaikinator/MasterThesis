from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.scipy as jsci
from jax import grad, jit, vmap
from jax import partial
from jax.ops import index, index_add, index_update


from typing import Dict
from jax_md.util import Array
from jax_md.util import f32
from jax_md.util import f64
from jax_md.util import safe_mask
from jax_md.smap import _kwargs_to_parameters
from jax_md.space import distance, square_distance



import dft_1d as dft
from Params_1d import *


def _kwargs_to_parameters(species: Array = None, **kwargs) -> Dict[str, Array]:
    """Extract parameters from keyword arguments."""
    # NOTE(schsam): We could pull out the species case from the generic case.
    s_kwargs = kwargs
    for key, val in kwargs.items():
        s_kwargs[key] = val[species]
    return s_kwargs


@jit
def gauss_func(x,alpha, beta, dist):
    """
    func to crate a gauss function
    :param x:  input grid array
    :param coeff: input an 2x1 array where coeff[0] is the prefactor of the exponential func
                  and coeff[0] the factor in the exponential func
    :return: gauss type func
    """
    return alpha * jnp.exp(- beta * (x - dist)** 2)

@jit
def kohn_sham_orb(x, alpha, beta, dist):
    #print(len(jnp.where(num_orb != 0)))
    return vmap(gauss_func, (None, 0, 0, None))(x, alpha, beta, dist)

# def atomic_kohn_sham(x, alpha, beta, orb , dist):
#     return vmap(kohn_sham_orb, (None, None, None, 0, None))(x, alpha, beta, orb, dist)


def system_args(n_grid, limit,dist_arr, **kwargs):
    grid_arr = jnp.linspace(-limit, limit, n_grid, dtype=jnp.float64)

    for i in range(len(dist_arr)):
        elec_alpha, elec_beta, elec_dist, elec_c = free_elec(dist = dist_arr[i])

        if i == 0:
            elec_alpha_arr = jnp.zeros((len(elec_alpha), len(dist_arr)))
            elec_beta_arr = jnp.zeros((len(elec_beta), len(dist_arr)))

        elec_alpha_arr = index_update(elec_alpha_arr,index[i, :],elec_alpha)
        elec_beta_arr = index_update(elec_beta_arr,index[i, :],elec_beta)

    arg_dict = {"x" : grid_arr,
                "alpha_arr": elec_alpha_arr,
                "beta_arr": elec_beta_arr,
                "dist_arr": dist_arr,
                "c" : jnp.zeros((len(elec1_c),len(elec1_c)))}
    return arg_dict


@jit
def S_int(system_kwargs):
    """
    Integral to calc the overlap Matrix
    :param gauss_arg: input for system information see func gauss_args
    :return: Overlapmatrix
    """
    inner_overlap_mat = vmap(jnp.outer, (1,1))(kohn_sham_orb(system_kwargs["x"],
                                                             system_kwargs["alpha_arr"][0, :],
                                                             system_kwargs["beta_arr"][0, :],
                                                             system_kwargs["dist_arr"][0]),
                                               kohn_sham_orb(system_kwargs["x"],
                                                             system_kwargs["alpha_arr"][0, :],
                                                             system_kwargs["beta_arr"][0, :],
                                                             system_kwargs["dist_arr"][0]))
    overlap_mat = jnp.trapz(inner_overlap_mat, axis = 0)
    return overlap_mat

#@jit
def kin_int(system_kwargs):
    """
    calc the kinetic energy Matrix of size (LxL)
    :param gauss_arg:
    :return: array [LxL]
    """
    laplace = jit(vmap(vmap(grad(grad(gauss_func)),(0, None, None ,None)),
                   (None, 0, 0 , None)))(system_kwargs["x"],
                                         system_kwargs["alpha_arr"][0, :],
                                         system_kwargs["beta_arr"][0, :],
                                         system_kwargs["dist_arr"][0])
    basis = kohn_sham_orb(system_kwargs["x"],system_kwargs["alpha_arr"][0, :],
                          system_kwargs["beta_arr"][0, :],system_kwargs["dist_arr"][0])


    kin_op = vmap(jnp.outer, (1, 1))(basis, laplace)
    overlap_mat = jnp.trapz(kin_op, axis=0)
    #(system_kwargs["x"],system_kwargs["alpha_arr"][0, :],system_kwargs["beta_arr"][0, :],system_kwargs["dist_arr"][0])
    # kin_op = lambda x,func_coeff, basis: basis * (-(vmap(grad(grad(kohn_sham_orb)),(0,None))(x,func_coeff))/2)
    # kin_part = jit(vmap(vmap(kin_op, (None,0, None)), (None, None, 0 )))(
    #     system_kwargs["x"], system_kwargs["bfunc_coeff"] , system_kwargs["basis_func"]) #verallgemeinern
    # return jnp.trapz(kin_part, axis=(2))
    return overlap_mat


def density_mat(gauss_arg):
    dens_mat = jnp.einsum('ni,mi -> nm', gauss_arg['c'], gauss_arg['c'])
    return dens_mat


if __name__ == "__main__":

    n_grid = 200
    limit = 5
    grid_arr = jnp.linspace(-limit, limit, n_grid, dtype=jnp.float64)
    elec1_alpha, elec1_beta, elec1_dist , elec1_c = free_elec()
    elec2_alpha, elec2_beta, elec2_dist, elec2_c = free_elec(dist = 1)
    print(elec1_alpha, elec1_beta, elec1_dist , elec1_c )
    print(kohn_sham_orb(grid_arr, elec1_alpha, elec1_beta , elec1_dist).shape)
    #print(kohn_sham_orb(grid_arr, elec2_alpha, elec2_beta, elec2_dist).shape)
    dist_arr = jnp.array([0,1])
    args = system_args(n_grid, limit, dist_arr)
    print(kin_int(args))