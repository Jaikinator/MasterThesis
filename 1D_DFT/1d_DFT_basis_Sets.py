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
def gauss_func(x,alpha, beta, pos0):
    """
    func to crate a gauss function
    :param x:  input grid array
    :param coeff: input an 2x1 array where coeff[0] is the prefactor of the exponential func
                  and coeff[0] the factor in the exponential func
    :return: gauss type func
    """
    return alpha * jnp.exp(- beta * (x - pos0)** 2)

@jit
def all_basis_func(x, alpha, beta, pos0):
    #print(len(jnp.where(num_orb != 0)))
    return vmap(gauss_func, (None, 0, 0, 0))(x, alpha, beta, pos0)



def system_args(n_grid, limit, **kwargs):
    grid_arr = jnp.linspace(-limit, limit, n_grid, dtype=jnp.float64)
    elec_alpha, elec_beta, elec_dist, elec_c = H2()

    arg_dict = {"x" : grid_arr,
                "alpha_arr": elec_alpha,
                "beta_arr": elec_beta,
                "dist_arr": elec_dist,
                "c" :  elec_c}
    return arg_dict


#@jit
def S_int(system_kwargs):
    """
    Integral to calc the overlap Matrix
    :param gauss_arg: input for system information see func gauss_args
    :return: Overlapmatrix
    """
    # all_basis_func() shape L, n_grid_points
    #Note give basis function array
    inner_overlap_mat = vmap(jnp.outer, (1,1))(all_basis_func(system_kwargs["x"],
                                                             system_kwargs["alpha_arr"],
                                                             system_kwargs["beta_arr"],
                                                             system_kwargs["dist_arr"][0]),
                                               all_basis_func(system_kwargs["x"],
                                                             system_kwargs["alpha_arr"],
                                                             system_kwargs["beta_arr"],
                                                             system_kwargs["dist_arr"][0]))

    # shape (n_grid_points, L,L)

    print(inner_overlap_mat.shape)
    overlap_mat = jnp.trapz(inner_overlap_mat, axis = 0)
    return overlap_mat

@jit
def kin_int(system_kwargs):
    """
    calc the kinetic energy Matrix of size (LxL)
    :param gauss_arg:
    :return: array [LxL]
    """
    laplace = jit(vmap(vmap(grad(grad(gauss_func)),(0, None, None ,None)),
                   (None, 0, 0, 0)))(system_kwargs["x"],
                                         system_kwargs["alpha_arr"],
                                         system_kwargs["beta_arr"],
                                         system_kwargs["dist_arr"])
    basis = all_basis_func(system_kwargs["x"],system_kwargs["alpha_arr"],
                          system_kwargs["beta_arr"],system_kwargs["dist_arr"])


    kin_op = vmap(jnp.outer, (1, 1))(basis, laplace)
    overlap_mat = jnp.trapz(kin_op, axis=0)
    return overlap_mat

@jit
def density_mat(system_kwargs):
    dens_mat = jnp.einsum('ni,mi -> nm', system_kwargs['c'], system_kwargs['c'])
    return dens_mat

def J_n_m(system_kwargs):
    P_m_n = density_mat(system_kwargs)
    def lam_delta_iterator(r1, dist_r1, r2, dist_r2, basis_mu_r1,basis_ny_r1, basis_lambda, basis_sigma):
        inner =  basis_mu_r1 * basis_ny_r1 * (1/((r2 - dist_r2)-(r1 - dist_r1))) * basis_lambda * basis_sigma
        return jnp.trapz(jnp.trapz(inner, system_kwargs["x"], axis= 0 ),system_kwargs["x"], axis=0)









if __name__ == "__main__":

    n_grid = 200
    limit = 5

    grid_arr = jnp.linspace(-limit, limit, n_grid, dtype=jnp.float64)

    elec1_alpha, elec1_beta, elec1_dist , elec1_c = H2()

    #print(all_basis_func(grid_arr, elec1_alpha, elec1_beta , elec1_dist[0]).shape)

    args = system_args(n_grid, limit)
    print(S_int(args))

