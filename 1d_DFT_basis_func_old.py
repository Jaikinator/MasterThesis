from jax.config import config

config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.scipy as jsci
from jax import grad, jit, vmap
from jax import partial
from jax.ops import index, index_add, index_update

import dft_1d as dft


@jit
def gauss_func(x, coeff, dist):
    """
    func to crate a gauss function
    :param x:  input grid array
    :param coeff: input an 2x1 array where coeff[0] is the prefactor of the exponential func
                  and coeff[0] the factor in the exponential func
    :return: gauss type func
    """
    return coeff[0] * jnp.exp(- coeff[1] * (x - dist) ** 2)


def atomic_kohn_sham(x, coeff, num_orb, dist):
    return vmap(gauss_func * num_orb, (None, 0, None))(x, coeff, dist)


def system_args(n_grid, limit, electron_number, bfunc_coeff, c=None):
    grid_arr = jnp.linspace(-limit, limit, n_grid, dtype=jnp.float64)
    basis_func = vmap(gauss_func, (None, 0))(grid_arr, bfunc_coeff)
    num_orb = dft.e_conf(electron_number, n_grid)
    print(num_orb.shape)
    if c == None:
        c_out = np.zeros((len(basis_func), len(num_orb)))
    arg_dict = {"x": grid_arr,
                "bfunc_coeff": bfunc_coeff,
                "num_orb": num_orb,
                "basis_func": basis_func,  # [b1,b2,...,bn]
                "c": c_out}
    return arg_dict


@jit
def S_int(system_kwargs):
    """
    Integral to calc the overlap Matrix
    :param gauss_arg: input for system information see func gauss_args
    :return: Overlapmatrix
    """

    inner_overlap_mat = vmap(jnp.outer, (1, 1))(system_kwargs["basis_func"], system_kwargs["basis_func"])
    overlap_mat = jnp.trapz(inner_overlap_mat, axis=0)
    return overlap_mat


# @jit
def kin_int(system_kwargs):
    """
    calc the kinetic energy Matrix of size (LxL)
    :param gauss_arg:
    :return: array [LxL]
    """
    kin_op = lambda x, func_coeff, basis: basis * (-(vmap(grad(grad(gauss_func)), (0, None))(x, func_coeff)) / 2)
    kin_part = jit(vmap(vmap(kin_op, (None, 0, None)), (None, None, 0)))(
        system_kwargs["x"], system_kwargs["bfunc_coeff"], system_kwargs["basis_func"])  # verallgemeinern
    return jnp.trapz(kin_part, axis=(2))


def density_mat(gauss_arg):
    dens_mat = jnp.einsum('ni,mi -> nm ', gauss_arg['c'], gauss_arg['c'])
    return dens_mat


if __name__ == "__main__":
    n_grid = 200
    limit = 5
    electron_number = jnp.array(17)
    grid_arr = jnp.linspace(-limit, limit, n_grid, dtype=jnp.float64)
    alpha = jnp.array([100, 2, 3, 4]).reshape((4, 1))
    beta = jnp.array([1, 1, 1, 1]).reshape((4, 1))
    gauss_coeff = jnp.column_stack((beta, alpha))
    gauss_argums = system_args(n_grid, limit, electron_number, gauss_coeff, None)
    # orbital_func(gauss_argums).shape
    # plt.plot(gauss_argums["x"], gauss_argums["basis_func"][0], label = "1")
    # plt.plot(gauss_argums["x"], kin_int(gauss_argums))
    # plt.show()