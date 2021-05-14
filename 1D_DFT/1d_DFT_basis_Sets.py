from jax.config import config
config.update("jax_enable_x64", True)

from Params_1d import *

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


def args_basis_func():
    elec_alpha, elec_beta, elec_dist, elec_c = H2()
    return {"alpha_arr": elec_alpha,
            "beta_arr": elec_beta,
            "pos0_arr": elec_dist}

@jit
def all_basis_func(grid, system_kwargs):
    #print(len(jnp.where(num_orb != 0)))
    return vmap(gauss_func, (None, 0, 0, 0))(grid, system_kwargs["alpha_arr"],
                                             system_kwargs["beta_arr"], system_kwargs["pos0_arr"])

@jit
def S_nm(grid, basis_function):
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

    overlap_mat = jnp.trapz(inner_overlap_mat,grid, axis = 0)
    return overlap_mat

@jit
def kin_int(system_kwargs):
    """
    calc the kinetic energy Matrix of size (LxL)
    :param gauss_arg:
    :return: array [LxL]
    """
    laplace = jit(vmap(vmap(grad(grad(gauss_func)),(0, None, None ,None)),
                   (None, 0, 0, 0)))(grid,
                                         basis_func_args["alpha_arr"],
                                         basis_func_args["beta_arr"],
                                         basis_func_args["pos0_arr"])
    kin_op = vmap(jnp.outer, (1, 1))(basis_function, laplace)
    overlap_mat = jnp.trapz(kin_op,grid, axis=0)
    return overlap_mat

@jit
def density_mat(c_arr):
    dens_mat = jnp.einsum('ni,mi -> nm', c_arr, c_arr)
    return dens_mat

@jit
def soft_coulomb(r_1,r_2):
        return 1/jnp.sqrt(1 + (r_1-r_2)**2)

@jit
def four_center_integral(grid, basis_function_m, basis_function_n, basis_function_l, basis_function_s):
        soft_coul_mat = vmap(soft_coulomb, (0, None), 1)(grid, grid)
        inner =  basis_function_m.reshape(1,-1) * basis_function_n.reshape(1,-1) \
                 * soft_coul_mat\
                 * basis_function_l.reshape(-1,1) * basis_function_s.reshape(-1,1)
        return jnp.trapz(jnp.trapz(inner, grid, axis = 0 ), grid, axis = 0)

@jit
def four_center_integral_vmap1(grid, basis_function_m, basis_function_n, basis_function_l, basis_function_s):
    #calculate four center integral for all basis function lambda sigma for one pair mu, nu
    return vmap(vmap(four_center_integral,(None, None, None, None, 0)),(None, None, None, 0, None))\
        (grid, basis_function_m, basis_function_n, basis_function_l, basis_function_s)






#
# def calc(grid, alpha, beta, pos, c_arr):
#     system_kwargs = {"grid": grid,
#              "alpha_arr": alpha,
#              "beta_arr": beta,
#              "pos0_arr":pos,
#              "c": c_arr}
#     basis_function = all_basis_func(system_kwargs)
#     c_matrix = density_mat(system_kwargs)
#
#     f_ks =-1/2 * kin_int(system_kwargs, basis_function) + J_nm(system_kwargs["grid"], basis_function, c_matrix) #kinetic part
#     S = S_nm(system_kwargs["grid"], basis_function) #overlap matrix
#     S_inverse = jnp.linalg.inv(S)
#     new_ham = jnp.dot(S_inverse, f_ks)
#     epsilon, c_matrix = jsci.linalg.eigh(new_ham, eigvals_only=False)
#     return jnp.sum(epsilon)




if __name__ == "__main__":

    n_grid = 200
    limit = 5

    grid_arr = jnp.linspace(-limit, limit, n_grid, dtype=jnp.float64)

    alpha, beta, pos , c_arr = H2()

    #print(all_basis_func(grid_arr, elec1_alpha, elec1_beta , elec1_dist[0]).shape)

    args = args_basis_func()

    #print(calc(args))
    # print(grad(calc,(1,2))(grid_arr, alpha, beta, pos, c_arr))
    print(grad(calc, (1, 2))(grid_arr,args, c_arr))


