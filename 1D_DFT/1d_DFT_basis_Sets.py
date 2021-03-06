import matplotlib.pyplot as plt
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)

from Params_1d import *
from dft_1d import *


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


def args_basis_func(**kwargs):
    elec_alpha, elec_beta, elec_dist, elec_c = test_LCAO_basis_func(**kwargs)
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
    inner_overlap_mat = vmap(jnp.outer, (1,1))(basis_function, basis_function)

    # shape (n_grid_points, L,L)

    overlap_mat = jnp.trapz(inner_overlap_mat,grid, axis = 0)
    return overlap_mat

@jit
def kin_int(grid, basis_func_args, basis_function):
    """
    calc the kinetic energy Matrix of size (LxL)
    :param gauss_arg:
    :return: array [LxL]
    """
    #  Note give basis function array
    laplace = jit(vmap(vmap(grad(grad(gauss_func)),(0, None, None ,None)),
                   (None, 0, 0, 0)))(grid,
                                         basis_func_args["alpha_arr"],
                                         basis_func_args["beta_arr"],
                                         basis_func_args["pos0_arr"])
    kin_op = vmap(jnp.outer, (1, 1))(basis_function, laplace)
    overlap_mat = jnp.trapz((-1/2 * kin_op) , grid, axis=0)
    return overlap_mat

@jit
def kin_int_numeric(grid, basis_function):
    laplace = vmap(jnp.gradient,(0, None))(vmap(jnp.gradient,(0, None))(basis_function, 0.05), 0.05)
    inner_overlap_mat = vmap(jnp.outer, (1, 1))(basis_function, laplace)
    overlap_mat = jnp.trapz(inner_overlap_mat, grid, axis=0)
    return (-1/2)* overlap_mat

@jit
def elec_nuclear_int(grid,  basis_function):

    interact_basis_func = (1/(grid)) * basis_function

    inner_overlap_mat = vmap(jnp.outer, (1, 1))(basis_function, interact_basis_func)

    overlap_mat = jnp.trapz(inner_overlap_mat, grid, axis=0)
    return overlap_mat

@jit
def dens_func(basis_function, c_matrix):
    return jnp.einsum('ni,mi, nr,mr -> r',c_matrix,c_matrix, basis_function, basis_function )

@jit
def LDA_exchange(grid, basis_function, c_matrix):
    potential = -(3. / jnp.pi) ** (1. / 3.) * dens_func(basis_function, c_matrix)** (1. / 3.)
    return external_potential(grid, basis_function, potential)

@jit
def LDA_exchange_energy(grid, basis_function, c_matrix):
    energy = -3./4.*(3./jnp.pi)**(1./3.)* jnp.trapz((dens_func(basis_function, c_matrix)**(4./3.)), grid)
    return  energy

########################################################################################################################
# calc Coulomb contribution as well as local density and four_center_integral
########################################################################################################################

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
@jit
def element_J_nm(grid, P_ls,  basis_function_m, basis_function_n, basis_function_l, basis_function_s):
    #calculate the sum over all lambda sigma to get one element of the J_nm term
    return jnp.sum(P_ls * four_center_integral_vmap1(grid, basis_function_m, basis_function_n,
                                                               basis_function_l, basis_function_s))
@jit
def J_nm(grid, basis_function, c_matrix):
    P_ls = density_mat(c_matrix)
    return vmap(vmap(element_J_nm, (None, None, None, 0, None, None)), (None, None, 0, None, None, None))\
        (grid, P_ls, basis_function, basis_function, basis_function, basis_function)


########################################################################################################################
# calc  hartree potential
########################################################################################################################


def hartree_exchange_integral_vmap1(grid, basis_function_m, basis_function_n, basis_function_l, basis_function_s):
    #calculate four center integral for all basis function lambda sigma for one pair mu, nu
    return vmap(vmap(four_center_integral,(None, None, None, None, 0)),(None, None, 0, None, None))\
        (grid, basis_function_m, basis_function_n, basis_function_l, basis_function_s)

def element_hartree_exchange(grid, P_ls,  basis_function_m, basis_function_n, basis_function_l, basis_function_s):
    # calculate the sum over all lambda sigma to get one element of the J_nm term
    return jnp.sum(P_ls * hartree_exchange_integral_vmap1(grid, basis_function_m, basis_function_n,
                                                     basis_function_l, basis_function_s))

def hatree_exchange_potential(grid, basis_function, c_matrix):
    P_ls = density_mat(c_matrix)
    return vmap(vmap(element_hartree_exchange, (None, None, 0, None, None, None)), (None, None, None, None, 0, None)) \
                                    (grid, P_ls, basis_function, basis_function, basis_function, basis_function)


@jit
def external_potential(grid, basis_function, ext_pot):
    # all_basis_func() shape L, n_grid_points
    # array of shape n_grid_points
    # Note give basis function array
    pot_therm = ext_pot * basis_function
    inner_overlap_mat = vmap(jnp.outer, (1, 1))(basis_function, pot_therm)

    overlap_mat = jnp.trapz(inner_overlap_mat, grid, axis=0)
    return overlap_mat
@jit
def total_energy(grid, energy, basis_function, c_matrix):
    eig_sum = jnp.sum(energy)
    hartree_energy =1/2 * jnp.sum(c_matrix * J_nm(grid, basis_function, c_matrix) )
    LDA_energy = LDA_exchange_energy(grid,basis_function, c_matrix)
    return eig_sum - hartree_energy + LDA_energy



########################################################################################################################
# Do it analyticaly
########################################################################################################################

#@jit
def calc(grid, basis_args, c_arr):
    # max_iter = 1000
    basis_function = all_basis_func(grid, basis_args)
    c_matrix = density_mat(c_arr)

    f_ks = kin_int(grid, basis_args, basis_function) + J_nm(grid, basis_function, c_matrix)  #kinetic part
    S = S_nm(grid, basis_function) #overlap matrix
    S_inverse = jnp.linalg.inv(S)
    new_ham = jnp.dot(S_inverse, f_ks)
    epsilon, c_matrix_new = jsci.linalg.eigh(new_ham, eigvals_only=False)

    c_matrix = 0.9 * c_matrix + 0.1 * c_matrix_new
    # calculates the Self consistent field calculation

    iter = 0
    energy_arr = [0]
    while iter < max_iter:

        energy, c_arr = calc(grid, basis_args,  c_arr)
        energy_arr.append(energy)

        energy_diff = energy_arr[iter + 1] - energy_arr[iter]

        print(f"step: {iter}, energy: {round(energy, 5)}, energy diff: {round(energy_diff, 5)}")

        if energy_diff < tol:
            print("converged!")
            break
        else:
            print("not converged")
        iter += 1
    return  energy


########################################################################################################################
# Do it numericaly
########################################################################################################################
@jit
def calc_numeric(grid, basis_function, c_arr):

    c_matrix = density_mat(c_arr)

    f_ks = kin_int_numeric(grid, basis_function) + J_nm(grid, basis_function, c_matrix) + LDA_exchange(grid, basis_function, c_matrix)  #kinetic part #+ LDA_exchange(grid, basis_function, c_matrix)
    # print(f"numeric kinetic enrergy: \n{kin_int_numeric(grid, basis_function)} \n "
    #       f"J_nm: \n {J_nm(grid, basis_function, c_matrix)}")
    S = S_nm(grid, basis_function) #overlap matrix
    overlap_eig, overlap_eig_vec = jsci.linalg.eigh(S, eigvals_only=False)
    v_div_sqrt_eig_00 = overlap_eig_vec * 1 / jnp.expand_dims(jnp.sqrt(overlap_eig), -1)
    v_div_sqrt_eig = jnp.matmul(v_div_sqrt_eig_00 , jnp.conj(jnp.transpose(overlap_eig_vec)))
    new_ham = jnp.matmul(v_div_sqrt_eig, jnp.matmul(f_ks, jnp.conj(jnp.transpose(v_div_sqrt_eig))))  #v_div_sqrt_eig adjungiert because Johnatan says

    epsilon, c_matrix_new = jsci.linalg.eigh(new_ham, eigvals_only=False)

    c_matrix_new =  jnp.matmul(v_div_sqrt_eig,c_matrix)
    return epsilon, c_matrix_new

def SCFC_numeric(grid, basis_func, c_arr, max_iter,tol):
    # calculates the Self consistent field calculation

    iter = 0
    energy_arr = [0]

    while iter < max_iter:

        energy, c_arr = calc_numeric(grid, basis_func,  c_arr)
        energy_arr.append(energy)

        energy_diff = energy_arr[iter + 1] - energy_arr[iter]

        print(f"step: {iter}, energy: {round(energy, 5)}, energy diff: {round(energy_diff, 5)}")

        # if abs(energy_diff) < tol:
        #     print("converged!")
        #     print(f"final energy is: {energy}")
        #     break
        # else:
        #     print("not converged")
        iter += 1

    return  energy, c_arr


if __name__ == "__main__":

    n_grid = 201
    limit = 5

    grid_arr = jnp.linspace(-limit, limit, n_grid, dtype=jnp.float64)

    c_arr = jnp. array([[1.0, 0.0],
             [0.0 ,1.0]])

########################################################################################################################
# Do it analyticaly
########################################################################################################################

    # args = args_basis_func(grid_arr = grid_arr, num_electrons = 2)
    #
    #
    # print("\n result using analytic \n ", SCFC(grid_arr, args, c_arr, 1000, 1e-5))
    # basis_func = all_basis_func(grid_arr, args)


    # print(four_center_integral_vmap1(grid_arr, basis_func[0], basis_func[1], basis_func[1], basis_func[0]))
    # print(basis_func[0])

    #plot basis func:

    # plt.title("basis function")
    # plt.plot(grid_arr, basis_func[0], label = "func 1")
    # plt.plot(grid_arr, basis_func[1], label = "func 2")
    # plt.legend()
    # plt.savefig("/home/jacob/PycharmProjects/MasterThesis/1D_DFT/Plots/basis_func.png")


########################################################################################################################
# Do it numericaly
########################################################################################################################
    print("\n \t Calculation numeric basis Set \n")

    basis_func = jnp.transpose(LCAO_basis_func(grid_arr=grid_arr, num_electrons = 2))/jnp.sqrt(0.05)

    tot_energy_real = LCAO_basis_func(grid_arr=grid_arr, num_electrons = 2, tot_energy= True)

    # args = args_basis_func(grid_arr = grid_arr, num_electrons = 2)
    # basis_func = all_basis_func(grid_arr, args)
    # plt.plot(grid_arr,basis_func[0])
    # plt.plot(grid_arr, basis_func[1])
    # plt.show()

    energy, c_matrix = SCFC_numeric(grid_arr, basis_func, c_arr, 2, 1e-5)
    tot_energy = total_energy(grid_arr, energy, basis_func, c_matrix)
    print(f"\n\t total energy basis func: {tot_energy} \n"
          f"\t total energy realspace: {tot_energy_real} \n"
          f"\t diff: {tot_energy - tot_energy_real}")



