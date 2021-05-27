calculate Basis functions --> array(n_basis functions, nrgidpoints)

bzw jax
    calculate Basis function --> array(1, nrgidpoints)
    and vmap


calculate Smunu (bf_1 = (1, gridpoint), bf_2 =(1,gridpoint)) --> scalar
    jnp.traps(bf_1*bf_2, grid)

    bf = bf_1 * bf_2

    jnp.trapz(bf)

basis_mu_r1 * basis_ny_r1 * 1 * basis_lambda_r2 * basis_sigma_r2

int basis_mu_r1 * basis_ny_r2

