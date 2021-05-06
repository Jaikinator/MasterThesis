from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import partial
from jax.ops import index, index_add, index_update


def create_c_arr(alpha): #u can use beta as well
    return jnp.zeros(alpha.shape)

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


def free_elec(**kwargs):
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
        dist = jnp.array(0.0)
    c = create_c_arr(alpha)
    return alpha, beta, dist, c









