from jax.config import config
config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.scipy as jsci
from jax import grad, jit, vmap
from jax.ops import index, index_add, index_update
from jax import random
from jax.tree_util import partial

import numpy as np

a = np.array([[1,2,3,4,5],
     [11,22,33,44,55],
     [111,222,333,444,555]])
b = ["eins", "zwei", "3", "4", "5"]

print([print(j) for i,j in zip(a,b)])


#how to use vmap

x = jnp.array([[1,2,3],
               [11,22,33],
               [111,222,333]])

y = jnp.array([[1],
              [2],
              [3]])
z = jnp.array([[1,2],
               [3,4],
               [5,6]])



dot = lambda x,y : jnp.dot(x,y)
summer = lambda x,y : jnp.sum(vmap(dot,(0,None),0)(x, z), axis= 0)

print(vmap(summer, (None, 0),  None)(x, z))

print(jnp.mod(17,2))


y1 = jnp.array([1,2,3,4,5,6,7,8,9]).reshape((9, 1))
y3 = jnp.array([11,22,33,44,55,66,77,88,99]).reshape((9, 1))
print(jnp.column_stack((y1,y3))[0,1])



ngrid = 200


@jit
def e_conf(num_electrons, ngrid):
    """
    :param num_electrons: describes of number of electrons used for the calculation
    :param ngrid: int, defines the grid size to calculate the maximal posibile number of orbitals
    :return: arraylike, return an 1D array, give the number of electrons in each possible orbital back
    """
    #print(type(ngrid), ngrid)
    #test = jnp.floor_divide(ngrid,2)
    #test = ngrid//2
    #print(type(test),test)
    max_orbs = jnp.zeros(ngrid//2)
    """   
    new_orb = False
    num_orb = num_electrons // 2
    full_orbs = index_update(max_orbs, index[0:num_orb], 2)
    if jnp.mod(num_electrons, 2) == 1:
        fn = index_update(full_orbs, index[num_orb], 1)
    else:
        fn = full_orbs
    #fnT = fn.reshape((-1, 1))
    return fn
    """
    return max_orbs

ele = jnp.array(17)
grid = jnp.array(200)
print(e_conf(17, 200))