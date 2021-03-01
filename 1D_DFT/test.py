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