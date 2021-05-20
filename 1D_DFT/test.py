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
import dft_1d as dft

# a = np.array([[1,2,3,4,5],
#      [11,22,33,44,55],
#      [111,222,333,444,555]])
# b = ["eins", "zwei", "3", "4", "5"]
#
# print([print(j) for i,j in zip(a,b)])
#
#
# #how to use vmap
#
# x = jnp.array([[1,2,3],
#                [11,22,33],
#                [111,222,333]])
#
# y = jnp.array([[1],
#               [2],
#               [3]])
# z = jnp.array([[1,2],
#                [3,4],
#                [5,6]])
#
#
#
# dot = lambda x,y : jnp.dot(x,y)
# summer = lambda x,y : jnp.sum(vmap(dot,(0,None),0)(x, z), axis= 0)
#
# print(vmap(summer, (None, 0),  None)(x, z))
#
# print(jnp.mod(17,2))
#
#
# y1 = jnp.array([1,2,3,4,5,6,7,8,9]).reshape((9,1))
# y3 = jnp.array([11,22,33,44,55,66,77,88,99]).reshape((9, 1))
# print(y1,y3)
#print(jnp.column_stack((y1,y3))[0,1])
#
# def E_field(x,c):
#     return x*c
#
# @jit
# def kwarg_ret(x,pot_kwargs):
#     for key in pot_kwargs.keys():
#         if key == "c":
#             return E_field(x,pot_kwargs[key])
# @jit
# def test_calc(x,y,pot_kwarg):
#     Energy = kwarg_ret(x,pot_kwarg)
#     return y*x * Energy
#
# test_dict = {"c" : 0}
#
# print(test_calc(x,y, test_dict))
#
# c = dft.e_conf(17,200)
# print("len",len(jnp.where(c != 0)[0]))
#
# test = jnp.array([[0,1,2,3,4,5],
#                     [0,1,2,3,4,5],
#                     [0,1,2,3,4,5],
#                     [0,1,2,3,4,5],
#                     [0,1,2,3,4,5]])
#
# print(f"coulum wise test[:,0]: {test[:,0]}")
# print(f"row wise test[0,:]: {test[0,:]}")

b = jnp.arange(16).reshape(4, 4)
c = b + b.T
print(np.matmul(c, np.linalg.eigh((c) * 1.0)[1][0]) / np.linalg.eigh(c)[0][0] - np.linalg.eigh(c)[1][0])
print(np.matmul(c, np.linalg.eigh((c) * 1.0)[1][:, 0]) / np.linalg.eigh(c)[0][0] - np.linalg.eigh(c)[1][:, 0])
