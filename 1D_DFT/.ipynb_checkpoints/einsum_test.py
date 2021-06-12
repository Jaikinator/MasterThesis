import jax.numpy as jnp

a = jnp.arange(25).reshape(5,5)

b = jnp.arange(5)

c = jnp.arange(6).reshape(2,3)

print(f"a: \n {a}, \n b: \n {b}, \n c:  \n {c}")

# Trace of a matrix:

print(f"Trace of a matrix:\n"
      f"jnp.einsum('ii', a): \n {jnp.einsum('ii', a)} \n"
      f"jnp.einsum(a, [0,0]): \n {jnp.einsum(a, [0,0])} \n"
      f"jnp.trace(a): \n {jnp.trace(a)} \n") #sum along diaginal elements

# Extrakt a diagonal

print(f"Extrakt a diagonal \n"
      f"jnp.einsum('ii->i', a): \n {jnp.einsum('ii->i', a)} \n"
      f"jnp.einsum(a, [0,0], [0]): \n {jnp.einsum(a, [0,0], [0])} \n"
      f"jnp.diag(a): \n {jnp.diag(a)} \n") #sum along diaginal elements

# sum over axis

print(f"sum over axis:\n"
      f"jnp.einsum('ij->i', a): \n {jnp.einsum('ij->i', a)} \n"
      f"jnp.einsum(a, [0,1], [0]): \n {jnp.einsum(a, [0,1], [0])} \n"
      f"jnp.sum(a, axis=1): \n {jnp.sum(a, axis=1)} \n") #sum along diaginal elements

# Compute a matrix transpose, or reorder any number of axes

print(f"Compute a matrix transpose, or reorder any number of axes: \n"
      f"jnp.einsum('ji', c): \n {jnp.einsum('ji', c)} \n"
      f"jnp.einsum('ij->ji', c): \n {jnp.einsum('ij->ji', c)} \n",
      f"jnp.einsum(c, [1,0]): \n {jnp.einsum(c, [1,0])} \n"
      f"jnp.transpose(c): \n {jnp.transpose(c)}")

# Vector inner products

print(f"Vector inner products: \n"
      f"jnp.einsum('i,i', b, b): \n {jnp.einsum('i,i', b, b)} \n"
      f"jnp.einsum(b, [0], b, [0]): \n {jnp.einsum(b, [0], b, [0])} \n",
      f"jnp.inner(b,b): \n {jnp.inner(b,b)} \n")

# Matrix vector multiplication:

print(f" Matrix vector multiplication:\n"
      f"jnp.einsum('ij,j', a, b): \n {jnp.einsum('ij,j', a, b)} \n"
      f"jnp.einsum(a, [0,1], b, [1]): \n {jnp.einsum(a, [0,1], b, [1])} \n",
      f"jnp.dot(a, b): \n {jnp.dot(a, b)} \n"
      f"jnp.einsum('...j,j', a, b): \n {jnp.einsum('...j,j', a, b)} \n")



tbfunc = jnp.array([[1,2,3],[1,1,1]])
c = jnp.arange(0, 20).reshape((2,10))
#print(jnp.einsum('ik,jk -> ijk', tbfunc,tbfunc))
print(jnp.einsum('ni,mi -> nm ', c,c))