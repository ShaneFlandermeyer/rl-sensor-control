import jax.numpy as jnp

def mish(x: jnp.ndarray) -> jnp.ndarray:
  return x * jnp.tanh(jnp.log1p(jnp.exp(x)))