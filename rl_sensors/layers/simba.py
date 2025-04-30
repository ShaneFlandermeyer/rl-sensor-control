from typing import *

import flax.linen as nn
import jax


class Simba(nn.Module):
  embed_dim: int
  num_blocks: int
  kernel_init: Callable = nn.initializers.he_normal()
  activation: Callable = nn.relu

  @nn.compact
  def __call__(self, input: jax.Array) -> jax.Array:
    x = nn.Dense(self.embed_dim, kernel_init=self.kernel_init)(input)
      

    for _ in range(self.num_blocks):
      skip = x
      x = nn.LayerNorm()(x)
      x = nn.Dense(self.embed_dim, kernel_init=self.kernel_init)(x)
      x = self.activation(x)
      x = nn.Dense(self.embed_dim, kernel_init=self.kernel_init)(x)
      x = x + skip

    x = nn.LayerNorm()(x)
    return x
