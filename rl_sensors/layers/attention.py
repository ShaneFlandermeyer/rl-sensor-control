
from functools import partial
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from rl_sensors.layers.activation import mish


class AttentionBlock(nn.Module):
  embed_dim: int
  hidden_dim: int
  num_heads: int
  normalize_qk: bool = False
  use_ffn: bool = True
  kernel_init: nn.initializers.Initializer = nn.initializers.xavier_normal()
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               query: jax.Array,
               key: jax.Array,
               query_mask: jax.Array = None,
               key_mask: jax.Array = None):
    if query_mask is None:
      query_mask = jnp.ones(query.shape[:-1], dtype=bool)
    if key_mask is None:
      key_mask = jnp.ones(key.shape[:-1], dtype=bool)
    mask = nn.make_attention_mask(query_mask, key_mask)

    # Attention
    mha = nn.MultiHeadAttention(
        num_heads=self.num_heads,
        kernel_init=self.kernel_init,
        dtype=self.dtype,
    )
    skip = query
    if self.normalize_qk:
      query = nn.LayerNorm()(query)
      key = nn.LayerNorm()(key)
    x = skip + mha(inputs_q=query, inputs_kv=key, mask=mask)

    # FFN
    if self.use_ffn:
      ffn = nn.Sequential([
          nn.LayerNorm(),
          nn.Dense(
              self.hidden_dim, kernel_init=self.kernel_init, dtype=self.dtype
          ),
          mish,
          nn.Dense(
              self.embed_dim, kernel_init=self.kernel_init, dtype=self.dtype
          ),
      ], name='ffn')

      x = x + ffn(x)

    return x


class PMA(nn.Module):
  attention_base: nn.Module
  num_seeds: int = 1
  seed_init: nn.initializers.Initializer = nn.initializers.xavier_normal()

  @nn.compact
  def __call__(
      self,
      x: jax.Array,
      valid: jax.Array = None,
  ):
    batch_dims, embed_dim = x.shape[:-2], x.shape[-1]

    S = self.param('S', self.seed_init, (self.num_seeds, embed_dim))
    S = jnp.tile(S, [*batch_dims, 1, 1])

    x = self.attention_base(
        query=S,
        key=x,
        query_mask=jnp.ones(self.num_seeds, dtype=bool),
        key_mask=valid
    )
    return x
