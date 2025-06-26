
from functools import partial
from typing import *

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

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


class PGAT(nn.Module):
  """Pooling by graph attention layer."""
  embed_dim: int
  num_heads: int
  normalize_inputs: bool = True
  residual: bool = True
  kernel_init: Callable = nn.initializers.xavier_normal()
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      query: jax.Array,
      key: jax.Array,
      query_mask: jax.Array,
      key_mask: jax.Array
  ) -> jax.Array:
    # Pre-processing
    skip = query
    if self.normalize_inputs:
      query = nn.LayerNorm()(query)
      key = nn.LayerNorm()(key)
    query = nn.Dense(
        self.embed_dim,
        name='W_query',
        kernel_init=self.kernel_init,
        dtype=self.dtype
    )(query)
    key = nn.Dense(
        self.embed_dim,
        name='W_key',
        kernel_init=self.kernel_init,
        dtype=self.dtype
    )(key)
    # Masking
    if query_mask is None:
      query_mask = jnp.ones(query.shape[:-1], dtype=bool)
    if key_mask is None:
      key_mask = jnp.ones(key.shape[:-1], dtype=bool)
    mask = (
        jnp.expand_dims(query_mask, axis=-1) *
        jnp.expand_dims(key_mask, axis=-2)
    )

    # Multi-head attention
    query = rearrange(query, '... (h d) -> ... h d', h=self.num_heads)
    key = rearrange(key, '... (h d) -> ... h d', h=self.num_heads)
    # Logits
    x = mish(jnp.expand_dims(query, axis=-3) + jnp.expand_dims(key, axis=-4))
    a = self.param(
        'a',
        self.kernel_init,
        (self.num_heads, self.embed_dim // self.num_heads),
    )
    logits = jnp.einsum('h d, ... h d -> ... h', a.astype(x.dtype), x)
    # Softmax weights
    logits = jnp.where(mask[..., None], logits, jnp.finfo(key.dtype).min)
    weights = jax.nn.softmax(logits, axis=-2)
    x = jnp.einsum('... m n h, ... n h d -> ... m h d', weights, key)

    x = rearrange(x, '... h d -> ... (h d)', h=self.num_heads)
    if self.residual:
      x = x + skip
    return x
