import time
from typing import Optional, Tuple
import flax.linen as nn
from rl_sensors.layers import segment_util
import jax
import jax.numpy as jnp
from einops import rearrange
from rl_sensors.layers.activation import mish


class GATv2(nn.Module):
  """
  Implementation of GATv2 from [1] with global and edge features.

  [1] https://arxiv.org/abs/2105.14491
  """
  embed_dim: int
  num_heads: int
  share_weights: bool = True
  kernel_init: nn.initializers.Initializer = nn.initializers.xavier_normal()
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               node_features: jax.Array,
               senders: jax.Array,
               receivers: jax.Array,
               edge_features: Optional[jax.Array] = None,
               global_features: Optional[jax.Array] = None,
               ) -> jax.Array:
    ############################
    # Pre-processing
    ############################
    batch_dims = node_features.shape[:-2]
    num_nodes = node_features.shape[-2]

    segment_sum = jax.ops.segment_sum
    segment_softmax = segment_util.segment_softmax
    for _ in range(len(batch_dims)):
      segment_sum = jax.vmap(segment_sum, in_axes=(0, 0, None))
      segment_softmax = jax.vmap(segment_softmax, in_axes=(0, 0, None))

    ############################
    # Node/edge update
    ############################
    if self.share_weights:
      W = nn.Dense(
          self.embed_dim,
          name='W',
          kernel_init=self.kernel_init,
          dtype=self.dtype
      )
      h_send = h_recv = W(node_features)
    else:
      W = nn.Dense(
          2*self.embed_dim,
          name='W',
          kernel_init=self.kernel_init,
          dtype=self.dtype
      )
      h_send, h_recv = jnp.split(W(node_features), 2, axis=-1)
    e_send = jnp.take_along_axis(h_send, senders[..., None], axis=-2)
    e_recv = jnp.take_along_axis(h_recv, receivers[..., None], axis=-2)
    e_attn = e_send + e_recv

    if edge_features is not None:
      W_e = nn.Dense(
          self.embed_dim,
          name='W_e',
          kernel_init=self.kernel_init,
          dtype=self.dtype,
      )
      e_attn += W_e(edge_features)

    e_attn = mish(e_attn)

    ############################
    # Attention
    ############################
    e_attn = rearrange(e_attn, '... (h d) -> ... h d', h=self.num_heads)
    logits = nn.Einsum(
        shape=(self.num_heads, self.embed_dim // self.num_heads),
        einsum_str='... h d, h d -> ... h',
        use_bias=False,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        name='a',
    )(e_attn)
    weights = segment_softmax(logits, receivers, num_nodes)

    ############################
    # Aggregate edges
    ############################
    e_send = rearrange(e_send, '... (h d) -> ... h d', h=self.num_heads)
    e_send = rearrange(weights[..., None] * e_send, '... h d -> ... (h d)')
    nodes = segment_sum(e_send, receivers, num_nodes)

    # Update graph
    return dict(
        node_features=nodes,
        senders=senders,
        receivers=receivers,
        edge_features=edge_features,
        global_features=global_features,
    )
