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

  @nn.compact
  def __call__(self,
               node_features: jax.Array,
               edge_features: jax.Array,
               global_features: jax.Array,
               senders: jax.Array,
               receivers: jax.Array,
               ) -> jax.Array:
    ############################
    # Pre-processing
    ############################
    graph = dict(
        node_features=node_features,
        edge_features=edge_features,
        global_features=global_features,
        senders=senders,
        receivers=receivers,
    )
    batch_dims = node_features.shape[:-2]
    num_nodes = node_features.shape[-2]

    segment_sum = jax.ops.segment_sum
    segment_softmax = segment_util.segment_softmax
    for _ in range(len(batch_dims)):
      segment_sum = jax.vmap(segment_sum, in_axes=(0, 0, None))
      segment_softmax = jax.vmap(segment_softmax, in_axes=(0, 0, None))

    ############################
    # Edge update
    ############################
    if self.share_weights:
      W = nn.Dense(self.embed_dim, name='W', kernel_init=self.kernel_init)
      send_nodes = recv_nodes = W(node_features)
    else:
      W = nn.Dense(2*self.embed_dim, name='W', kernel_init=self.kernel_init)
      send_nodes, recv_nodes = jnp.split(W(node_features), 2, axis=-1)
    send_edges = jnp.take_along_axis(send_nodes, senders[..., None], axis=-2)
    recv_edges = jnp.take_along_axis(recv_nodes, receivers[..., None], axis=-2)
    x = send_edges + recv_edges

    if edge_features is not None:
      W_e = nn.Dense(self.embed_dim, name='W_e', kernel_init=self.kernel_init)
      x += W_e(edge_features)

    x = mish(x)

    ############################
    # Attention
    ############################
    x = rearrange(x, '... (h d) -> ... h d', h=self.num_heads)
    a = self.param(
        'a',
        self.kernel_init,
        (self.num_heads, self.embed_dim // self.num_heads)
    )
    a = jnp.tile(a, (*x.shape[:-2], 1, 1))
    logits = jnp.sum(x * a, axis=-1, keepdims=True)
    weights = segment_softmax(logits, receivers, num_nodes)

    ############################
    # Node Update
    ############################
    send_edges = rearrange(
        send_edges, '... (h d) -> ... h d', h=self.num_heads
    )
    send_edges = weights * send_edges
    send_edges = rearrange(send_edges, '... h d -> ... (h d)')
    updated_node_features = segment_sum(send_edges, receivers, num_nodes)

    # Update graph
    graph.update(node_features=updated_node_features)
    return graph
