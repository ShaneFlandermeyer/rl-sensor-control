import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Optional


class GCN(nn.Module):
  """
  Implementation of a Graph Convolution layer from "Semi-Supervised Classification with Graph Convolutional Networks".

  Incorporates edge features in the node update step
  """
  embed_dim: int
  normalize: bool = True
  add_self_edges: bool = False
  kernel_init: Callable = nn.initializers.xavier_normal()

  @nn.compact
  def __call__(self,
               node_features: jax.Array,
               senders: jax.Array,
               receivers: jax.Array,
               edge_features: Optional[jax.Array] = None,
               global_features: Optional[jax.Array] = None,
               ) -> jax.Array:
    batch_dims = node_features.shape[:-2]
    num_nodes = node_features.shape[-2]

    segment_sum = jax.ops.segment_sum
    for _ in range(len(batch_dims)):
      segment_sum = jax.vmap(segment_sum, in_axes=(0, 0, None))

    ####################################
    # Node/edge update
    ####################################
    W = nn.Dense(self.embed_dim, name='W', kernel_init=self.kernel_init)
    h = W(node_features)

    send_edges = jnp.take_along_axis(h, senders[..., None], axis=-2)
    if edge_features is not None:
      W_e = nn.Dense(self.embed_dim, name='W_e', kernel_init=self.kernel_init)
      send_edges = send_edges + W_e(edge_features)

    if self.add_self_edges:
      send_edges = jnp.concatenate([send_edges, h], axis=-2)
      node_inds = jnp.tile(jnp.arange(num_nodes), [*batch_dims, 1])
      senders_ = jnp.concatenate([senders, node_inds], axis=-1)
      receivers_ = jnp.concatenate([receivers, node_inds], axis=-1)
    else:
      senders_ = senders
      receivers_ = receivers

    #####################################
    # Aggregate edges
    #####################################
    if self.normalize:
      in_degree = segment_sum(
          jnp.ones_like(receivers_), receivers_, num_nodes
      ).astype(float)
      send_degree = jnp.take_along_axis(in_degree, senders_, axis=-1)
      recv_degree = jnp.take_along_axis(in_degree, receivers_, axis=-1)
      send_edges *= jax.lax.rsqrt(
          send_degree.clip(1, None) * recv_degree.clip(1, None)
      )[..., None]

    nodes = segment_sum(send_edges, receivers_, num_nodes)

    # Update graph and return
    return dict(
        node_features=nodes,
        senders=senders,
        receivers=receivers,
        edge_features=edge_features,
        global_features=global_features,
    )
