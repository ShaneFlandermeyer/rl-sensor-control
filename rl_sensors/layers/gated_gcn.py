import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Optional


class ResidualGatedGCN(nn.Module):
  embed_dim: int
  share_kv: bool = False
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
    if self.share_kv:
      W = nn.Dense(3*self.embed_dim, name='W', kernel_init=self.kernel_init)
      h, Q, KV = jnp.split(W(node_features), 3, axis=-1)
      key_edges = value_edges = jnp.take_along_axis(
          KV, senders[..., None], axis=-2
      )
    else:
      W = nn.Dense(4*self.embed_dim, name='W', kernel_init=self.kernel_init)
      h, Q, K, V = jnp.split(W(node_features), 4, axis=-1)
      key_edges = jnp.take_along_axis(K, senders[..., None], axis=-2)
      value_edges = jnp.take_along_axis(V, senders[..., None], axis=-2)
    query_edges = jnp.take_along_axis(Q, receivers[..., None], axis=-2)

    #####################################
    # Aggregate edges
    #####################################
    edges = query_edges + key_edges
    if edge_features is not None:
      W_e = nn.Dense(self.embed_dim, name='W_e', kernel_init=self.kernel_init)
      edges = edges + W_e(edge_features)
    eta = jax.nn.sigmoid(edges)
    nodes = h + segment_sum(eta * value_edges, receivers, num_nodes)

    # Update graph and return
    return dict(
        node_features=nodes,
        senders=senders,
        receivers=receivers,
        edge_features=edges,
        global_features=global_features,
    )
