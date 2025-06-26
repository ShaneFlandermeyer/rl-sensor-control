import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Optional


class GraphSAGE(nn.Module):
  """
  The GraphSAGE operator from the “Inductive Representation Learning on Large Graphs” paper.

  Incorporates edge features in the node update step
  """
  embed_dim: int
  kernel_init: Callable = nn.initializers.xavier_normal()
  dtype: jnp.dtype = jnp.float32

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
    W = nn.Dense(
        2*self.embed_dim,
        name='W',
        kernel_init=self.kernel_init,
        dtype=self.dtype
    )
    xi, xj = jnp.split(W(node_features), 2, axis=-1)
    xji = jnp.take_along_axis(xj, senders[..., None], axis=-2)
    if edge_features is not None:
      W_e = nn.Dense(
          self.embed_dim,
          name='W_e',
          kernel_init=self.kernel_init,
          dtype=self.dtype
      )
      xji = xji + W_e(edge_features)

    #####################################
    # Aggregate edges
    #####################################
    in_degree = segment_sum(jnp.ones_like(receivers), receivers, num_nodes)
    xji /= jnp.take_along_axis(in_degree, receivers, axis=-1)[..., None] + 1e-6

    nodes = xi + segment_sum(xji, receivers, num_nodes)

    # Update graph and return
    return dict(
        node_features=nodes,
        senders=senders,
        receivers=receivers,
        edge_features=edge_features,
        global_features=global_features,
    )
