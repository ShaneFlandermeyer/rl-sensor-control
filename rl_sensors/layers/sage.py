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
    W = nn.Dense(2*self.embed_dim, name='W', kernel_init=self.kernel_init)
    h_self, h = jnp.split(W(node_features), 2, axis=-1)
    send_edges = jnp.take_along_axis(h, senders[..., None], axis=-2)
    if edge_features is not None:
      W_e = nn.Dense(self.embed_dim, name='W_e', kernel_init=self.kernel_init)
      send_edges = send_edges + W_e(edge_features)

    #####################################
    # Aggregate edges
    #####################################
    in_degree = segment_sum(
        jnp.ones_like(receivers), receivers, num_nodes
    ).astype(float)
    recv_degree = jnp.take_along_axis(in_degree, receivers, axis=-1)
    send_edges *= recv_degree.clip(1, None)[..., None]

    nodes = h_self + segment_sum(send_edges, receivers, num_nodes)

    # Update graph and return
    return dict(
        node_features=nodes,
        senders=senders,
        receivers=receivers,
        edge_features=edge_features,
        global_features=global_features,
    )
