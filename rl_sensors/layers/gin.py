import time
from typing import *
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from rl_sensors.layers.activation import mish


class GIN(nn.Module):
  mlp: nn.Module
  epsilon: Optional[float] = None
  kernel_init: Callable = nn.initializers.xavier_normal()

  @nn.compact
  def __call__(self,
               node_features: jax.Array,
               edge_features: jax.Array,
               global_features: jax.Array,
               senders: jax.Array,
               receivers: jax.Array
               ) -> jax.Array:
    ############################
    # Pre-processing
    ############################
    leading_dims = node_features.shape[:-2]
    num_nodes = node_features.shape[-2]
    embed_dim = node_features.shape[-1]

    segment_sum = jax.ops.segment_sum
    for _ in range(len(leading_dims)):
      segment_sum = jax.vmap(segment_sum, in_axes=(0, 0, None))

    ####################################
    # Edge update
    ####################################
    send_edges = jnp.take_along_axis(
        node_features, senders[..., None], axis=-2
    )
    if edge_features is not None:
      W_e = nn.Dense(embed_dim, name='W_e', kernel_init=self.kernel_init)
      send_edges = nn.relu(send_edges + W_e(edge_features))

    ####################################
    # Node update
    ####################################
    if self.epsilon is None:
      epsilon = self.param('epsilon', nn.initializers.zeros, (1, 1))
    else:
      epsilon = self.epsilon
    epsilon = jnp.tile(epsilon, (*leading_dims, 1, 1))

    new_nodes = self.mlp(
        (1 + epsilon) * node_features +
        segment_sum(send_edges, receivers, num_nodes)
    )
    
    if global_features is not None:
      W_g = nn.Dense(embed_dim, name='W_g', kernel_init=self.kernel_init)
      new_nodes = nn.relu(new_nodes + W_g(global_features))

    return dict(
        node_features=new_nodes,
        edge_features=edge_features,
        global_features=global_features,
        senders=senders,
        receivers=receivers,
    )
