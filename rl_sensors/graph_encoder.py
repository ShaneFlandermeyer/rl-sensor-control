
from typing import *

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
from einops import rearrange

from rl_sensors.envs.graph_search_track import GraphSearchTrackEnv
from rl_sensors.layers.activation import mish
from rl_sensors.layers.attention import PGAT
from rl_sensors.layers.gat import GATv2
from rl_sensors.layers.gcn import GCN, ResidualGatedGCN


class GraphEncoder(nn.Module):
  embed_dim: int
  num_layers: int
  num_heads: int
  kernel_init: Callable = nn.initializers.xavier_normal()

  @nn.compact
  def __call__(self, input: Dict[str, Any]):
    ######################
    # Pre-processing
    ######################
    current_agent_node_ind = input['current_agent_node_ind'][..., None]
    edge_features = input['edge_features']
    edge_list = input['edge_list']
    edge_mask = input['edge_mask']
    global_features = input['global_features']
    node_features = input['node_features']
    node_mask = input['node_mask']
    senders, receivers = edge_list[..., 0], edge_list[..., 1]
    senders = jnp.where(edge_mask, senders, -1)
    receivers = jnp.where(edge_mask, receivers, -1)

    batch_dims = node_features.shape[:-2]
    num_nodes = node_features.shape[-2]

    # Add global features to node representation
    if global_features is not None:
      node_features = jnp.concatenate([
          node_features,
          global_features.repeat(num_nodes, axis=-2),
      ], axis=-1
      )

    node_features = nn.Dense(
        self.embed_dim, kernel_init=self.kernel_init
    )(node_features)
    edge_features = nn.Dense(
        self.embed_dim, kernel_init=self.kernel_init
    )(edge_features)
    # Only have to do this once since edge features aren't updated
    edge_features = nn.relu(nn.LayerNorm()(edge_features))

    ######################
    # Graph Processing
    ######################
    graph = dict(
        node_features=node_features,
        senders=senders,
        receivers=receivers,
        edge_features=edge_features,
        global_features=None,
    )

    for i in range(self.num_layers):
      # Layer definitions
      gnn = GCN(
          embed_dim=self.embed_dim,
          kernel_init=self.kernel_init,
          normalize=True,
          add_self_edges=False,
          residual=True,
      )
      # Graph update
      graph['node_features'] = nn.relu(nn.LayerNorm()(graph['node_features']))
      graph = gnn(**graph)

    ######################
    # Decode
    ######################
    current_agent_node = jnp.take_along_axis(
        graph['node_features'], current_agent_node_ind[..., None], axis=-2
    )
    x = PGAT(
        embed_dim=self.embed_dim,
        num_heads=self.num_heads,
        kernel_init=self.kernel_init,
        normalize_inputs=True,
        residual=True,
    )(
        query=current_agent_node,
        key=graph['node_features'],
        query_mask=None,
        key_mask=node_mask
    )
    x = nn.relu(nn.LayerNorm()(x))
    x = rearrange(x, '... n d -> ... (n d)')

    return x


if __name__ == '__main__':
  env = gym.vector.SyncVectorEnv(
      [lambda: GraphSearchTrackEnv() for _ in range(1)])
  obs, _ = env.reset(seed=0)
  for i in range(30):
    obs = env.step(env.action_space.sample())[0]

  embed_dim = 128
  latent_dim = 512
  num_layers = 3
  num_heads = 4
  model = nn.Sequential([
      GraphEncoder(
          embed_dim=embed_dim,
          num_layers=num_layers,
          num_heads=num_heads,
          kernel_init=nn.initializers.truncated_normal(stddev=0.02)
      ),
      nn.Dense(latent_dim)
      # NormedLinear(latent_dim, activation=mish),
  ])

  model.init(jax.random.PRNGKey(0), obs)
  print(model.tabulate(jax.random.key(0), obs, compute_flops=True))
