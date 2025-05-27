
from functools import partial
from typing import *

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from rl_sensors.layers.attention import PMA, AttentionBlock
from rl_sensors.layers.gat import GATv2
from rl_sensors.layers.gin import GIN
from rl_sensors.envs.graph_search import GraphSearchEnv
from rl_sensors.layers.simba import Simba


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
    edge_features = input['edge_features']
    edge_list = input['edge_list']
    edge_mask = input['edge_mask']
    global_features = input['global_features']
    node_features = input['node_features']
    node_mask = input['node_mask']
    senders, receivers = edge_list[..., 0], edge_list[..., 1]

    batch_dims = node_features.shape[:-2]
    num_nodes = node_features.shape[-2]

    # Add global features to node representation
    if global_features is not None:
      node_features = jnp.concatenate([
          node_features,
          global_features.repeat(num_nodes, axis=-2),
      ], axis=-1
      )

    # Add a dummy node for padding edges
    num_nodes += 1
    pad_node = jnp.zeros((*batch_dims, 1, node_features.shape[-1]))
    pad_mask = jnp.zeros((*batch_dims, 1))
    node_features = jnp.concatenate([node_features, pad_node], axis=-2)
    node_mask = jnp.concatenate([node_mask, pad_mask], axis=-1)
    senders = jnp.where(edge_mask, senders, -1)
    receivers = jnp.where(edge_mask, receivers, -1)

    graph = dict(
        node_features=node_features,
        edge_features=edge_features,
        senders=senders,
        receivers=receivers,
        global_features=jnp.tile(
            self.param('global', nn.initializers.zeros, (1, self.embed_dim)),
            [*batch_dims, 1, 1]
        ),
    )

    ######################
    # Graph Processing
    ######################
    # Encode
    graph['node_features'] = nn.Sequential([
        nn.Dense(self.embed_dim, kernel_init=self.kernel_init),
        nn.LayerNorm(),
        nn.relu,
        nn.Dense(self.embed_dim, kernel_init=self.kernel_init),
    ])(graph['node_features'])
    graph['global_features'] = AttentionBlock(
        embed_dim=self.embed_dim,
        num_heads=self.num_heads,
        hidden_dim=self.embed_dim,
        normalize_qk=True,
        use_ffn=False,
        kernel_init=self.kernel_init,
    )(
        query=graph['global_features'], key=graph['node_features'],
        query_mask=None, key_mask=node_mask,
    )
    graph['node_features'] = jnp.concatenate([
        graph['node_features'],
        graph['global_features'].repeat(num_nodes, axis=-2),
    ], axis=-1)
    graph['node_features'] = nn.relu(
        nn.Dense(
            self.embed_dim, kernel_init=self.kernel_init
        )(graph['node_features'])
    )

    for i in range(self.num_layers):
      # Layer definitions
      W_skip = nn.Dense(
          self.embed_dim, kernel_init=self.kernel_init, name=f'W_skip_{i}'
      )
      gnn = GATv2(
          embed_dim=self.embed_dim,
          num_heads=self.num_heads,
          share_weights=False,
          kernel_init=self.kernel_init,
      )

      # Graph update
      graph['node_features'] = nn.LayerNorm()(graph['node_features'])
      skip = W_skip(graph['node_features'])
      graph = gnn(**graph)
      graph['node_features'] = nn.relu(graph['node_features'] + skip)

    # Decode
    graph['global_features'] = AttentionBlock(
        embed_dim=self.embed_dim,
        num_heads=self.num_heads,
        hidden_dim=self.embed_dim,
        normalize_qk=True,
        use_ffn=False,
        kernel_init=self.kernel_init,
    )(
        query=graph['global_features'],
        key=graph['node_features'],
        query_mask=None,
        key_mask=node_mask,
    )
    graph['global_features'] = nn.relu(
        nn.LayerNorm()(graph['global_features'])
    )
    x = rearrange(graph['global_features'], '... n d -> ... (n d)')

    return x


if __name__ == '__main__':
  env = gym.vector.SyncVectorEnv([lambda: GraphSearchEnv() for _ in range(1)])
  obs, _ = env.reset(seed=0)
  for i in range(0):
    obs = env.step(env.action_space.sample())[0]

  embed_dim = 128
  latent_dim = 512
  num_layers = 1
  num_heads = 4
  model = nn.Sequential([
      GraphEncoder(
          embed_dim=embed_dim,
          num_layers=num_layers,
          num_heads=num_heads,
      ),
      nn.Dense(latent_dim)
      #   NormedLinear(latent_dim, activation=nn.relu),
  ])

  model.init(jax.random.PRNGKey(0), obs)
  print(model.tabulate(jax.random.key(0), obs, compute_flops=True))
