
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
    global_features = input['global_features']
    node_features = input['node_features']

    senders, receivers = edge_list[..., 0], edge_list[..., 1]

    ######################
    # Graph Processing
    ######################
    # Add global features to ALL node embeddings
    if global_features is not None:
      node_features = jnp.concatenate([
          node_features,
          global_features.repeat(node_features.shape[-2], axis=-2),
      ], axis=-1
      )

    graph = dict(
        node_features=node_features,
        edge_features=edge_features,
        senders=senders,
        receivers=receivers,
        global_features=None,
    )

    for i in range(self.num_layers):
      # Graph update
      graph['node_features'] = skip = nn.Dense(
          self.embed_dim, kernel_init=self.kernel_init
      )(graph['node_features'])
      graph = GIN(
          mlp=nn.Sequential([
              nn.LayerNorm(),
              nn.Dense(self.embed_dim, kernel_init=self.kernel_init),
              nn.relu,
              nn.Dense(self.embed_dim, kernel_init=self.kernel_init),
          ])
      )(**graph)
      graph['node_features'] = nn.relu(
          nn.LayerNorm()(graph['node_features'] + skip)
      )

      # Global pooling
      graph['global_features'] = PMA(
          num_seeds=1,
          seed_init=nn.initializers.zeros,
          attention_base=AttentionBlock(
              embed_dim=self.embed_dim,
              hidden_dim=None,
              num_heads=self.num_heads,
              use_ffn=False,
              kernel_init=self.kernel_init,
          )
      )(x=graph['node_features'])
    # Post-processing
    x = rearrange(graph['global_features'], '... n d -> ... (n d)')

    return x


if __name__ == '__main__':
  env = gym.vector.SyncVectorEnv([lambda: GraphSearchEnv() for _ in range(1)])
  obs, _ = env.reset(seed=0)
  for i in range(10):
    obs = env.step(env.action_space.sample())[0]

  embed_dim = 128
  latent_dim = 512
  num_layers = 2
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
  print(model.tabulate(jax.random.key(0), obs, compute_flops=True))

  model.init(jax.random.PRNGKey(0), obs)
