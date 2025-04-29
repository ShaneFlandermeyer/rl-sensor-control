
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
    edge_features = None  # input['edge_features']
    # edge_mask = input['edge_mask']
    edge_list = input['edge_list']
    global_features = input['global_features']
    node_features = input['node_features']
    # node_mask = input['node_mask']

    # Handle masked edges
    # batch_dims = node_features.shape[:-2]
    # pad_node = jnp.zeros((*batch_dims, 1, node_features.shape[-1]))
    # pad_mask = jnp.zeros((*batch_dims, 1), dtype=bool)
    # node_features = jnp.concatenate([node_features, pad_node], axis=-2)
    # node_mask = jnp.concatenate([node_mask, pad_mask], axis=-1)
    # senders = jnp.where(edge_mask, edge_list[..., 0], -1)
    # receivers = jnp.where(edge_mask, edge_list[..., 1], -1)
    senders, receivers = edge_list[..., 0], edge_list[..., 1]

    ######################
    # Graph Processing
    ######################
    graph = dict(
        node_features=node_features,
        edge_features=edge_features,
        senders=senders,
        receivers=receivers,
        global_features=global_features,
    )
    # Pre-process node features with an MLP
    graph['node_features'] = nn.Sequential([
        nn.Dense(self.embed_dim, kernel_init=self.kernel_init),
        nn.RMSNorm(),
        nn.relu,
        nn.Dense(self.embed_dim, kernel_init=self.kernel_init),
        nn.RMSNorm(),
        nn.relu,
    ])(graph['node_features'])

    for i in range(self.num_layers):
      # Graph update
      skip = nn.Dense(
          self.embed_dim, kernel_init=self.kernel_init
      )(graph['node_features'])
      graph = GIN(
          mlp=nn.Sequential([
              nn.Dense(self.embed_dim, kernel_init=self.kernel_init),
              nn.RMSNorm(),
              nn.relu,
              nn.Dense(self.embed_dim, kernel_init=self.kernel_init),
          ]),
          epsilon=0.0,
          kernel_init=self.kernel_init,
      )(**graph)
      graph['node_features'] = nn.relu(
          nn.RMSNorm()(graph['node_features'] + skip)
      )

    # Global pooling
    graph['global_features'] = PMA(
        num_seeds=1,
        seed_init=nn.initializers.xavier_normal(),
        attention_base=AttentionBlock(
            embed_dim=self.embed_dim,
            hidden_dim=self.embed_dim,
            num_heads=self.num_heads,
            norm_qk=False,
            use_ffn=False,
            kernel_init=self.kernel_init,
        ),
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
