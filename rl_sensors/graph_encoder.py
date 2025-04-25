
from typing import *

from einops import rearrange
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
from flax_gnn.layers.attention import PMA, AttentionBlock
from flax_gnn.layers.gatv2 import GATv2
from bmpc_jax.common.activations import mish, simnorm
from tdmpc2_jax.networks.mlp import NormedLinear
from rl_rrm.envs.radar2d import Radar2DEnv
from rl_rrm.networks.simba import Simba
from functools import partial
import numpy as np


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
    edge_mask = input['edge_mask']
    edge_list = input['edge_list']
    global_features = input['global_features']
    node_features = input['node_features']
    node_mask = input['node_mask']

    # Handle masked edges
    batch_dims = node_features.shape[:-2]
    pad_node = jnp.zeros((*batch_dims, 1, node_features.shape[-1]))
    pad_mask = jnp.zeros((*batch_dims, 1), dtype=bool)
    node_features = jnp.concatenate([node_features, pad_node], axis=-2)
    node_mask = jnp.concatenate([node_mask, pad_mask], axis=-1)
    senders = jnp.where(edge_mask, edge_list[..., 0], -1)
    receivers = jnp.where(edge_mask, edge_list[..., 1], -1)

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
    graph['node_features'] = nn.Sequential([
        nn.Dense(self.embed_dim, kernel_init=self.kernel_init),
        nn.RMSNorm(),
        mish,
        nn.Dense(self.embed_dim, kernel_init=self.kernel_init),
        nn.RMSNorm(),
    ])(graph['node_features'])

    for i in range(self.num_layers):
      # Graph update
      skip = nn.Dense(
          self.embed_dim, kernel_init=self.kernel_init
      )(graph['node_features'])
      graph = GATv2(
          embed_dim=self.embed_dim,
          num_heads=self.num_heads,
          share_weights=False,
          add_self_edges=False,
          kernel_init=self.kernel_init,
      )(**graph)
      graph['node_features'] = mish(
          nn.RMSNorm()(graph['node_features'] + skip)
      )

      # Global pooling
      graph['global_features'] = PMA(
          num_seeds=1,
          seed_init=self.kernel_init,
          attention_base=AttentionBlock(
              embed_dim=self.embed_dim,
              hidden_dim=self.embed_dim,
              num_heads=self.num_heads,
              norm_qk=False,
              use_ffn=False,
              kernel_init=self.kernel_init,
          ),
      )(x=graph['node_features'], valid=node_mask)
      if input['global_features'] is not None:
        graph['global_features'] = jnp.concatenate(
            [graph['global_features'], input['global_features']], axis=-1
        )

    # Post-processing
    x = rearrange(graph['global_features'], '... n d -> ... (n d)')

    return x


if __name__ == '__main__':
  env = gym.vector.SyncVectorEnv([lambda: Radar2DEnv() for _ in range(1)])
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
      NormedLinear(latent_dim, activation=nn.relu),
  ])
  print(model.tabulate(jax.random.key(0), obs, compute_flops=True))

  model.init(jax.random.PRNGKey(0), obs)
