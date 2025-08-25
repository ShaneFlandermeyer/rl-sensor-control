import numpy as np
import orbax.checkpoint as ocp
import jax
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from tdmpc2_jax.world_model import WorldModel
from tdmpc2_jax.tdmpc2 import TDMPC2
import gymnasium as gym
import rl_sensors.envs
import os

from rl_sensors.graph_encoder import GraphEncoder
from tdmpc2_jax.networks.mlp import NormedLinear
import yaml


def create_agent(env, config, rng):
  encoder_config = config['encoder']
  model_config = config['world_model']
  tdmpc2_config = config['tdmpc2']

  model_key, encoder_key = jax.random.split(rng, 2)
  encoder_module = nn.Sequential([
      GraphEncoder(
          embed_dim=encoder_config['embed_dim'],
          num_layers=encoder_config['num_layers'],
          num_heads=encoder_config['num_heads'],
          kernel_init=nn.initializers.truncated_normal(0.02),
          dtype=encoder_config['dtype'],
      ),
      NormedLinear(
          model_config['latent_dim'],
          activation=None,
          dtype=encoder_config['dtype'],
      ),
  ]
  )

  dummy_obs, _ = env.reset()
  encoder = TrainState.create(
      apply_fn=encoder_module.apply,
      params=encoder_module.init(encoder_key, dummy_obs)['params'],
      tx=optax.chain(
          optax.zero_nans(),
          optax.clip_by_global_norm(model_config['max_grad_norm']),
          optax.adam(encoder_config['learning_rate']),
      )
  )

  model = WorldModel.create(
      action_dim=np.prod(env.action_space.shape),
      encoder=encoder,
      **model_config,
      key=model_key
  )
  if model.action_dim >= 20:
    tdmpc2_config.mppi_iterations += 2

  agent = TDMPC2.create(world_model=model, **tdmpc2_config)

  return agent


def make_gym_env(env_id, seed):
  # From training driver
  env = gym.make(env_id)
  env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
  env = gym.wrappers.RecordEpisodeStatistics(env)
  env = gym.wrappers.Autoreset(env)
  env.action_space.seed(seed)
  env.observation_space.seed(seed)
  return env


if __name__ == '__main__':
  seed = 0
  rng = jax.random.PRNGKey(seed)
  np.random.seed(seed)
  exp_path = 'outputs/BeamOptimization-v0_s2/2025-08-24_10-25-17'

  # Load config
  config = yaml.safe_load(
      open(os.path.join(exp_path, '.hydra/config.yaml'), 'r')
  )

  # Create eval env
  env = make_gym_env(config['env']['env_id'], seed)

  # Create target agent tree
  agent = create_agent(env=env, config=config, rng=rng)

  with ocp.CheckpointManager(
      f"{os.path.abspath(exp_path)}/checkpoint",
      item_names=('agent', 'global_step')
  ) as mngr:
    print('Checkpoint folder found, restoring from', mngr.latest_step())
    restored = mngr.restore(
        mngr.latest_step(),
        args=ocp.args.Composite(
            agent=ocp.args.StandardRestore(agent),
            global_step=ocp.args.JsonRestore(),
        )
    )
    agent, global_step = restored.agent, restored.global_step

  print(agent)
  print(type(agent))
