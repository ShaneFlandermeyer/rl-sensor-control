from rl_sensors.envs.graph_search import GraphSearchEnv
from rl_sensors.envs.graph_search_track import GraphSearchTrackEnv
import matplotlib.pyplot as plt
import numpy as np


def greedy_search(obs):
  # angles = obs['node_features'][..., -2]
  pos = obs['node_features'][..., 4:6]
  angles = np.arctan2(pos[..., 1], pos[..., 0])
  weights = obs['node_features'][..., 8]

  # Select the angle with the highest weight
  max_index = np.argmax(weights)
  action = np.array([angles[max_index]])/np.pi + np.random.normal(0.00, 0.00)
  return action


if __name__ == '__main__':
  env = GraphSearchTrackEnv()

  seed = 0
  obs, _ = env.reset(seed=seed)
  np.random.seed(seed)
  total_r = 0
  for i in range(1000):
    # env.render()
    action = greedy_search(obs)
    # action = env.action_space.sample()
    obs, reward, _, _, _ = env.step(action)

    total_r += reward
    if (i+1) % 1000 == 0:
      print("Total reward:", total_r)
      total_r = 0
      obs, _ = env.reset()

  # env.render()
  # plt.show()
