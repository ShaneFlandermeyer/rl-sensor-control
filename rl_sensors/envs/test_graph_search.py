from rl_sensors.envs.graph_search import GraphSearchEnv
import matplotlib.pyplot as plt
import numpy as np

def greedy_search(obs):
  # angles = obs['node_features'][..., -2]
  pos = obs['node_features'][..., :2]
  angles = np.arctan2(pos[..., 1], pos[..., 0])
  weights = obs['node_features'][..., 2]
  
  # Select the angle with the highest weight
  max_index = np.argmax(weights)
  action = np.array([angles[max_index]])/np.pi + np.random.uniform(-0.01, 0.01)
  return action

if __name__ == '__main__':
  env = GraphSearchEnv()
  
  obs, _ = env.reset(seed=None)
  total_r = 0
  for i in range(500):
    action = greedy_search(obs)
    # action = env.action_space.sample()
    obs, reward, _, _, _ = env.step(action)
    # env.render()
    # plt.show()
    print(reward)
    total_r += reward
  print("Total reward:", total_r)
  
    
  
  # env.render()
  # plt.show()