from rl_sensors.envs.graph_search import GraphSearchEnv
import matplotlib.pyplot as plt
import numpy as np

def greedy_search(obs):
  angles = obs['node_features'][..., 0]
  weights = obs['node_features'][..., 1]
  
  # Select the angle with the highest weight
  max_index = np.argmax(weights)
  action = np.array([angles[max_index]])
  return action

if __name__ == '__main__':
  env = GraphSearchEnv()
  
  obs, _ = env.reset(seed=0)
  for i in range(500):
    # action = greedy_search(obs)
    action = env.action_space.sample()
    obs, reward, _, _, _ = env.step(action)
    env.render()
    print(reward)
    plt.pause(0.01)
    
  
  env.render()
  plt.show()