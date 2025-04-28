from rl_sensors.envs.graph_search import GraphSearchEnv
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
  env = GraphSearchEnv()
  
  obs, _ = env.reset(seed=0)
  for i in range(500):
    obs, reward, _, _, _ = env.step(env.action_space.sample())
    env.render()
    print(reward)
    plt.pause(0.01)
    
  
  env.render()
  plt.show()