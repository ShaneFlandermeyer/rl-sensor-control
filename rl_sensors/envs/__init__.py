import gymnasium as gym

gym.register(
    id='GraphSearch-v0',
    entry_point='rl_sensors.envs.graph_search:GraphSearchEnv',
    max_episode_steps=500,
)
