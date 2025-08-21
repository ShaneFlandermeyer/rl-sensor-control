import gymnasium as gym

gym.register(
    id='BeamOptimization-v0',
    entry_point='rl_sensors.envs.beam_optimization:BeamOptimizationEnv',
    max_episode_steps=1000,
)
