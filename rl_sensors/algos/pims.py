import copy
from typing import *
import numpy as np
from motpy.rfs.bernoulli import MultiBernoulli
from motpy.models.transition.constant_velocity import ConstantVelocity
from motpy.distributions.gaussian import Gaussian
from rl_sensors.envs.graph_search_track import GraphSearchTrackEnv
import gymnasium as gym
import functools
from motpy.rfs.tomb import TOMBP

from rl_sensors.envs.util import merge_poisson

# ASSUMPTIONS:
# pd(x) = pd(E[x])
# ps(x) = ps(E[x])
# Objects in FOV generate a noise-free measurement
# No false alarms


def simulate_ideal(env: gym.Env, tracker: TOMBP, actions: np.ndarray):

  for action in actions:
    # Update sensor
    env.update_sensor_state(action)

    # Predict step
    tracker = tracker.predict(
        state_estimator=env.state_estimator,
        dt=env.scenario['dt'],
        ps_model=functools.partial(
            env.ps, scenario=env.scenario, pos_inds=env.pos_inds
        ),
    )
    tracker.poisson, tracker.poisson_metadata = merge_poisson(
        distribution=tracker.poisson,
        metadata=tracker.poisson_metadata,
        source_inds=np.arange(len(tracker.poisson)//2),
        target_inds=np.arange(
            len(tracker.poisson)//2, len(tracker.poisson)
        )
    )

    # Update step
    poisson_pd = env.pd(
        object_state=tracker.poisson.state.mean,
        sensor=env.sensor,
        pos_inds=env.pos_inds,
    )
    tracker.poisson.state.weight *= (1 - poisson_pd)

    if len(tracker.mb) > 0:
      pred_mb_pd = env.pd(
          object_state=tracker.mb.state.mean,
          sensor=env.sensor,
          pos_inds=env.pos_inds,
      )
      updated_mb_inds = np.where(pred_mb_pd > 0)[0]
      measurements = env.state_estimator.measurement_model(
          tracker.mb.state.mean[updated_mb_inds],
          noise=False,
          sensor_pos=env.sensor['position'],
          sensor_vel=env.sensor['velocity'],
      )
      for iz, imb in enumerate(updated_mb_inds):
        tracker.mb.state[imb] = env.state_estimator.update(
            state=tracker.mb[imb].state,
            measurement=measurements[iz],
            sensor_pos=env.sensor['position'],
            sensor_vel=env.sensor['velocity'],
        )

  # Planning score
  search_score = -tracker.poisson.state.weight.sum()
  if len(tracker.mb) == 0:
    track_score = 0
  else:
    track_score = env.track_quality(tracker.mb)
  score = search_score + track_score
  return score


if __name__ == '__main__':
  import time

  seed = 0
  np.random.seed(seed)

  # Configure environment
  env = GraphSearchTrackEnv()
  env.reset(seed=seed)

  # Plan
  A = 1 # Action space dimension
  H = 1 # Planning horizon
  N = 10 # Number of discrete action bins
  start = time.time()
  actions = np.array(
      np.meshgrid(
          *[
              [np.linspace(-1, 1, N) for _ in range(A)]
              for _ in range(H)
          ]
      )
  ).reshape(-1, H, A)
  state = copy.deepcopy(env.tracker)
  for action in actions:
    score = simulate_ideal(env=env, tracker=state, actions=action)
    print(score)

  stop = time.time()
  print(f"Planning took {1e3*(stop - start):.1f} ms")
