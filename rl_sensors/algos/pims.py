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


def simulate_ideal(env: gym.Env, tracker: TOMBP, action_seq: np.ndarray):
  # ASSUMPTIONS:
  # pd(x) = pd(E[x])
  # ps(x) = ps(E[x])
  # Objects in FOV generate a noise-free measurement
  # No false alarms

  for action in action_seq:
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
    track_score = env.track_quality(tracker.mb).sum()
  score = search_score + track_score
  return score


def plan_pims(
    env: gym.Env,
    N: int = 10,
    H: int = 1,
    r_th: float = 0.1
) -> np.ndarray:
  # Create action array
  A = np.prod(env.action_space.shape)
  actions = np.array(
      np.meshgrid(
          *[
              [np.linspace(-1, 1, N) for _ in range(A)]
              for _ in range(H)
          ]
      )
  ).reshape(-1, H, A)

  # 
  state = copy.deepcopy(env.tracker)
  if len(state.mb) > 0:
    state.mb, _ = state.mb.prune(valid_fn=lambda mb: mb.r > r_th)

  # Plan
  scores = np.zeros(actions.shape[0])
  for i, action in enumerate(actions):
    scores[i] = simulate_ideal(env=env, tracker=state, action_seq=action)
  best_score, best_action = np.max(scores), actions[np.argmax(scores), 0, :]
  return best_score, best_action


if __name__ == '__main__':
  import time

  seed = 0
  np.random.seed(seed)

  # Configure environment
  env = GraphSearchTrackEnv()
  env.reset(seed=seed)
 
  for _ in range(5):
    total_r = 0
    for i in range(1000):
      # env.render()
      # print(i)
      score, a_star = plan_pims(env=env, N=30, H=1, r_th=1e-2)
      obs, reward, term, trunc, info = env.step(a_star)
      total_r += reward
    print(total_r)
    env.reset()
