import copy
from typing import *
import numpy as np
from rl_sensors.envs.graph_search_track import GraphSearchTrackEnv
import gymnasium as gym
import functools
from motpy.rfs.tomb import TOMBP
import tqdm


def simulate_ideal(
    env: gym.Env,
    tracker: TOMBP,
    action_seq: np.ndarray,
    rng: np.random.RandomState
):
  # ASSUMPTIONS:
  # pd(x) = pd(E[x])
  # ps(x) = ps(E[x])
  # Objects in FOV generate a noise-free measurement
  # No false alarms

  cost = 0
  for action in action_seq:
    # Update sensor
    env.update_sensor_state(action)

    # Predict step
    tracker = tracker.predict(
        state_estimator=env.state_estimator,
        dt=env.scenario['dt'],
        ps_model=functools.partial(
            env.ps,
            scenario=env.scenario,
            pos_inds=env.pos_inds,
            rng=rng,
        ),
    )
    # Update step
    poisson_pd = env.pd(
        object_state=tracker.poisson.state.mean,
        sensor=env.sensor,
        pos_inds=env.pos_inds,
        rng=rng,
    )
    tracker.poisson.state.weight *= (1 - poisson_pd)

    if len(tracker.mb) > 0:
      pred_mb_pd = env.pd(
          object_state=tracker.mb.state.mean,
          sensor=env.sensor,
          pos_inds=env.pos_inds,
          rng=rng,
      )
      updated_mb_inds = np.where(pred_mb_pd > 0)[0]
      measurements = env.state_estimator.measurement_model(
          tracker.mb.state.mean[updated_mb_inds],
          noise=False,
          sensor_pos=env.sensor['position'],
          sensor_vel=env.sensor['velocity'],
          rng=rng,
      )
      for iz, imb in enumerate(updated_mb_inds):
        tracker.mb.state[imb] = env.state_estimator.update(
            state=tracker.mb[imb].state,
            measurement=measurements[iz],
            sensor_pos=env.sensor['position'],
            sensor_vel=env.sensor['velocity'],
        )

    # Planning score
    search_cost = tracker.poisson.state.weight.sum()
    if len(tracker.mb) == 0:
      track_cost = 0
    else:
      covars = tracker.mb.state.covar[
          np.ix_(np.arange(len(tracker.mb)), [0, 2], [0, 2])
      ]
      track_cost = np.linalg.trace(covars).sum()
    c = np.sqrt(env.scenario['max_trace'])
    eta = c**2 / 2
    cost += track_cost + eta * search_cost
  return cost


def plan_pims(
    env: gym.Env,
    N_bins: int,
    H: int,
    r_th: float,
    rng: np.random.RandomState,
) -> np.ndarray:
  # Enumerate over all possible action sequences
  action_dim = len(N_bins)
  actions = np.array(
      np.meshgrid(*[np.linspace(-1, 1, nb) for nb in N_bins] * H)
  ).T.reshape(-1, H, action_dim)

  state = copy.deepcopy(env.tracker)
  if len(state.mb) > 0:
    state.mb, _ = state.mb.prune(valid_fn=lambda mb: mb.r > r_th)

  # Plan
  costs = np.zeros(actions.shape[0])
  for i, action_seq in enumerate(actions):
    costs[i] = simulate_ideal(
        env=env, tracker=state, action_seq=action_seq, rng=rng
    )
  best_cost, best_action = np.min(costs), actions[np.argmin(costs), 0, :]
  return best_cost, best_action


if __name__ == '__main__':
  import time

  seed = 0
  rng = np.random.RandomState(seed)

  # Configure environment
  env = GraphSearchTrackEnv()
  env.reset(seed=seed)

  num_ep = 10
  num_steps = 1000
  mean_r = 0
  
  for iep in range(num_ep):
    total_r = 0
    env.reset()
    pbar = tqdm.tqdm(initial=0, total=num_steps)
    for i in range(num_steps):
      cost, action = plan_pims(
          env=env, N_bins=[20, 5], H=1, r_th=0.05, rng=rng
      )
      obs, reward, term, trunc, info = env.step(action)
      total_r += reward

      pbar.update()
    pbar.close()

    print(iep, total_r)
    # Update moving average
    mean_r = 1 / (iep + 1) * total_r + iep / (iep + 1) * mean_r
    print(mean_r)
