import functools
from typing import *

import gymnasium as gym
import igraph
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from motpy.distributions.gaussian import Gaussian
from motpy.estimators.kalman.sigma_points import (merwe_scaled_sigma_points,
                                                  merwe_sigma_weights)
from motpy.estimators.kalman.ukf import UnscentedKalmanFilter
from motpy.models.measurement import LinearMeasurementModel
from motpy.models.transition import ConstantVelocity
from motpy.rfs.bernoulli import MultiBernoulli
from motpy.rfs.poisson import Poisson
from motpy.rfs.tomb import TOMBP

from rl_sensors.envs.util import merge_poisson


def symlog(x):
  return np.sign(x) * np.log1p(np.abs(x))


class GraphSearchTrackEnv(gym.Env):
  def __init__(self):
    self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))

    # Search grid config
    self.nx_grid = 8
    self.ny_grid = 8
    self.n_grid = self.nx_grid * self.ny_grid
    self.max_search_nodes = self.n_grid

    # Agent config
    self.max_agent_nodes = 20
    self.top_k_search_update = 4

    # Track config
    self.max_track_history = 3
    self.max_active_tracks = 10
    self.max_track_nodes = self.max_active_tracks * self.max_track_history

    self.max_nodes = self.max_search_nodes + \
        self.max_agent_nodes + self.max_track_nodes
    self.max_edges = (
        # Agent transition
        + 2 * (self.max_agent_nodes - 1)
        # Agent-search update
        + 2 * (self.top_k_search_update * self.max_agent_nodes)
        # Track transition
        + 2 * max((self.max_track_nodes - 1), 0)
        # Track update
        + 2 * self.max_track_nodes
    )
    self.observation_space = gym.spaces.Dict(
        current_agent_node_ind=gym.spaces.Discrete(self.max_nodes),
        edge_features=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_edges, 9),
            dtype=np.float32,
        ),
        edge_list=gym.spaces.MultiDiscrete(
            np.full((self.max_edges, 2), self.max_nodes+1),
            start=np.full((self.max_edges, 2), -1)
        ),
        edge_mask=gym.spaces.MultiBinary(self.max_edges),
        global_features=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, 5),
            dtype=np.float32,
        ),
        node_features=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_nodes, 20),
            dtype=np.float32,
        ),
        node_mask=gym.spaces.MultiBinary(self.max_nodes),
    )

  def reset(
      self,
      seed: Optional[int] = None,
      **kwargs
  ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ###########################
    # Initialize environment
    ###########################
    self.timestep = 0
    if seed is not None:
      self.np_random, seed = gym.utils.seeding.np_random(seed)
      self.action_space.seed(seed)
      self.observation_space.seed(seed)

    self.scenario = dict(
        extents=np.array([
            [-1000, 1000],
            [-1000, 1000]
        ]),
        max_velocity=10,
        birth_rate=1/25,
        clutter_rate=0.0,
        dt=1.0,
        max_trace=50**2,
        num_initiate_detections=3,
    )
    self.sensor = dict(
        position=np.zeros(2),
        velocity=np.zeros(2),
        beamwidth=20*np.pi/180,
        steering_angle=0,
        action=np.zeros(1),
    )

    self.ground_truth = []

    ###########################
    # Initialize tracker
    ###########################
    self.pos_inds = [0, 2]
    self.vel_inds = [1, 3]
    self.transition_model = ConstantVelocity(
        state_dim=4,
        w=1e-1,
        position_inds=self.pos_inds,
        velocity_inds=self.vel_inds,
        noise_type='continuous',
    )
    self.measurement_model = LinearMeasurementModel(
        state_dim=4,
        covar=1*np.eye(2),
        measured_dims=self.pos_inds,
    )
    self.state_estimator = UnscentedKalmanFilter(
        transition_model=self.transition_model,
        measurement_model=self.measurement_model,
        state_subtract_fn=np.subtract,
        state_average_fn=np.average,
        measurement_subtract_fn=np.subtract,
        measurement_average_fn=np.average,
        sigma_params=dict(alpha=1e-3, beta=2, kappa=0),
    )

    # Birth distribution
    xmin, xmax = self.scenario['extents'][0]
    ymin, ymax = self.scenario['extents'][1]
    x = np.linspace(xmin, xmax, self.nx_grid)
    y = np.linspace(ymin, ymax, self.ny_grid)
    dx = 0.5*(xmax - xmin) / self.nx_grid
    dy = 0.5*(ymax - ymin) / self.ny_grid
    dvx = dvy = self.scenario['max_velocity']
    grid_x, grid_y = np.meshgrid(x, y, indexing='ij')
    grid_x, grid_y = grid_x.ravel(), grid_y.ravel()
    grid_vx = grid_vy = np.zeros_like(grid_x)
    birth_means = np.stack([
        grid_x, grid_vx, grid_y, grid_vy
    ], axis=-1)
    birth_covars = np.diag(np.array([dx, dvx, dy, dvy])**2)[None, ...].repeat(
        self.n_grid, axis=0
    )
    birth_distribution = Gaussian(
        mean=birth_means,
        covar=birth_covars,
        weight=np.full(self.n_grid, self.scenario['birth_rate'] / self.n_grid)
    )

    # Initial undetected distribution
    init_wsum = self.np_random.uniform(1, 5)
    undetected_weight = self.np_random.uniform(0, 1, size=self.n_grid)
    undetected_state = Gaussian(
        mean=birth_means,
        covar=birth_covars,
        weight=init_wsum * (undetected_weight / undetected_weight.sum())
    )

    self.tracker = TOMBP(
        poisson=Poisson(state=undetected_state),
        mb=None,
        birth_distribution=birth_distribution
    )
    self.global_track_counter = 0

    self.graph = igraph.Graph(directed=True)
    self.update_graph()

    obs = self.get_obs()
    info = {}

    return obs, info

  def step(
      self,
      action: np.ndarray
  ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
    self.timestep += 1

    # Update simulation
    self.update_sensor_state(action)
    self.update_ground_truth(dt=self.scenario['dt'])
    measurements = self.measure()

    # Predict step
    self.tracker = self.tracker.predict(
        state_estimator=self.state_estimator,
        dt=self.scenario['dt'],
        ps_model=functools.partial(
            self.ps, scenario=self.scenario, pos_inds=self.pos_inds
        ),
    )
    self.tracker.poisson = merge_poisson(
        distribution=self.tracker.poisson,
        source_inds=np.arange(len(self.tracker.poisson)//2),
        target_inds=np.arange(
            len(self.tracker.poisson)//2, len(self.tracker.poisson)
        )
    )

    # Update step
    volume = np.prod(np.diff(self.scenario['extents'], axis=-1).ravel())
    self.tracker = self.tracker.update(
        measurements=measurements,
        state_estimator=self.state_estimator,
        pd_model=functools.partial(
            self.pd, sensor=self.sensor, pos_inds=self.pos_inds
        ),
        lambda_fa=self.scenario['clutter_rate'] / volume,
        pg=0.999,
    )
    if len(self.tracker.mb) > 0:
      self.tracker.mb, self.tracker.mb_metadata = self.tracker.mb.prune(
          valid_fn=lambda mb: np.logical_and(
              mb.r > 1e-4,
              np.linalg.trace(
                  mb.state.covar[
                      np.ix_(
                          np.arange(len(mb)), self.pos_inds, self.pos_inds
                      )
                  ]
              ) < 5*self.scenario['max_trace']
          ),
          meta=self.tracker.mb_metadata,
      )
      # Update track metadata
      track_pd = self.pd(
          object_state=self.tracker.mb.state,
          sensor=self.sensor,
          pos_inds=self.pos_inds,
      )
      for i, meta in enumerate(self.tracker.mb_metadata):
        if meta['new']:
          meta['id'] = f'track_{self.global_track_counter}'
          self.global_track_counter += 1
        new = meta.get('new', False)
        updated = meta.get('updated', False)
        missed = track_pd[i] > 0 and not (updated or new)
        measurement_type = (
            'update' if (updated or new) else
            'miss' if missed else
            'predict'
        )
        num_detections = meta.get('num_detections', 0)
        if measurement_type == 'update':
          num_detections += 1
        initiated = num_detections >= self.scenario['num_initiate_detections']
        self.tracker.mb_metadata[i].update(
            pd=track_pd[i],
            measurement_type=measurement_type,
            num_detections=num_detections,
            initiated=initiated,
        )

    # Env update
    self.update_graph()
    obs = self.get_obs()
    reward = self.get_reward()
    terminated = truncated = False
    info = {}

    return obs, reward, terminated, truncated, info

  def update_graph(self) -> None:
    node_label_map = dict(
        search=[1, 0, 0],
        agent=[0, 1, 0],
        track=[0, 0, 1],
    )
    edge_label_map = dict(
        transition=[1, 0],
        update=[0, 1],
    )
    measurement_label_map = dict(
        predict=[1, 0, 0],
        update=[0, 1, 0],
        miss=[0, 0, 1],
        none=[0, 0, 0],
    )
    #############################
    # Search nodes
    #############################
    if self.timestep == 0:
      search_ids = [
          f'search_{i}' for i in range(len(self.tracker.poisson))
      ]
      self.graph.add_vertices(
          n=len(self.tracker.poisson),
          attributes=dict(
              type='search',
              label=[node_label_map['search']],
              name=search_ids,
              id=search_ids,
              timestep=self.timestep,
              position=self.tracker.poisson.state.mean[:, self.pos_inds],
              velocity=self.tracker.poisson.state.mean[:, self.vel_inds],
              covar_diag=np.sqrt(np.diagonal(
                  self.tracker.poisson.state.covar, axis1=-1, axis2=-2
              )),
              # Search features
              weight=self.tracker.poisson.state.weight,
              # Agent features
              sensor_action=np.zeros(
                  (len(self.tracker.poisson), self.action_space.shape[0])
              ),
              # Track features
              measurement_type='none',
              measurement_label=[measurement_label_map['none']],
              track_quality=np.zeros(len(self.tracker.poisson)),
              existence_probability=np.zeros(len(self.tracker.poisson)),
              initiation_progress=np.zeros(len(self.tracker.poisson)),
          )
      )
    else:
      search_nodes = self.graph.vs(type_eq='search')
      search_nodes['weight'] = self.tracker.poisson.state.weight / \
          self.tracker.poisson.state.weight.max()
      search_nodes['timestep'] = self.timestep

    ##############################
    # Agent nodes
    ##############################
    current_agent_node = dict(
        type='agent',
        label=node_label_map['agent'],
        name=f'agent_t{self.timestep}',
        id='agent',
        timestep=self.timestep,
        position=self.sensor['position'],
        velocity=self.sensor['velocity'],
        covar_diag=np.zeros(4),
        # Search features
        weight=0.0,
        # Agent features
        sensor_action=self.sensor['action'],
        # Track features
        measurement_type='none',
        measurement_label=measurement_label_map['none'],
        track_quality=0.0,
        existence_probability=0.0,
        initiation_progress=0.0,
    )
    agent_edges = []
    agent_edge_attributes = dict(
        type=[],
        label=[],
        pd=[],
        distance=[],
        angle=[],
        relative_position=[],
        relative_velocity=[],
    )
    # Agent transition edge
    if self.timestep > 0:
      last_agent = self.graph.vs(
          type_eq='agent', timestep_eq=self.timestep-1
      )[0]

      agent_edges.extend([
          (last_agent['name'], current_agent_node['name']),
          (current_agent_node['name'], last_agent['name']),
      ])
      agent_edge_attributes.update(
          type=agent_edge_attributes['type'] + ['transition', 'transition'],
          label=agent_edge_attributes['label'] + 2*[
              edge_label_map['transition']
          ],
          pd=agent_edge_attributes['pd'] + [0.0, 0.0],
          distance=agent_edge_attributes['distance'] + [0.0, 0.0],
          angle=agent_edge_attributes['angle'] + [0.0, 0.0],
          relative_position=agent_edge_attributes['relative_position'] + [
              last_agent['position'] - current_agent_node['position'],
              current_agent_node['position'] - last_agent['position'],
          ],
          relative_velocity=agent_edge_attributes['relative_velocity'] + [
              last_agent['velocity'] - current_agent_node['velocity'],
              current_agent_node['velocity'] - last_agent['velocity'],
          ],
      )

    # Search detection edges
    search_nodes = self.graph.vs(type_eq='search')
    search_pos = np.array(search_nodes['position'])
    if self.timestep > 0:
      search_pd = self.pd(
          object_state=self.tracker.poisson.state,
          sensor=self.sensor,
          pos_inds=self.pos_inds,
      )
      detected_search = np.where(search_pd > 0)[0]
      num_search_updates = min(len(detected_search), self.top_k_search_update)
      num_search_edges = 2 * num_search_updates
      if num_search_updates > 0:
        detected_search = detected_search[
            np.argpartition(
                search_pd[detected_search], -num_search_updates
            )[-num_search_updates:]
        ]

        # NOTE: Assumes search nodes have the same ordering as the search grid
        search_edge_pd = search_pd[detected_search].tolist()
        search_edge_dist = np.linalg.norm(
            search_pos[detected_search] - self.sensor['position'], axis=-1
        ).tolist()
        search_edge_angles = (
            np.arctan2(
                search_pos[detected_search, 1] - self.sensor['position'][1],
                search_pos[detected_search, 0] - self.sensor['position'][0]
            ).tolist() +
            np.arctan2(
                self.sensor['position'][1] - search_pos[detected_search, 1],
                self.sensor['position'][0] - search_pos[detected_search, 0]
            ).tolist()
        )

        agent_edges.extend([
            (search_nodes[i]['name'], current_agent_node['name'])
            for i in detected_search
        ] + [
            (current_agent_node['name'], search_nodes[i]['name'])
            for i in detected_search
        ])
        agent_edge_attributes.update(
            type=agent_edge_attributes['type'] + num_search_edges*['update'],
            label=agent_edge_attributes['label'] + num_search_edges*[
                edge_label_map['update']
            ],
            pd=agent_edge_attributes['pd'] + 2*search_edge_pd,
            distance=agent_edge_attributes['distance'] + 2*search_edge_dist,
            angle=agent_edge_attributes['angle'] + search_edge_angles,
            relative_position=agent_edge_attributes['relative_position'] +
            num_search_edges*[np.zeros(2)],
            relative_velocity=agent_edge_attributes['relative_velocity'] +
            num_search_edges*[np.zeros(2)],
        )

    self.graph.add_vertex(**current_agent_node)
    self.graph.add_edges(es=agent_edges, attributes=agent_edge_attributes)
    # Remove old agent nodes
    agent_nodes = self.graph.vs(type_eq='agent')
    if len(agent_nodes) > self.max_agent_nodes:
      num_to_delete = len(agent_nodes) - self.max_agent_nodes
      agent_nodes[:num_to_delete]['delete'] = True
    else:
      agent_nodes['delete'] = False

    #################################
    # Track nodes
    #################################
    if len(self.tracker.mb) > 0 and self.max_active_tracks > 0:
      # Delete stale track nodes
      track_ids = [meta['id'] for meta in self.tracker.mb_metadata]
      stale_track_nodes = self.graph.vs(type_eq='track', id_notin=track_ids)
      if len(stale_track_nodes) > 0:
        self.graph.delete_vertices(stale_track_nodes)

      # Collect track info for this timestep
      num_new_track_nodes = min(len(self.tracker.mb), self.max_active_tracks)
      track_node_attributes = dict(
          type='track',
          label=[node_label_map['track']],
          name=[],
          id=[],
          timestep=self.timestep,
          position=self.tracker.mb.state.mean[
              :self.max_active_tracks, self.pos_inds
          ],
          velocity=self.tracker.mb.state.mean[
              :self.max_active_tracks, self.vel_inds
          ],
          covar_diag=np.sqrt(np.diagonal(
              self.tracker.mb.state.covar[:self.max_active_tracks],
              axis1=-1, axis2=-2
          )),
          # Search features
          weight=np.zeros(num_new_track_nodes),
          # Agent features
          sensor_action=np.zeros((num_new_track_nodes, 1)),
          # Track features
          measurement_type=[],
          measurement_label=[],
          track_quality=[],
          existence_probability=self.tracker.mb.r[:num_new_track_nodes],
          initiation_progress=[],
      )
      track_edge_attributes = dict(
          type=[],
          label=[],
          pd=[],
          distance=[],
          angle=[],
          relative_position=[],
          relative_velocity=[],
      )
      track_edges = []
      for i, meta in enumerate(self.tracker.mb_metadata):
        if i >= self.max_active_tracks:  # At track capacity
          track_history['delete'] = True
          continue

        # Update node attributes
        track_id = meta['id']
        track_node_name = f"{track_id}_t{self.timestep}"
        track_measurement_type = meta['measurement_type']
        track_initiation_progress = np.clip(
            meta['num_detections'] / self.scenario['num_initiate_detections'],
            0, 1
        )

        track_covar = self.tracker.mb.state.covar[i]
        track_trace = np.linalg.trace(
            track_covar[np.ix_(self.pos_inds, self.pos_inds)]
        )
        track_quality = 1 - np.clip(
            track_trace / self.scenario['max_trace'], 0, 1
        )
        track_node_attributes.update(
            id=track_node_attributes['id'] + [track_id],
            name=track_node_attributes['name'] + [track_node_name],
            measurement_type=track_node_attributes['measurement_type'] +
            [track_measurement_type],
            measurement_label=track_node_attributes['measurement_label'] +
            [measurement_label_map[track_measurement_type]],
            track_quality=track_node_attributes['track_quality'] +
            [track_quality],
            initiation_progress=track_node_attributes['initiation_progress'] +
            [track_initiation_progress],
        )

        # Transition edge
        track_history = self.graph.vs(type_eq='track', id_eq=track_id)
        track_updates = track_history(measurement_type_eq='update')
        if len(track_updates) > 0:
          last_update = track_updates[-1]
          track_pos = track_node_attributes['position'][i]
          track_vel = track_node_attributes['velocity'][i]

          track_edges.extend([
              (last_update['name'], track_node_name),
              (track_node_name, last_update['name'])
          ])
          track_edge_attributes.update(
              type=track_edge_attributes['type'] +
              ['transition', 'transition'],
              label=track_edge_attributes['label'] + 2*[
                  edge_label_map['transition']
              ],
              pd=track_edge_attributes['pd'] + [0.0, 0.0],
              distance=track_edge_attributes['distance'] + [0.0, 0.0],
              angle=track_edge_attributes['angle'] + [0.0, 0.0],
              relative_position=track_edge_attributes['relative_position'] + [
                  last_update['position'] - track_pos,
                  track_pos - last_update['position'],
              ],
              relative_velocity=track_edge_attributes['relative_velocity'] + [
                  last_update['velocity'] - track_vel,
                  track_vel - last_update['velocity'],
              ],
          )

        # Measurement update/miss edge
        track_pd = self.tracker.mb_metadata[i]['pd']
        track_pos = self.tracker.mb.state.mean[i, self.pos_inds]
        track_update_dist = np.linalg.norm(track_pos - self.sensor['position'])
        track_update_angles = [
            np.arctan2(
                track_pos[1] - self.sensor['position'][1],
                track_pos[0] - self.sensor['position'][0]
            ),
            np.arctan2(
                self.sensor['position'][1] - track_pos[1],
                self.sensor['position'][0] - track_pos[0]
            ),
        ]
        track_edges.extend([
            (track_node_name, current_agent_node['name']),
            (current_agent_node['name'], track_node_name),
        ])
        track_edge_attributes.update(
            type=track_edge_attributes['type'] + 2*['update'],
            label=track_edge_attributes['label'] + 2*[
                edge_label_map['update']
            ],
            pd=track_edge_attributes['pd'] + 2*[track_pd],
            distance=track_edge_attributes['distance'] + 2*[track_update_dist],
            angle=track_edge_attributes['angle'] + track_update_angles,
            relative_position=track_edge_attributes['relative_position'] +
            2*[np.zeros(2)],
            relative_velocity=track_edge_attributes['relative_velocity'] +
            2*[np.zeros(2)],
        )

        # Track graph pruning
        old_predicts = track_history(measurement_type_eq='predict')
        if len(old_predicts) > 0:
          old_predicts['delete'] = True
        if track_measurement_type == 'update':
          old_misses = track_history(measurement_type_eq='miss')
          if len(old_misses) > 0:
            old_misses['delete'] = True
        if len(track_history) >= self.max_track_history:
          track_history[:-(self.max_track_history-1)]['delete'] = True

      # Add track nodes and edges
      self.graph.add_vertices(
          n=num_new_track_nodes, attributes=track_node_attributes
      )
      self.graph.add_edges(es=track_edges, attributes=track_edge_attributes)
    else:
      self.graph.vs(type_eq='track')['delete'] = True

    ###############################
    # Global graph update
    ###############################
    deleted_nodes = self.graph.vs(delete_eq=True)
    if len(deleted_nodes) > 0:
      self.graph.delete_vertices(deleted_nodes)
    self.graph.vs['age'] = self.timestep - np.array(self.graph.vs['timestep'])

  def get_obs(self) -> Dict[str, Any]:
    ###########################
    # Nodes
    ###########################

    nodes = self.graph.vs
    node_keys = [
        'label',
        'age',
        'position',
        'velocity',
        'covar_diag',
        # Search features
        'weight',
        # Agent features
        'sensor_action',
        # Track features
        'measurement_label',
        'track_quality',
        'existence_probability',
        'initiation_progress',
    ]
    node_features = np.concatenate(
        [
            np.array(nodes[k]).reshape((len(nodes), -1)) for k in node_keys
        ], axis=-1
    )

    # Pad nodes
    node_features = np.pad(
        node_features,
        ((0, self.max_nodes - len(nodes)), (0, 0)),
        mode='constant', constant_values=0,
    )
    node_mask = np.arange(self.max_nodes) < len(nodes)
    current_agent_node_ind = nodes(
        type_eq='agent', timestep_eq=self.timestep
    )[0].index

    ###########################
    # Edges
    ###########################
    edges = self.graph.es
    if len(edges) > 0:
      edge_keys = [
          'label',
          # Measurement features
          'pd',
          'distance',
          'angle',
          # Transition features
          'relative_position',
          'relative_velocity',
      ]
      edge_features = np.concatenate(
          [
              np.array(edges[k]).reshape((len(edges), -1)) for k in edge_keys
          ], axis=-1
      )
      edge_list = np.array(self.graph.get_edgelist())
      # Pad edges
      edge_features = np.pad(
          edge_features,
          ((0, self.max_edges - len(edges)), (0, 0)),
          mode='constant', constant_values=0,
      )
      edge_list = np.pad(
          edge_list,
          ((0, self.max_edges - len(edges)), (0, 0)),
          mode='constant', constant_values=-1,
      )
      edge_mask = np.arange(self.max_edges) < len(edges)
    else:
      edge_features = np.zeros(
          self.observation_space['edge_features'].shape,
          dtype=self.observation_space['edge_features'].dtype
      )
      edge_list = np.zeros(
          self.observation_space['edge_list'].shape,
          dtype=self.observation_space['edge_list'].dtype
      )
      edge_mask = np.zeros(
          self.observation_space['edge_mask'].shape,
          dtype=self.observation_space['edge_mask'].dtype
      )

    ###########################
    # Global features
    ###########################
    # Search state
    w_sum = np.sum(self.tracker.poisson.state.weight, keepdims=True)

    # Track state
    if len(self.tracker.mb) > 0:
      initiated = np.array([
          meta['initiated'] for meta in self.tracker.mb_metadata
      ])
      num_active_tracks = np.count_nonzero(initiated)
      if num_active_tracks > 0:
        valid_mb = self.tracker.mb[initiated]
        valid_covars = valid_mb.state.covar[
            np.ix_(np.arange(len(valid_mb)), self.pos_inds, self.pos_inds)
        ]
        traces = np.linalg.trace(valid_covars)
        track_qualities = 1 - (traces / self.scenario['max_trace']).clip(0, 1)
      else:
        track_qualities = np.zeros(1)
    else:
      num_active_tracks = 0
      track_qualities = np.zeros(1)

    global_features = np.stack([
        w_sum,
        np.array([num_active_tracks]),
        np.array([track_qualities.mean()]),
        np.array([track_qualities.min()]),
        np.array([track_qualities.max()]),
    ], axis=-1)

    obs = dict(
        current_agent_node_ind=current_agent_node_ind,
        edge_features=symlog(edge_features).astype(np.float32),
        edge_list=edge_list,
        edge_mask=edge_mask,
        global_features=symlog(global_features).astype(np.float32),
        node_features=symlog(node_features).astype(np.float32),
        node_mask=node_mask,
    )
    return obs

  def get_reward(self) -> float:
    w = self.tracker.poisson.state.weight
    search_reward = -w.sum()

    initiated = np.array([
        meta['initiated'] for meta in self.tracker.mb_metadata
    ])
    num_confirmed = np.count_nonzero(initiated)

    # Search-only case
    if num_confirmed == 0:
      return search_reward

    # Search-and-track case
    mb = self.tracker.mb[initiated]
    covars = mb.state.covar[
        np.ix_(np.arange(len(mb)), self.pos_inds, self.pos_inds)
    ]
    traces = np.linalg.trace(covars)
    track_errors = traces / self.scenario['max_trace']
    track_reward = (1 - track_errors.clip(0, 1)).sum()
    return search_reward + track_reward

  def update_sensor_state(self, action: np.ndarray) -> None:
    self.sensor['steering_angle'] = action[0] * np.pi
    self.sensor['action'] = action

  def update_ground_truth(self, dt: float) -> None:
    if len(self.ground_truth) > 0:
      states = np.atleast_2d([x[-1] for x in self.ground_truth])
      next_states = self.transition_model(
          states, dt=dt, noise=True, rng=self.np_random
      )

      # Object survival
      ps = self.ps(
          object_state=next_states,
          scenario=self.scenario,
          pos_inds=self.pos_inds,
      )
      survived = ps > 0  # Only actually die outside scenario region
      for i, path in enumerate(self.ground_truth.copy()):
        if survived[i]:
          path.append(next_states[i])
        else:
          self.ground_truth.remove(path)

    # New object birth
    num_birth = self.np_random.poisson(lam=self.scenario['birth_rate'] * dt)
    if num_birth > 0:
      birth_distribution = self.tracker.birth_distribution
      inds = self.np_random.choice(
          a=np.arange(birth_distribution.shape[0]),
          size=num_birth,
          p=birth_distribution.weight / np.sum(birth_distribution.weight)
      )
      new_states = birth_distribution[inds].sample(
          num_points=1, rng=self.np_random, max_distance=1,
      )
      # Clip to scenario extents
      new_states[..., self.pos_inds] = np.clip(
          new_states[..., self.pos_inds],
          self.scenario['extents'][:, 0],
          self.scenario['extents'][:, 1]
      )
      self.ground_truth.extend([list(state) for state in new_states])

  def measure(self) -> List[np.ndarray]:
    if len(self.ground_truth) == 0:
      return []

    # Measure
    states = np.array([path[-1] for path in self.ground_truth])
    Z = self.measurement_model(states, noise=True, rng=self.np_random)

    # Only keep detected measurements
    pd = self.pd(states, sensor=self.sensor, pos_inds=self.pos_inds)
    detected = self.np_random.uniform(size=len(self.ground_truth)) < pd
    Z = Z[detected]

    return Z

  @staticmethod
  def pd(
      object_state: Union[np.ndarray, Gaussian],
      sensor: Dict[str, Any],
      pos_inds: List[int],
  ) -> np.ndarray:
    if isinstance(object_state, Gaussian):
      n = 2
      alpha, beta, kappa = 0.5, 2, 0
      x = merwe_scaled_sigma_points(
          x=object_state.mean[:, pos_inds],
          P=object_state.covar[
              np.ix_(np.arange(object_state.shape[0]), pos_inds, pos_inds)
          ],
          alpha=alpha,
          beta=beta,
          kappa=kappa
      )
      weights = merwe_sigma_weights(
          ndim_state=n, alpha=alpha, beta=beta, kappa=kappa)[0]
      weights = abs(weights) / abs(weights).sum()
    else:
      x = object_state[..., None, pos_inds]
      weights = np.ones(1)

    # Detect objects within beam
    sensor_pos = sensor['position']
    beamwidth = sensor['beamwidth']
    steering_angle = sensor['steering_angle']
    az = np.arctan2(
        x[..., 1] - sensor_pos[1],
        x[..., 0] - sensor_pos[0]
    )
    angle_diff = np.mod((az - steering_angle) + np.pi, 2*np.pi) - np.pi
    in_region = abs(angle_diff) <= beamwidth/2
    pd = np.where(in_region, 0.9, 0)
    return np.average(pd, weights=weights, axis=-1)

  @staticmethod
  def ps(
      object_state: Union[np.ndarray, Gaussian],
      scenario: Dict[str, Any],
      pos_inds: List[int],
  ) -> np.ndarray:
    if isinstance(object_state, Gaussian):
      x = object_state.mean[:, pos_inds]
      weights = np.ones(1)
    else:
      x = object_state[:, pos_inds]
      weights = np.ones(1)
    in_region = np.logical_and.reduce([
        x[:, 0] >= scenario['extents'][0][0],
        x[:, 0] <= scenario['extents'][0][1],
        x[:, 1] >= scenario['extents'][1][0],
        x[:, 1] <= scenario['extents'][1][1],
    ])
    ps = np.where(in_region, 0.99, 0)[:, None]

    return np.average(ps, weights=weights, axis=-1)

  def render(self, graph: igraph.Graph = None):
    if graph is None:
      graph = self.graph

    plt.clf()

    ##################################
    # Scenario visualization
    ##################################
    # Plot sensor
    plt.plot(*self.sensor['position'], 'k.')
    # Plot a transparent blue semicircle with angle beamwidth centered at the steering angle. Only plot that wedge/arc, not the full circle
    beamwidth = self.sensor['beamwidth']
    steering_angle = self.sensor['steering_angle']
    plt.gca().add_patch(matplotlib.patches.Wedge(
        center=self.sensor['position'],
        r=np.max(np.linalg.norm(self.scenario['extents'], axis=-1)),
        theta1=np.degrees(steering_angle - beamwidth/2),
        theta2=np.degrees(steering_angle + beamwidth/2),
        color='r',
        alpha=0.5,
    ))

    ##################################
    # Graph visualization
    ##################################
    # Agent nodes
    agent_nodes = graph.vs(type_eq='agent')
    if len(agent_nodes) > 0:
      agent_pos = np.array(agent_nodes['position'])
      plt.scatter(agent_pos[:, 0], agent_pos[:, 1], c='r')

    # Search nodes
    search_nodes = graph.vs(type_eq='search')
    nx, ny = 20, 20
    search_grid = np.meshgrid(
        np.linspace(*self.scenario['extents'][0], nx),
        np.linspace(*self.scenario['extents'][1], ny)
    )
    search_grid = np.stack([search_grid[0].ravel(), search_grid[1].ravel()]).T
    # Plot gaussian mixture (likelihood) as an image
    mixture = np.zeros((nx, ny))
    norm_weights = (
        search_nodes['weight'] /
        (np.max(search_nodes['weight']) + 1e-10)
    )
    for i in range(len(search_nodes)):
      mixture += (
          search_nodes[i]['weight'] *
          scipy.stats.multivariate_normal.pdf(
              search_grid,
              mean=search_nodes[i]['position'],
              cov=search_nodes[i]['covar_diag'][self.pos_inds]**2
          ).reshape((nx, ny))
      )
      # Print search weight on the plot
      plt.text(
          search_nodes[i]['position'][0],
          search_nodes[i]['position'][1],
          f"{norm_weights[i]:.2f}",
          fontsize=8,
          color='white',
          ha='center',
          va='center',
      )
    plt.imshow(mixture, extent=self.scenario['extents'].ravel(
    ), origin='lower', aspect='auto')
    plt.colorbar()

    # Track nodes
    track_nodes = graph.vs(type_eq='track')
    if len(track_nodes) > 0:
      track_pos = np.array(track_nodes['position'])
      color = [
          'green' if track_nodes[i]['initiation_progress'] == 1 else 'orange'
          for i in range(len(track_nodes))
      ]
      plt.scatter(track_pos[:, 0], track_pos[:, 1], c=color, s=50)
      # Print track quality
      for i, pos in enumerate(track_pos):
        plt.text(
            pos[0], pos[1],
            f"{track_nodes[i]['track_quality']:.2f}",
            fontsize=8,
            color='white',
            ha='center',
            va='center',
        )

    # Edges
    if len(graph.es) > 0:
      source_pos = np.array(
          [graph.vs[e.source]['position'] for e in graph.es]
      )
      target_pos = np.array(
          [graph.vs[e.target]['position'] for e in graph.es]
      )
      edge_x = np.vstack([source_pos[:, 0], target_pos[:, 0]])
      edge_y = np.vstack([source_pos[:, 1], target_pos[:, 1]])

      plt.plot(edge_x, edge_y, color='black', linewidth=0.5)

    # Ground truth
    for path in self.ground_truth:
      # Plot the entire path as a dotted red line
      path = np.array(path)
      plt.plot(path[:, 0], path[:, 2], 'r--')
      p = np.array(path[-1])
      plt.plot(p[0], p[2], '*', color='red')

    plt.xlim(self.scenario['extents'][0])
    plt.ylim(self.scenario['extents'][1])
    plt.pause(0.01)
    plt.draw()
