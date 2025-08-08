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
from motpy.models.measurement import LinearMeasurementModel, Radar2D
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
            shape=(1, 3),
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
        birth_rate=1/50,
        clutter_rate=0,
        dt=1.0,
        max_trace=50**2,
        pg=0.999,
        # Pruning
        r_prune=1e-4,
        trace_prune=5*(50**2),
        # Initiation
        num_initiate_detections=3,
        min_active_r=1e-2,
        min_active_quality=1e-2,
    )
    self.sensor = dict(
        position=np.zeros(2),
        velocity=np.zeros(2),
        max_range=1500,
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
    self.measurement_model = Radar2D(
        covar=np.diag([10, 1*np.pi/180, 1])**2,
        pos_inds=self.pos_inds,
        vel_inds=self.vel_inds,
    )
    self.state_estimator = UnscentedKalmanFilter(
        transition_model=self.transition_model,
        measurement_model=self.measurement_model,
        state_subtract_fn=np.subtract,
        state_average_fn=np.average,
        measurement_subtract_fn=Radar2D.subtract_fn,
        measurement_average_fn=Radar2D.average_fn,
        sigma_params=dict(alpha=1e-3, beta=2, kappa=0),
    )

    # Birth distribution
    xmin, xmax = self.scenario['extents'][0]
    ymin, ymax = self.scenario['extents'][1]
    dx = 0.5*(xmax - xmin) / self.nx_grid
    dy = 0.5*(ymax - ymin) / self.ny_grid
    x = np.linspace(xmin+dx, xmax-dx, self.nx_grid)
    y = np.linspace(ymin+dy, ymax-dy, self.ny_grid)
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
    undetected_weight = self.np_random.uniform(0, 1, size=self.n_grid)
    undetected_state = Gaussian(
        mean=birth_means,
        covar=birth_covars,
        weight=undetected_weight / undetected_weight.sum()
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
    self.tracker.poisson, self.tracker.poisson_metadata = merge_poisson(
        distribution=self.tracker.poisson,
        metadata=self.tracker.poisson_metadata,
        source_inds=np.arange(len(self.tracker.poisson)//2),
        target_inds=np.arange(
            len(self.tracker.poisson)//2, len(self.tracker.poisson)
        )
    )

    # Update step
    volume = (self.sensor['beamwidth'] / (2*np.pi)) * \
        (np.pi * self.sensor['max_range']**2)
    self.tracker = self.tracker.update(
        measurements=measurements,
        state_estimator=self.state_estimator,
        pd_model=functools.partial(
            self.pd, sensor=self.sensor, pos_inds=self.pos_inds
        ),
        lambda_fa=self.scenario['clutter_rate'] / volume,
        pg=self.scenario['pg'],
        sensor_pos=self.sensor['position'],
        sensor_vel=self.sensor['velocity'],
    )
    if len(self.tracker.mb) > 0:
      self.tracker.mb, self.tracker.mb_metadata = self.tracker.mb.prune(
          valid_fn=lambda mb: np.logical_and(
              mb.r > self.scenario['r_prune'],
              np.linalg.trace(
                  mb.state.covar[
                      np.ix_(
                          np.arange(len(mb)), self.pos_inds, self.pos_inds
                      )
                  ]
              ) < self.scenario['trace_prune']
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
        initiated = meta.get('initiated', False) or \
            (num_detections >= self.scenario['num_initiate_detections'])
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
              active=True,
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
      search_nodes['timestep'] = self.timestep
      search_nodes['weight'] = self.tracker.poisson.state.weight / \
          self.tracker.poisson.state.weight.max()
      search_nodes['covar_diag'] = np.sqrt(
          np.diagonal(self.tracker.poisson.state.covar, axis1=-1, axis2=-2)
      )

    ##############################
    # Agent nodes
    ##############################
    current_agent_node = dict(
        type='agent',
        label=node_label_map['agent'],
        name=f'agent_t{self.timestep}',
        id='agent',
        active=True,
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
    new_agent_edges = []
    new_agent_edge_features = dict(
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

      new_agent_edges.extend([
          (last_agent['name'], current_agent_node['name']),
          (current_agent_node['name'], last_agent['name']),
      ])
      new_agent_edge_features.update(
          type=new_agent_edge_features['type'] + ['transition', 'transition'],
          label=new_agent_edge_features['label'] +
          2*[edge_label_map['transition']],
          pd=new_agent_edge_features['pd'] + [0.0, 0.0],
          distance=new_agent_edge_features['distance'] + [0.0, 0.0],
          angle=new_agent_edge_features['angle'] + [0.0, 0.0],
          relative_position=new_agent_edge_features['relative_position'] + [
              last_agent['position'] - current_agent_node['position'],
              current_agent_node['position'] - last_agent['position'],
          ],
          relative_velocity=new_agent_edge_features['relative_velocity'] + [
              last_agent['velocity'] - current_agent_node['velocity'],
              current_agent_node['velocity'] - last_agent['velocity'],
          ],
      )

    # Search detection edges
    search_nodes = self.graph.vs(type_eq='search')
    search_pos = np.array(search_nodes['position'])
    if self.timestep > 0:
      search_pd = np.array([
          self.tracker.poisson_metadata[i]['pd']
          for i in range(len(search_nodes))
      ])
      detected_search = np.where(search_pd > 0)[0]
      num_search_edges = 2*min(len(detected_search), self.top_k_search_update)
      if len(detected_search) > self.top_k_search_update:
        detected_search = detected_search[
            np.argpartition(
                search_pd[detected_search], -self.top_k_search_update
            )[-self.top_k_search_update:]
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

        new_agent_edges.extend([
            (search_nodes[i]['name'], current_agent_node['name'])
            for i in detected_search
        ] + [
            (current_agent_node['name'], search_nodes[i]['name'])
            for i in detected_search
        ])
        new_agent_edge_features.update(
            type=new_agent_edge_features['type'] + num_search_edges*['update'],
            label=new_agent_edge_features['label'] +
            num_search_edges*[edge_label_map['update']],
            pd=new_agent_edge_features['pd'] + 2*search_edge_pd,
            distance=new_agent_edge_features['distance'] + 2*search_edge_dist,
            angle=new_agent_edge_features['angle'] + search_edge_angles,
            relative_position=new_agent_edge_features['relative_position'] +
            num_search_edges*[np.zeros(2)],
            relative_velocity=new_agent_edge_features['relative_velocity'] +
            num_search_edges*[np.zeros(2)],
        )

    self.graph.add_vertex(**current_agent_node)
    self.graph.add_edges(
        es=new_agent_edges, attributes=new_agent_edge_features
    )
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
    # TODO: Prioritize tracks when at max capacity
    # 1. Initiated tracks (just the first N if this is also over capacity)
    # 2. Undetected tracks according to initiation progress
    if len(self.tracker.mb) > 0 and self.max_active_tracks > 0:
      # Delete stale track nodes
      track_ids = [meta['id'] for meta in self.tracker.mb_metadata]
      self.graph.vs(type_eq='track', id_notin=track_ids)['delete'] = True

      # Collect track info for this timestep
      num_tracks = min(len(self.tracker.mb), self.max_active_tracks)
      new_track_nodes = dict(
          type='track',
          label=[node_label_map['track']],
          name=[],
          id=[],
          timestep=self.timestep,
          active=[],
          position=self.tracker.mb.state.mean[:num_tracks, self.pos_inds],
          velocity=self.tracker.mb.state.mean[:num_tracks, self.vel_inds],
          covar_diag=np.sqrt(np.diagonal(
              self.tracker.mb.state.covar[:num_tracks],
              axis1=-1, axis2=-2
          )),
          # Search features
          weight=np.zeros(num_tracks),
          # Agent features
          sensor_action=np.zeros((num_tracks, 1)),
          # Track features
          measurement_type=[],
          measurement_label=[],
          track_quality=[],
          existence_probability=self.tracker.mb.r[:num_tracks],
          initiation_progress=[],
      )
      new_track_edge_features = dict(
          type=[],
          label=[],
          pd=[],
          distance=[],
          angle=[],
          relative_position=[],
          relative_velocity=[],
      )
      new_track_edges = []
      track_nodes = self.graph.vs(type_eq='track')
      track_qualities = self.track_quality(self.tracker.mb)
      for i, meta in enumerate(self.tracker.mb_metadata):
        track_id = meta['id']
        track_history = track_nodes(id_eq=track_id)
        if i >= self.max_active_tracks:  # At track capacity
          track_history['delete'] = True
          continue

        # Update node attributes
        track_node_name = f"{track_id}_t{self.timestep}"
        track_measurement_type = meta['measurement_type']
        track_initiation_progress = 1.0 if meta['initiated'] else \
            meta['num_detections'] / self.scenario['num_initiate_detections']
        track_active = (
            (track_qualities[i] > self.scenario['min_active_quality']) and
            (self.tracker.mb.r[i] > self.scenario['min_active_r'])
        )
        track_history['active'] = track_active

        new_track_nodes.update(
            id=new_track_nodes['id'] + [track_id],
            name=new_track_nodes['name'] + [track_node_name],
            active=new_track_nodes['active'] + [track_active],
            measurement_type=new_track_nodes['measurement_type'] +
            [track_measurement_type],
            measurement_label=new_track_nodes['measurement_label'] +
            [measurement_label_map[track_measurement_type]],
            track_quality=new_track_nodes['track_quality'] +
            [track_qualities[i]],
            initiation_progress=new_track_nodes['initiation_progress'] +
            [track_initiation_progress],
        )

        # Transition edge
        track_updates = track_history(measurement_type_eq='update')
        if len(track_updates) > 0:
          last_update = track_updates[-1]
          track_pos = new_track_nodes['position'][i]
          track_vel = new_track_nodes['velocity'][i]

          new_track_edges.extend([
              (last_update['name'], track_node_name),
              (track_node_name, last_update['name'])
          ])
          new_track_edge_features.update(
              type=new_track_edge_features['type'] +
              ['transition', 'transition'],
              label=new_track_edge_features['label'] +
              2*[edge_label_map['transition']],
              pd=new_track_edge_features['pd'] + [0.0, 0.0],
              distance=new_track_edge_features['distance'] + [0.0, 0.0],
              angle=new_track_edge_features['angle'] + [0.0, 0.0],
              relative_position=new_track_edge_features['relative_position'] + [
                  last_update['position'] - track_pos,
                  track_pos - last_update['position'],
              ],
              relative_velocity=new_track_edge_features['relative_velocity'] + [
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
        new_track_edges.extend([
            (track_node_name, current_agent_node['name']),
            (current_agent_node['name'], track_node_name),
        ])
        new_track_edge_features.update(
            type=new_track_edge_features['type'] + 2*['update'],
            label=new_track_edge_features['label'] +
            2*[edge_label_map['update']],
            pd=new_track_edge_features['pd'] + 2*[track_pd],
            distance=new_track_edge_features['distance'] +
            2*[track_update_dist],
            angle=new_track_edge_features['angle'] + track_update_angles,
            relative_position=new_track_edge_features['relative_position'] +
            2*[np.zeros(2)],
            relative_velocity=new_track_edge_features['relative_velocity'] +
            2*[np.zeros(2)],
        )

        # Track graph pruning
        track_history(measurement_type_eq='predict')['delete'] = True
        if track_measurement_type == 'update':
          track_history(measurement_type_eq='miss')['delete'] = True
        if len(track_history) >= self.max_track_history:
          track_history[:-(self.max_track_history-1)]['delete'] = True

      # Add track nodes and edges
      self.graph.add_vertices(n=num_tracks, attributes=new_track_nodes)
      self.graph.add_edges(
          es=new_track_edges, attributes=new_track_edge_features
      )
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
    nodes = self.graph.vs(active_eq=True)
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
    graph = self.graph.subgraph(nodes)
    edges = graph.es
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
      edge_list = np.array(graph.get_edgelist())
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
        track_qualities = self.track_quality(self.tracker.mb[initiated])
      else:
        track_qualities = np.zeros(1)
    else:
      num_active_tracks = 0
      track_qualities = np.zeros(1)

    global_features = np.stack([
        w_sum,
        np.array([num_active_tracks]),
        np.array([track_qualities.mean()]),
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
    track_qualities = self.track_quality(self.tracker.mb[initiated])
    track_reward = track_qualities.sum()
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
      survived = self.np_random.uniform(size=len(self.ground_truth)) < ps
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

  def measure(self) -> np.ndarray:
    if len(self.ground_truth) == 0:
      return []

    # Object measurements
    states = np.array([path[-1] for path in self.ground_truth])
    pd = self.pd(states, sensor=self.sensor, pos_inds=self.pos_inds)
    detected = self.np_random.uniform(size=len(self.ground_truth)) < pd
    if np.any(detected):
      Z = self.measurement_model(
          states[detected],
          sensor_pos=self.sensor['position'],
          sensor_vel=self.sensor['velocity'],
          noise=True,
          rng=self.np_random
      )
    else:
      Z = np.empty((0, 2))

    # Clutter measurements
    Z_clutter = self.measure_clutter()
    if len(Z_clutter) > 0:
      Z = np.concatenate([Z, Z_clutter], axis=0)

    return Z

  def measure_clutter(self) -> np.ndarray:
    if self.scenario['clutter_rate'] > 0:
      num_clutter = self.np_random.poisson(
          lam=self.scenario['clutter_rate'] * self.scenario['dt']
      )
      if num_clutter > 0:
        clutter_range = self.np_random.uniform(
            low=100, high=self.sensor['max_range'], size=num_clutter
        )
        clutter_angle = self.np_random.uniform(
            low=self.sensor['steering_angle'] - self.sensor['beamwidth']/2,
            high=self.sensor['steering_angle'] + self.sensor['beamwidth']/2,
            size=num_clutter
        )
        Z = self.sensor['position'] + np.array([
            clutter_range * np.cos(clutter_angle),
            clutter_range * np.sin(clutter_angle),
        ]).T
        return Z

    return np.empty((0, 2))

  @staticmethod
  def pd(
      object_state: Union[np.ndarray, Gaussian],
      sensor: Dict[str, Any],
      pos_inds: List[int],
  ) -> np.ndarray:
    if isinstance(object_state, Gaussian):
      alpha, beta, kappa = 0.25, 2, 0
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
          ndim_state=len(pos_inds), alpha=alpha, beta=beta, kappa=kappa
      )[0]
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
      state_dim = len(pos_inds)
      alpha, beta, kappa = 1/np.sqrt(state_dim), 2, 0
      sigma_points = merwe_scaled_sigma_points(
          x=object_state.mean[:, pos_inds],
          P=object_state.covar[
              np.ix_(np.arange(object_state.shape[0]), pos_inds, pos_inds)
          ],
          alpha=alpha,
          beta=beta,
          kappa=kappa
      )
      weights = merwe_sigma_weights(
          ndim_state=len(pos_inds), alpha=alpha, beta=beta, kappa=kappa
      )[0]
      weights = abs(weights) / abs(weights).sum()
      ps = np.where(
          np.logical_and.reduce([
              sigma_points[..., 0] >= scenario['extents'][0][0],
              sigma_points[..., 0] <= scenario['extents'][0][1],
              sigma_points[..., 1] >= scenario['extents'][1][0],
              sigma_points[..., 1] <= scenario['extents'][1][1],
          ]), 0.999, 0
      )
      return np.average(ps, weights=weights, axis=-1)
    else:
      return np.where(
          np.logical_and.reduce([
              object_state[..., pos_inds[0]] >= scenario['extents'][0][0],
              object_state[..., pos_inds[0]] <= scenario['extents'][0][1],
              object_state[..., pos_inds[1]] >= scenario['extents'][1][0],
              object_state[..., pos_inds[1]] <= scenario['extents'][1][1],
          ]), 0.999, 0
      )

  def track_quality(self, tracks: MultiBernoulli) -> np.ndarray:
    """Compute track quality based on trace of covariance."""
    covars = tracks.state.covar[
        np.ix_(np.arange(len(tracks)), self.pos_inds, self.pos_inds)
    ]
    traces = np.linalg.trace(covars)
    track_quality = 1 - (traces / self.scenario['max_trace']).clip(0, 1)
    return track_quality

  def render(self, graph: igraph.Graph = None):
    if graph is None:
      graph = self.graph
    graph = graph.subgraph(graph.vs(active_eq=True))

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
    nx, ny = 50, 50
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
          f"{norm_weights[i]:.2f}\n{search_nodes[i]['covar_diag'][self.pos_inds].sum():.1f}",
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
            f"q={track_nodes[i]['track_quality']:.2f}\nr={track_nodes[i]['existence_probability']:.2f}",
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
