from typing import *

import gymnasium as gym
import igraph
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from motpy.distributions.gaussian import Gaussian
import scipy.stats
from motpy.estimators.kalman.sigma_points import (merwe_scaled_sigma_points,
                                                  merwe_sigma_weights)
from motpy.rfs.tomb import TOMBP


class GraphSearchEnv(gym.Env):
  def __init__(self):
    self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))

    # Search grid config
    self.nx_grid = 8
    self.ny_grid = 8
    self.n_grid = self.nx_grid * self.ny_grid

    # Agent config
    self.max_agent_nodes = 20
    self.top_k_search_update = 4

    self.max_nodes = self.n_grid + self.max_agent_nodes
    self.max_edges = (
        # Agent transition
        + 2*(self.max_agent_nodes - 1)
        # Agent-search update
        + 2*(self.top_k_search_update*self.max_agent_nodes)
    )
    self.observation_space = gym.spaces.Dict(
        edge_features=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_edges, 5),
            dtype=np.float64,
        ),
        edge_list=gym.spaces.MultiDiscrete(
            np.full((self.max_edges, 2), self.max_nodes+1),
            start=np.full((self.max_edges, 2), -1)
        ),
        edge_mask=gym.spaces.MultiBinary(self.max_edges),
        global_features=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, 2),
            dtype=np.float64,
        ),
        node_features=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_nodes, 7),
            dtype=np.float64,
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
        birth_rate=1/25,
    )
    self.sensor = dict(
        position=np.zeros(2),
        beamwidth=10*np.pi/180,
        steering_angle=0,
        action=np.zeros(1),
    )

    ###########################
    # Initialize tracker
    ###########################
    # Birth distribution
    xmin, xmax = self.scenario['extents'][0]
    ymin, ymax = self.scenario['extents'][1]
    x = np.linspace(xmin, xmax, self.nx_grid)
    y = np.linspace(ymin, ymax, self.ny_grid)
    dx = 0.5*(xmax - xmin) / self.nx_grid
    dy = 0.5*(ymax - ymin) / self.ny_grid
    birth_means = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    birth_covars = np.diag(np.array([dx, dy])**2)[None, ...].repeat(
        self.n_grid, axis=0
    )
    birth_state = Gaussian(
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
        birth_state=birth_state,
        undetected_state=undetected_state,
        pg=None,
        poisson_pd_threshold=None,
    )

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

    # Update scenario state
    self.update_sensor_state(action)

    # Predict step: Surivival and birth
    search_ps = self.ps(
        object_state=self.tracker.poisson.state,
        scenario=self.scenario,
        sensor=self.sensor,
    )
    # TODO: Replace with TOMB/P predict
    self.tracker.poisson.state.weight = \
        search_ps * self.tracker.poisson.state.weight + \
        self.tracker.poisson.birth_distribution.weight

    # Update step
    search_pd = self.pd(
        object_state=self.tracker.poisson.state,
        sensor=self.sensor,
    )
    # TODO: Replace with TOMB/P update
    self.tracker.poisson.state.weight = (1 - search_pd) * \
        self.tracker.poisson.state.weight

    # Env update
    self.update_graph()
    obs = self.get_obs()
    reward = self.get_reward()
    terminated = truncated = False
    info = {}

    return obs, reward, terminated, truncated, info

  def update_graph(self) -> None:
    #############################
    # Search nodes
    #############################
    if self.timestep == 0:
      self.graph.add_vertices(
          n=self.tracker.poisson.size,
          attributes=dict(
              name=[
                  f'search_{i}' for i in range(self.tracker.poisson.size)
              ],
              type='search',
              timestep=self.timestep,
              class_label=np.array([[1, 0]]),
              position=self.tracker.poisson.state.mean,
              covar=self.tracker.poisson.state.covar,
              weight=self.tracker.poisson.state.weight,
              sensor_action=np.zeros((1, 1)),
          )
      )
    else:
      search_nodes = self.graph.vs(type_eq='search')
      search_nodes['weight'] = self.tracker.poisson.state.weight
      search_nodes['timestep'] = self.timestep

    ##############################
    # Agent nodes
    ##############################
    self.graph.add_vertex(
        name=f'agent_{self.timestep}',
        type='agent',
        timestep=self.timestep,
        class_label=np.array([0, 1]),
        position=self.sensor['position'],
        weight=0.0,
        sensor_action=self.sensor['action'],
    )
    current_agent = self.graph.vs(
        type_eq='agent', timestep_eq=self.timestep
    )[0]
    if self.timestep > 0:
      # Add an edge from the previous agent node to the current
      last_agent = self.graph.vs(
          type_eq='agent', timestep_eq=self.timestep-1
      )[0]
      agent_edge_dist = np.linalg.norm(
          np.array(current_agent['position']) -
          np.array(last_agent['position']),
      )
      self.graph.add_edge(
          type='transition',
          source=last_agent,
          target=current_agent,
          class_label=np.array([1, 0]),
          distance=agent_edge_dist,
          angle=np.arctan2(
              last_agent['position'][1] - current_agent['position'][1],
              last_agent['position'][0] - current_agent['position'][0]
          ),
          pd=0.0,
      )
      self.graph.add_edge(
          type='transition',
          source=current_agent,
          target=last_agent,
          class_label=np.array([1, 0]),
          distance=agent_edge_dist,
          angle=np.arctan2(
              current_agent['position'][1] - last_agent['position'][1],
              current_agent['position'][0] - last_agent['position'][0]
          ),
          pd=0.0,
      )

    # Search detection edges
    search_nodes = self.graph.vs(type_eq='search')
    search_pos = np.array(search_nodes['position'])
    sensor_pos = np.array(self.sensor['position'])
    if self.timestep > 0:
      search_pd = self.pd(
          object_state=self.tracker.poisson.state,
          sensor=self.sensor,
      )
      detected_search = np.where(search_pd > 0)[0]
      num_search_updates = min(len(detected_search), self.top_k_search_update)
      if num_search_updates > 0:
        detected_search = detected_search[
            np.argpartition(
                search_pd[detected_search], -num_search_updates
            )[-num_search_updates:]
        ]

        # NOTE: Assumes search nodes have the same ordering as the search grid
        search_edge_dist = np.linalg.norm(
            search_pos[detected_search] - sensor_pos, axis=-1
        )

        self.graph.add_edges(
            es=[
                (search_nodes[i], current_agent)
                for i in detected_search
            ],
            attributes=dict(
                type='update',
                class_label=np.array([[0, 1]]),
                pd=search_pd[detected_search],
                distance=search_edge_dist,
                angle=np.arctan2(
                    search_pos[detected_search, 1] - sensor_pos[1],
                    search_pos[detected_search, 0] - sensor_pos[0]
                )
            )
        )
        self.graph.add_edges(
            es=[
                (current_agent, search_nodes[i])
                for i in detected_search
            ],
            attributes=dict(
                type='update',
                class_label=np.array([[0, 1]]),
                pd=search_pd[detected_search],
                distance=search_edge_dist,
                angle=np.arctan2(
                    sensor_pos[1] - search_pos[detected_search, 1],
                    sensor_pos[0] - search_pos[detected_search, 1]
                ),
            )
        )

    # Remove old agent nodes
    agent_nodes = self.graph.vs(type_eq='agent')
    if len(agent_nodes) > self.max_agent_nodes:
      self.graph.delete_vertices(agent_nodes[:-self.max_agent_nodes])

    ###############################
    # Shared features
    ###############################
    self.graph.vs['age'] = self.timestep - np.array(self.graph.vs['timestep'])

  def get_obs(self) -> Dict[str, Any]:
    ###########################
    # Scale factors
    ###########################
    position_scale = 0.5*np.diff(self.scenario['extents'], axis=-1).ravel()
    distance_scale = np.linalg.norm(position_scale)

    ###########################
    # Nodes
    ###########################
    nodes = self.graph.vs
    node_keys = [
        'class_label',
        'age',
        'position',
        'weight',
        'sensor_action',
    ]
    node_dict = {
        k: np.array(nodes[k]).reshape((len(nodes), -1)) for k in node_keys
    }
    # Pre-process features
    node_dict.update(
        age=np.log1p(node_dict['age']),
        position=node_dict['position'] / position_scale[None, :],
        weight=node_dict['weight'] / (node_dict['weight'].max() + 1e-10),
    )

    node_features = np.concatenate(
        [
            node_dict[key].astype(
                self.observation_space['node_features'].dtype
            ) for key in node_keys
        ],
        axis=-1
    )

    # Pad nodes
    node_features = np.pad(
        node_features,
        ((0, self.max_nodes - len(nodes)), (0, 0)),
        mode='constant', constant_values=0,
    )
    node_mask = np.arange(self.max_nodes) < len(nodes)

    ###########################
    # Edges
    ###########################
    edges = self.graph.es
    if len(edges) > 0:
      edge_keys = [
          'class_label',
          'pd',
          'distance',
          'angle',
      ]
      edge_dict = {
          k: np.array(edges[k]).reshape((len(edges), -1)) for k in edge_keys
      }
      # Pre-process features
      edge_dict.update(
          distance=edge_dict['distance'] / distance_scale,
          angle=edge_dict['angle'] / np.pi,
      )
      edge_features = np.concatenate(
          [
              edge_dict[key].astype(
                  self.observation_space['edge_features'].dtype
              ) for key in edge_keys
          ],
          axis=-1
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
    w_sum = np.sum(self.tracker.poisson.state.weight, keepdims=True)
    w_max = np.max(self.tracker.poisson.state.weight, keepdims=True)
    global_features = np.stack([
        w_sum,
        np.log(w_max + 1e-10),
    ], axis=-1)

    obs = dict(
        edge_features=edge_features,
        edge_list=edge_list,
        edge_mask=edge_mask,
        global_features=global_features,
        node_features=node_features,
        node_mask=node_mask,
    )
    return obs

  def get_reward(self) -> float:
    w = self.tracker.poisson.state.weight
    reward = -w.sum()
    return reward

  def update_sensor_state(self, action: np.ndarray) -> None:
    self.sensor['steering_angle'] = action[0] * np.pi
    self.sensor['action'] = action

  @staticmethod
  def pd(
      object_state: Union[np.ndarray, Gaussian],
      sensor: Dict[str, Any],
  ) -> np.ndarray:
    n = 2
    alpha, beta, kappa = 0.5, 2, 0
    points = merwe_scaled_sigma_points(
        x=object_state.mean,
        P=object_state.covar,
        alpha=alpha,
        beta=beta,
        kappa=kappa
    )
    weights = merwe_sigma_weights(
        ndim_state=n, alpha=alpha, beta=beta, kappa=kappa)[0]
    weights = abs(weights) / abs(weights).sum()
    weights = weights[None, :]

    sensor_pos = sensor['position']
    beamwidth = sensor['beamwidth']
    steering_angle = sensor['steering_angle']

    az = np.arctan2(
        points[..., 1] - sensor_pos[1],
        points[..., 0] - sensor_pos[0]
    )
    angle_diff = np.mod((az - steering_angle) + np.pi, 2*np.pi) - np.pi
    in_region = abs(angle_diff) <= beamwidth/2
    pd = np.where(in_region, 0.9, 0)
    return np.sum(pd * weights, axis=-1)

  @staticmethod
  def ps(
      object_state: Union[np.ndarray, Gaussian],
      scenario: Dict[str, Any],
      sensor: Dict[str, Any],
  ) -> np.ndarray:
    return 0.999

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
    # Plot gaussian mixture (likelihood) as an image
    mixture = np.zeros((nx, ny))
    norm_weights = search_nodes['weight'] / \
        (np.max(search_nodes['weight']) + 1e-10)
    for i in range(len(search_nodes)):
      mixture += search_nodes[i]['weight'] * \
          scipy.stats.multivariate_normal.pdf(
          np.stack([search_grid[0].ravel(), search_grid[1].ravel()]).T,
          mean=search_nodes[i]['position'],
          cov=search_nodes[i]['covar'],
      ).reshape((nx, ny))
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

    plt.xlim(self.scenario['extents'][0])
    plt.ylim(self.scenario['extents'][1])
    plt.pause(0.01)
    plt.draw()
