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


class GraphSearchEnv(gym.Env):
  def __init__(self):
    self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))

    # Search grid config
    self.nx_grid = 8
    self.ny_grid = 8
    self.knn = 4
    self.n_grid = self.nx_grid * self.ny_grid

    self.max_nodes = self.n_grid
    self.max_edges = self.knn * self.n_grid
    self.observation_space = gym.spaces.Dict(
        edge_features=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_edges, 1),
            dtype=np.float64,
        ),
        edge_list=gym.spaces.MultiDiscrete(
            np.full((self.max_edges, 2), self.max_nodes+1),
            start=np.full((self.max_edges, 2), -1)
        ),
        global_features=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, 2),
            dtype=np.float64,
        ),
        node_features=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_nodes, 3),
            dtype=np.float64,
        ),
    )

  def reset(
      self,
      seed: Optional[int] = None,
      **kwargs
  ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    self.timestep = 0
    if seed is not None:
      self.np_random, seed = gym.utils.seeding.np_random(seed)
      self.action_space.seed(seed)
      self.observation_space.seed(seed)

    self.scenario = dict(
        extents=np.array([
            [-1000, 1000],
            [-1000, 1000]
        ])
    )
    self.sensor = dict(
        position=np.zeros(2),
        beamwidth=10*np.pi/180,
        steering_angle=0
    )

    # Create the search grid
    xmin, xmax = self.scenario['extents'][0]
    ymin, ymax = self.scenario['extents'][1]
    x = np.linspace(xmin, xmax, self.nx_grid)
    y = np.linspace(ymin, ymax, self.ny_grid)
    dx = 0.5*(xmax - xmin) / self.nx_grid
    dy = 0.5*(ymax - ymin) / self.ny_grid
    grid_means = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    grid_covars = np.diag(np.array([dx, dy])**2)[None, ...].repeat(
        self.n_grid, axis=0
    )

    # Randomly initialize search grid
    birth_rate = 1/25
    init_wsum = self.np_random.uniform(1, 5)
    weights = self.np_random.uniform(0, 1, size=self.n_grid)
    weights = (weights / weights.sum()) * init_wsum

    self.search_grid = dict(
        positions=grid_means,
        covars=grid_covars,
        weights=weights,
        birth_rate=birth_rate,
        num_components=grid_means.shape[0]
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
        object_state=self.search_grid,
        scenario=self.scenario,
        sensor=self.sensor,
    )
    self.search_grid['weights'] = search_ps * self.search_grid['weights'] + \
        (self.search_grid['birth_rate'] / self.n_grid)

    # For reward calculation
    self.w_pred_sum = self.search_grid['weights'].sum()

    # Update step
    search_pd = self.pd(
        object_state=self.search_grid,
        sensor=self.sensor,
    )
    self.search_grid['weights'] = (1 - search_pd) * self.search_grid['weights']

    # Env update
    self.update_graph()
    obs = self.get_obs()
    reward = self.get_reward()
    terminated = truncated = False
    info = {}

    return obs, reward, terminated, truncated, info

  def update_graph(self) -> None:
    if self.timestep == 0:
      # Create search nodes
      self.graph.add_vertices(
          n=self.search_grid['num_components'],
          attributes=dict(
              type='search',
              name=[
                  f'search_{i}'
                  for i in range(self.search_grid['num_components'])
              ],
              position=self.search_grid['positions'],
              weight=self.search_grid['weights'],
          )
      )

      # Create static edges to K nearest neighbors
      distances = np.linalg.norm(
          self.search_grid['positions'][:, None] -
          self.search_grid['positions'][None, :],
          axis=-1
      )
      diag_inds = np.diag_indices(self.search_grid['num_components'])
      distances[diag_inds] = np.inf
      knn_inds = np.argpartition(distances, self.knn, axis=1)[:, :self.knn]
      self.graph.add_edges(
          es=[
              (self.graph.vs['name'][knn_inds[i, j]],
               self.graph.vs['name'][i])
              for i in range(self.search_grid['num_components'])
              for j in range(self.knn)
          ],
      )
      # Edge features
      src_pos = np.array([
          self.graph.vs['position'][edge.source] for edge in self.graph.es
      ])
      dst_pos = np.array([
          self.graph.vs['position'][edge.target] for edge in self.graph.es
      ])
      rel_pos = src_pos - dst_pos
      self.graph.es['distance'] = np.linalg.norm(rel_pos, axis=-1)
    else:
      self.graph.vs['weight'] = self.search_grid['weights']

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
        'position',
        'weight',
    ]
    node_dict = {
        k: np.array(nodes[k]).reshape((len(nodes), -1)) for k in node_keys
    }
    # Pre-process features
    node_dict.update(
        position=node_dict['position'] / position_scale[None, :],
        weight=node_dict['weight'] / node_dict['weight'].max(),
    )

    node_features = np.concatenate(
        [
            node_dict[key].astype(
                self.observation_space['node_features'].dtype
            ) for key in node_keys
        ],
        axis=-1
    )

    ###########################
    # Edges
    ###########################
    edges = self.graph.es
    edge_keys = [
        'distance',
    ]
    edge_dict = {
        k: np.array(edges[k]).reshape((len(edges), -1)) for k in edge_keys
    }
    # Pre-process features
    edge_dict.update(
        distance=edge_dict['distance'] / distance_scale
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

    ###########################
    # Global features
    ###########################
    wsum = np.sum(self.search_grid['weights'], keepdims=True)
    wmax = np.max(self.search_grid['weights'], keepdims=True)
    global_features = np.stack([
        wsum,
        np.log(wmax),
    ], axis=-1)

    obs = dict(
        edge_features=edge_features,
        edge_list=edge_list,
        global_features=global_features,
        node_features=node_features,
    )
    return obs

  def get_reward(self) -> float:
    w = self.search_grid['weights']
    # reward = (self.w_pred_sum - w.sum()) / w.max()
    reward = -w.sum()
    return reward

  def update_sensor_state(self, action: np.ndarray) -> None:
    self.sensor['steering_angle'] = action[0] * np.pi

  @staticmethod
  def pd(
      object_state: Union[np.ndarray, Gaussian],
      sensor: Dict[str, Any],
  ) -> np.ndarray:
    n = 2
    alpha, beta, kappa = 0.5, 2, 0
    points = merwe_scaled_sigma_points(
        x=object_state['positions'],
        P=object_state['covars'],
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
    pd = np.where(in_region, 0.9, 0).mean(axis=-1)
    return pd

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
    for i in range(len(search_nodes)):
      mixture += search_nodes[i]['weight'] * \
          scipy.stats.multivariate_normal.pdf(
          np.stack([search_grid[0].ravel(), search_grid[1].ravel()]).T,
          mean=search_nodes[i]['position'],
          cov=search_nodes[i]['covar'],
      ).reshape((nx, ny))
    plt.imshow(mixture, extent=self.scenario['extents'].ravel(
    ), origin='lower', aspect='auto')
    plt.colorbar()
    # plt.clim([0, 1e-6])

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
