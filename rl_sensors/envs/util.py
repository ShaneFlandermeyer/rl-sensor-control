import numpy as np
from motpy.rfs.poisson import Poisson
from motpy.distributions import Gaussian
from motpy.distributions.gaussian import merge_gaussians
from typing import *


def static_merge_poisson(
    distribution: Poisson,
    metadata: List[Dict[str, Any]],
    source_inds: np.ndarray,
    target_inds: np.ndarray
) -> Poisson:
  # Static merge source components into target components
  merged_weights = (
      distribution.state.weight[source_inds] +
      distribution.state.weight[target_inds]
  )
  merged_means = distribution.state.mean[target_inds]
  merged_covars = distribution.state.covar[target_inds]
  merged_state = Gaussian(
      mean=merged_means,
      covar=merged_covars,
      weight=merged_weights
  )

  merged_metadata = [metadata[i] for i in target_inds]

  return Poisson(state=merged_state), merged_metadata


def merge_poisson(
    distribution: Poisson,
    metadata: List[Dict[str, Any]],
    source_inds: np.ndarray,
    target_inds: np.ndarray
) -> Poisson:
  mu = distribution.state.mean[[source_inds, target_inds]].swapaxes(0, 1)
  P = distribution.state.covar[[source_inds, target_inds]].swapaxes(0, 1)
  w = distribution.state.weight[[source_inds, target_inds]].swapaxes(0, 1)
  merged_mu, merged_P, merged_w = merge_gaussians(
      means=mu, covars=P, weights=w
  )
  merged_state = Gaussian(
      mean=merged_mu,
      covar=merged_P,
      weight=merged_w
  )

  merged_metadata = [metadata[i] for i in target_inds]

  return Poisson(state=merged_state), merged_metadata
