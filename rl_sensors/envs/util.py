import numpy as np
from motpy.rfs.poisson import Poisson
from motpy.distributions import Gaussian


def merge_poisson(
    distribution: Poisson,
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

  return Poisson(state=merged_state)
