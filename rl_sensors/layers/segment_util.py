
import functools
from typing import Any, Callable, Generator, Iterable, Iterator, List, Mapping, Optional, Sequence, Union

import jax
from jax import lax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np

# As of 04/2020 pytype doesn't support recursive types.
# pytype: disable=not-supported-yet
ArrayTree = Union[jnp.ndarray,
                  Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]


def segment_max(data: jnp.ndarray,
                segment_ids: jnp.ndarray,
                num_segments: Optional[int] = None,
                indices_are_sorted: bool = False,
                unique_indices: bool = False):
  """Alias for jax.ops.segment_max.

  Args:
    data: an array with the values to be maxed over.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis) to be maxed over. Values can be repeated
      and need not be sorted. Values outside of the range [0, num_segments) are
      dropped and do not contribute to the result.
    num_segments: optional, an int with positive value indicating the number of
      segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
      jnp.max(-segment_ids))`` but since `num_segments` determines the size of
      the output, a static value must be provided to use ``segment_max`` in a
      ``jit``-compiled function.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted
    unique_indices: whether ``segment_ids`` is known to be free of duplicates

  Returns:
    An array with shape ``(num_segments,) + data.shape[1:]`` representing
    the segment maxs.
  """
  return jax.ops.segment_max(data, segment_ids, num_segments,
                             indices_are_sorted, unique_indices)


def _replace_empty_segments_with_constant(aggregated_segments: jnp.ndarray,
                                          segment_ids: jnp.ndarray,
                                          num_segments: Optional[int] = None,
                                          constant: float = 0.):
  """Replaces the values of empty segments with constants."""
  result_shape = (len(segment_ids),) + aggregated_segments.shape[1:]
  num_elements_in_segment = jax.ops.segment_sum(
      jnp.ones(result_shape, dtype=jnp.int32),
      segment_ids,
      num_segments=num_segments)
  return jnp.where(num_elements_in_segment > 0, aggregated_segments,
                   jnp.array(constant, dtype=aggregated_segments.dtype))


def segment_max_or_constant(data: jnp.ndarray,
                            segment_ids: jnp.ndarray,
                            num_segments: Optional[int] = None,
                            indices_are_sorted: bool = False,
                            unique_indices: bool = False,
                            constant: float = 0.):
  """As segment_max, but returns a constant for empty segments.

  `segment_max` returns `-inf` for empty segments, which can cause `nan`s in the
  backwards pass of a neural network, even with masking. This method overrides
  the default behaviour of `segment_max` and returns a constant for empty
  segments.

  Args:
    data: an array with the values to be maxed over.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis) to be maxed over. Values can be repeated
      and need not be sorted. Values outside of the range [0, num_segments) are
      dropped and do not contribute to the result.
    num_segments: optional, an int with positive value indicating the number of
      segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
      jnp.max(-segment_ids))`` but since `num_segments` determines the size of
      the output, a static value must be provided to use ``segment_max`` in a
      ``jit``-compiled function.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted
    unique_indices: whether ``segment_ids`` is known to be free of duplicates
    constant: The constant to replace empty segments with, defaults to zero.

  Returns:
    An array with shape ``(num_segments,) + data.shape[1:]`` representing
    the segment maxs.
  """
  maxs_ = segment_max(data, segment_ids, num_segments, indices_are_sorted,
                      unique_indices)
  return _replace_empty_segments_with_constant(maxs_, segment_ids, num_segments,
                                               constant)


def segment_softmax(logits: jnp.ndarray,
                    segment_ids: jnp.ndarray,
                    num_segments: Optional[int] = None,
                    indices_are_sorted: bool = False,
                    unique_indices: bool = False) -> ArrayTree:
  """Computes a segment-wise softmax.

  For a given tree of logits that can be divded into segments, computes a
  softmax over the segments.

    logits = jnp.ndarray([1.0, 2.0, 3.0, 1.0, 2.0])
    segment_ids = jnp.ndarray([0, 0, 0, 1, 1])
    segment_softmax(logits, segments)
    >> DeviceArray([0.09003057, 0.24472848, 0.66524094, 0.26894142, 0.7310586],
    >> dtype=float32)

  Args:
    logits: an array of logits to be segment softmaxed.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis) to be maxed over. Values can be repeated
      and need not be sorted. Values outside of the range [0, num_segments) are
      dropped and do not contribute to the result.
    num_segments: optional, an int with positive value indicating the number of
      segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
      jnp.max(-segment_ids))`` but since ``num_segments`` determines the size of
      the output, a static value must be provided to use ``segment_sum`` in a
      ``jit``-compiled function.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted
    unique_indices: whether ``segment_ids`` is known to be free of duplicates

  Returns:
    The segment softmax-ed ``logits``.
  """
  # First, subtract the segment max for numerical stability
  maxs = segment_max_or_constant(
      logits,
      segment_ids,
      num_segments,
      indices_are_sorted,
      unique_indices
  )
  logits = logits - maxs[segment_ids]
  # Then take the exp
  logits = jnp.exp(logits)
  # Then calculate the normalizers
  normalizers = jax.ops.segment_sum(logits, segment_ids, num_segments,
                                    indices_are_sorted, unique_indices)
  normalizers = normalizers[segment_ids]
  softmax = logits / (normalizers + jnp.finfo(logits).eps)
  return softmax
