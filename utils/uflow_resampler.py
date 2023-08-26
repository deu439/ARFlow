# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for resampling images."""

import torch
import numpy as np


#def gather_nd_torch(params, indices, batch_dims=0):
#  """ The same as tf.gather_nd.
#  indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:
#
#  output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]
#
#  Args:
#      params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
#      indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.
#
#  Returns: gathered Tensor.
#      shape [y_0,y_2,...y_{k-2}] + params.shape[m:]
#
#  """
#  if isinstance(indices, torch.Tensor):
#    indices = indices.numpy()
#  else:
#    if not isinstance(indices, np.array):
#      raise ValueError(f'indices must be `torch.Tensor` or `numpy.array`. Got {type(indices)}')
#  if batch_dims == 0:
#    orig_shape = list(indices.shape)
#    num_samples = int(np.prod(orig_shape[:-1]))
#    m = orig_shape[-1]
#    n = len(params.shape)
#
#    if m <= n:
#      out_shape = orig_shape[:-1] + list(params.shape[m:])
#    else:
#      raise ValueError(
#        f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
#      )
#    indices = indices.reshape((num_samples, m)).transpose().tolist()
#    output = params[indices]  # (num_samples, ...)
#    return output.reshape(out_shape).contiguous()
#  else:
#    batch_shape = params.shape[:batch_dims]
#    orig_indices_shape = list(indices.shape)
#    orig_params_shape = list(params.shape)
#    assert (
#            batch_shape == indices.shape[:batch_dims]
#    ), f'if batch_dims is not 0, then both "params" and "indices" have batch_dims leading batch dimensions that exactly match.'
#    mbs = np.prod(batch_shape)
#    if batch_dims != 1:
#      params = params.reshape(mbs, *(params.shape[batch_dims:]))
#      indices = indices.reshape(mbs, *(indices.shape[batch_dims:]))
#    output = []
#    for i in range(mbs):
#      output.append(gather_nd(params[i], indices[i], batch_dims=0))
#    output = torch.stack(output, dim=0)
#    output_shape = orig_indices_shape[:-1] + list(orig_params_shape[orig_indices_shape[-1] + batch_dims:])
#    return output.reshape(*output_shape).contiguous()
def gather_nd_torch(params, indices):
  """ The same as tf.gather_nd but batched gather is not supported yet.
  indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

  output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

  Args:
      params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
      indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

  Returns: gathered Tensor.
      shape [y_0,y_2,...y_{k-2}] + params.shape[m:]

  """
  orig_shape = list(indices.shape)
  num_samples = np.prod(orig_shape[:-1])
  m = orig_shape[-1]
  n = len(params.shape)

  if m <= n:
    out_shape = orig_shape[:-1] + list(params.shape)[m:]
  else:
    raise ValueError(
      f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
    )

  indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
  output = params[indices]  # (num_samples, ...)
  return output.reshape(out_shape).contiguous()


def safe_gather_nd(params, indices):
  """Gather slices from params into a Tensor with shape specified by indices.

  Similar functionality to tf.gather_nd with difference: when index is out of
  bound, always return 0.

  Args:
    params: A Tensor. The tensor from which to gather values.
    indices: A Tensor. Must be one of the following types: int32, int64. Index
      tensor.

  Returns:
    A Tensor. Has the same type as params. Values from params gathered from
    specified indices (if they exist) otherwise zeros, with shape
    indices.shape[:-1] + params.shape[indices.shape[-1]:].
  """
  params_shape = params.size()
  indices_shape = indices.size()
  slice_dimensions = indices_shape[-1]

  max_index = torch.tensor(params_shape[:slice_dimensions], dtype=torch.int32, device=params.device) - 1
  min_index = torch.zeros_like(max_index, dtype=torch.int32, device=params.device)

  clipped_indices = torch.clamp(indices, min=min_index, max=max_index)

  # Check whether each component of each index is in range [min, max], and
  # allow an index only if all components are in range:
  mask = torch.all((indices >= min_index) & (indices <= max_index), dim=-1)
  mask = torch.unsqueeze(mask, -1)

  return mask.to(params.dtype) * gather_nd_torch(params, clipped_indices)


def resampler(data, warp):
  """Resamples input data at user defined coordinates.

  Args:
    data: Tensor of shape `[batch_size, data_height, data_width,
      data_num_channels]` containing 2D data that will be resampled.
    warp: Tensor shape `[batch_size, dim_0, ... , dim_n, 2]` containing the
      coordinates at which resampling will be performed.
    name: Optional name of the op.

  Returns:
    Tensor of resampled values from `data`. The output tensor shape is
    `[batch_size, dim_0, ... , dim_n, data_num_channels]`.
  """
  warp_x, warp_y = torch.unbind(warp, axis=-1)
  return resampler_with_unstacked_warp(data, warp_x, warp_y)


def resampler_with_unstacked_warp(data,
                                  warp_x,
                                  warp_y,
                                  safe=True):
  """Resamples input data at user defined coordinates.

  The resampler functions in the same way as `resampler` above, with the
  following differences:
  1. The warp coordinates for x and y are given as separate tensors.
  2. If warp_x and warp_y are known to be within their allowed bounds, (that is,
     0 <= warp_x <= width_of_data - 1, 0 <= warp_y <= height_of_data - 1) we
     can disable the `safe` flag.

  Args:
    data: Tensor of shape `[batch_size, data_height, data_width,
      data_num_channels]` containing 2D data that will be resampled.
    warp_x: Tensor of shape `[batch_size, dim_0, ... , dim_n]` containing the x
      coordinates at which resampling will be performed.
    warp_y: Tensor of the same shape as warp_x containing the y coordinates at
      which resampling will be performed.
    safe: A boolean, if True, warp_x and warp_y will be clamped to their bounds.
      Disable only if you know they are within bounds, otherwise a runtime
      exception will be thrown.
    name: Optional name of the op.

  Returns:
     Tensor of resampled values from `data`. The output tensor shape is
    `[batch_size, dim_0, ... , dim_n, data_num_channels]`.

  Raises:
    ValueError: If warp_x, warp_y and data have incompatible shapes.
  """

  assert warp_x.size() == warp_y.size(), "warp_x and warp_y incompatible!"
  warp_shape = torch.tensor(warp_x.size(), dtype=torch.int32, device=data.device)
  assert warp_x.shape[0] == data.shape[0], "warp_x and data have incompatible first dimension (batch size)"

  # Compute the four points closest to warp with integer value.
  warp_floor_x = torch.floor(warp_x)
  warp_floor_y = torch.floor(warp_y)
  # Compute the weight for each point.
  right_warp_weight = warp_x - warp_floor_x
  down_warp_weight = warp_y - warp_floor_y

  warp_floor_x = warp_floor_x.to(torch.int32)
  warp_floor_y = warp_floor_y.to(torch.int32)
  warp_ceil_x = torch.ceil(warp_x).to(torch.int32)
  warp_ceil_y = torch.ceil(warp_y).to(torch.int32)

  left_warp_weight = 1.0 - right_warp_weight
  up_warp_weight = 1.0 - down_warp_weight

  # Extend warps from [batch_size, dim_0, ... , dim_n, 2] to
  # [batch_size, dim_0, ... , dim_n, 3] with the first element in last
  # dimension being the batch index.

  # A shape like warp_shape but with all sizes except the first set to 1:
  #warp_batch_shape = [warp_shape[0]] + [1]*len(warp_shape[1:])
  warp_batch_shape = torch.concat([warp_shape[0:1], torch.ones_like(warp_shape[1:], device=data.device)], dim=0)
  warp_batch = torch.reshape(torch.arange(warp_shape[0], dtype=torch.int32, device=data.device), warp_batch_shape.tolist())

  # Broadcast to match shape:
  warp_batch = warp_batch + torch.zeros_like(warp_y, dtype=torch.int32, device=data.device)
  left_warp_weight = torch.unsqueeze(left_warp_weight, dim=-1)
  down_warp_weight = torch.unsqueeze(down_warp_weight, dim=-1)
  up_warp_weight = torch.unsqueeze(up_warp_weight, dim=-1)
  right_warp_weight = torch.unsqueeze(right_warp_weight, dim=-1)

  up_left_warp = torch.stack([warp_batch, warp_floor_y, warp_floor_x], dim=-1)
  up_right_warp = torch.stack([warp_batch, warp_floor_y, warp_ceil_x], dim=-1)
  down_left_warp = torch.stack([warp_batch, warp_ceil_y, warp_floor_x], dim=-1)
  down_right_warp = torch.stack([warp_batch, warp_ceil_y, warp_ceil_x], dim=-1)

  def gather_nd(params, indices):
    return (safe_gather_nd if safe else gather_nd_torch)(params, indices)

  # gather data then take weighted average to get resample result.
  result = (
      (gather_nd(data, up_left_warp) * left_warp_weight +
       gather_nd(data, up_right_warp) * right_warp_weight) * up_warp_weight +
      (gather_nd(data, down_left_warp) * left_warp_weight +
       gather_nd(data, down_right_warp) * right_warp_weight) *
      down_warp_weight)
  #result_shape = (
  #    warp_x.get_shape().as_list() + data.get_shape().as_list()[-1:])
  #result.set_shape(result_shape)
  return result
