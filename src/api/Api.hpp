#pragma once

#include "tensor/Tensor.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace Tensor::api {

template <typename T> Tensor<T> make_scalar(const T &value) {
  auto storage = make_host_storage(sizeof(T), 64);
  std::vector<int64_t> shape{1};
  DTensor tensor(std::move(storage), shape, default_strides(shape), 0, dtype_of<T>(),
                 true, false);
  *static_cast<T *>(tensor.data()) = value;
  return Tensor<T>(std::move(tensor));
}

inline DTensor empty(const std::vector<int64_t> &shape, DType dtype,
                     bool requires_grad = false) {
  const auto bytes = static_cast<std::size_t>(numel_from_shape(shape)) * dtype_size(dtype);
  return DTensor(make_host_storage(bytes, 64), shape, default_strides(shape), 0,
                 dtype, true, requires_grad);
}

inline DTensor zeros(const std::vector<int64_t> &shape, DType dtype,
                     bool requires_grad = false) {
  DTensor tensor = empty(shape, dtype, requires_grad);
  std::memset(tensor.data(), 0, static_cast<std::size_t>(tensor.numel()) * dtype_size(dtype));
  return tensor;
}

template <typename T>
inline Tensor<T> empty(const std::vector<int64_t> &shape, bool requires_grad = false) {
  return Tensor<T>(empty(shape, dtype_of<T>(), requires_grad));
}

template <typename T>
inline Tensor<T> zeros(const std::vector<int64_t> &shape, bool requires_grad = false) {
  return Tensor<T>(zeros(shape, dtype_of<T>(), requires_grad));
}

template <typename T>
inline Tensor<T> ones(const std::vector<int64_t> &shape, bool requires_grad = false) {
  Tensor<T> tensor = empty<T>(shape, requires_grad);
  T *ptr = tensor.data();
  for (int64_t index = 0; index < tensor.numel(); ++index) {
    ptr[index] = T{1};
  }
  return tensor;
}

inline bool is_contiguous(const DTensor &tensor) {
  return is_default_contiguous(tensor.shape(), tensor.stride());
}

inline DTensor reshape(const DTensor &tensor, const std::vector<int64_t> &new_shape) {
  if (!is_contiguous(tensor)) {
    throw std::invalid_argument("reshape only supports contiguous tensors");
  }
  if (tensor.numel() != numel_from_shape(new_shape)) {
    throw std::invalid_argument("reshape requires the same element count");
  }
  return DTensor(tensor.storage(), new_shape, default_strides(new_shape), tensor.offset(),
                 tensor.dtype(), true, tensor.requires_grad(),
                 tensor.autograd_state());
}

inline DTensor permute(const DTensor &tensor, const std::vector<int> &perm) {
  if (static_cast<int>(perm.size()) != tensor.rank()) {
    throw std::invalid_argument("permute rank mismatch");
  }

  std::vector<int64_t> new_shape(perm.size());
  std::vector<int64_t> new_stride(perm.size());
  for (std::size_t index = 0; index < perm.size(); ++index) {
    const int axis = perm[index];
    if (axis < 0 || axis >= tensor.rank()) {
      throw std::invalid_argument("permute axis out of range");
    }
    new_shape[index] = tensor.shape()[static_cast<std::size_t>(axis)];
    new_stride[index] = tensor.stride()[static_cast<std::size_t>(axis)];
  }

  return DTensor(tensor.storage(), std::move(new_shape), std::move(new_stride),
                 tensor.offset(), tensor.dtype(), false, tensor.requires_grad(),
                 tensor.autograd_state());
}

} // namespace Tensor::api
