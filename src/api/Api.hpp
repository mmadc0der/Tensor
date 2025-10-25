#pragma once

#include "tensor/Tensor.hpp"
#include <cstdint>
#include <cstring>
#include <vector>

namespace Tensor::api {

// Create a scalar tensor on host and initialize with value
template <typename T> Tensor<T> make_scalar(const T &value) {
  // allocate 1 element host storage
  auto st = make_host_storage(sizeof(T), 64);
  std::vector<int64_t> shp{1};
  auto str = default_strides(shp);
  DTensor dt{std::move(st), std::move(shp), std::move(str),
             /*offset*/ 0,
             // map dtype via Tensor<T>
             [] {
               if constexpr (std::is_same_v<T, int>)
                 return DType::i32;
               else if constexpr (std::is_same_v<T, std::int32_t>)
                 return DType::i32;
               else if constexpr (std::is_same_v<T, std::int64_t>)
                 return DType::i64;
               else if constexpr (std::is_same_v<T, float>)
                 return DType::f32;
               else if constexpr (std::is_same_v<T, double>)
                 return DType::f64;
               else
                 return DType::f32;
             }(),
             Layout::contiguous,
             /*is_contiguous*/ true,
             /*requires_grad*/ false};
  // write the value
  *static_cast<T *>(dt.data()) = value;
  return Tensor<T>(std::move(dt));
}

// Create uninitialized contiguous DTensor on host
inline DTensor empty(const std::vector<int64_t> &shape, DType dtype) {
  const std::size_t elem = dtype_size(dtype);
  std::size_t bytes = elem;
  for (auto d : shape)
    bytes *= static_cast<std::size_t>(d);
  auto st = make_host_storage(bytes, 64);
  auto str = default_strides(shape);
  DTensor dt{std::move(st),          shape, std::move(str),
             /*offset*/ 0,           dtype, Layout::contiguous,
             /*is_contiguous*/ true,
             /*requires_grad*/ false};
  return dt;
}

// Create zero-initialized DTensor on host
inline DTensor zeros(const std::vector<int64_t> &shape, DType dtype) {
  DTensor dt = empty(shape, dtype);
  std::memset(dt.data(), 0, dt.storage()->size_bytes());
  return dt;
}

// Typed factories
template <typename T>
inline Tensor<T> empty(const std::vector<int64_t> &shape) {
  DTensor dt = empty(shape, [] {
    if constexpr (std::is_same_v<T, int>)
      return DType::i32;
    else if constexpr (std::is_same_v<T, std::int32_t>)
      return DType::i32;
    else if constexpr (std::is_same_v<T, std::int64_t>)
      return DType::i64;
    else if constexpr (std::is_same_v<T, float>)
      return DType::f32;
    else if constexpr (std::is_same_v<T, double>)
      return DType::f64;
    else
      return DType::f32;
  }());
  return Tensor<T>(std::move(dt));
}

template <typename T>
inline Tensor<T> zeros(const std::vector<int64_t> &shape) {
  DTensor dt = zeros(shape, [] {
    if constexpr (std::is_same_v<T, int>)
      return DType::i32;
    else if constexpr (std::is_same_v<T, std::int32_t>)
      return DType::i32;
    else if constexpr (std::is_same_v<T, std::int64_t>)
      return DType::i64;
    else if constexpr (std::is_same_v<T, float>)
      return DType::f32;
    else if constexpr (std::is_same_v<T, double>)
      return DType::f64;
    else
      return DType::f32;
  }());
  return Tensor<T>(std::move(dt));
}

template <typename T> inline Tensor<T> ones(const std::vector<int64_t> &shape) {
  Tensor<T> t = empty<T>(shape);
  T *p = t.data();
  for (int64_t i = 0, n = t.numel(); i < n; ++i)
    p[i] = T{1};
  return t;
}

// Helpers
inline bool is_contiguous(const DTensor &t) {
  const auto &shp = t.shape();
  const auto &str = t.stride();
  auto def = default_strides(shp);
  return def == str;
}

// View ops
inline DTensor reshape(const DTensor &t,
                       const std::vector<int64_t> &new_shape) {
  // Only allow reshape without copy if tensor is contiguous and numel matches
  int64_t old_n = t.numel();
  int64_t new_n = 1;
  for (auto d : new_shape)
    new_n *= d;
  assert(old_n == new_n && "reshape: numel mismatch");
  assert(is_contiguous(t) &&
         "reshape: only contiguous tensors supported for now");
  auto ns = default_strides(new_shape);
  DTensor out{t.storage(), new_shape,  std::move(ns), t.offset(),
              t.dtype(),   t.layout(), true,          t.requires_grad()};
  out.set_grad_fn(t.grad_fn());
  return out;
}

inline DTensor permute(const DTensor &t, const std::vector<int> &perm) {
  assert(static_cast<int>(perm.size()) == t.rank());
  std::vector<int64_t> new_shape(perm.size());
  std::vector<int64_t> new_stride(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    int p = perm[i];
    assert(p >= 0 && p < t.rank());
    new_shape[i] = t.shape()[p];
    new_stride[i] = t.stride()[p];
  }
  DTensor out{t.storage(),
              std::move(new_shape),
              std::move(new_stride),
              t.offset(),
              t.dtype(),
              t.layout(),
              false,
              t.requires_grad()};
  out.set_grad_fn(t.grad_fn());
  return out;
}

} // namespace Tensor::api
