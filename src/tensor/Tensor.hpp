#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace Tensor {

enum class DType : uint8_t { f32, f64, i32, i64 };

class DTensor;
struct AutogradNode;
struct TensorAutogradState;

class Storage {
public:
  Storage() = default;
  Storage(std::shared_ptr<void> ptr, std::size_t bytes, std::size_t alignment)
      : data_(std::move(ptr)), bytes_(bytes), alignment_(alignment) {}

  bool valid() const noexcept { return static_cast<bool>(data_); }
  void *data() noexcept { return data_.get(); }
  const void *data() const noexcept { return data_.get(); }
  std::size_t size_bytes() const noexcept { return bytes_; }
  std::size_t alignment() const noexcept { return alignment_; }

private:
  std::shared_ptr<void> data_{};
  std::size_t bytes_{0};
  std::size_t alignment_{64};
};

inline constexpr std::size_t dtype_size(DType dt) noexcept {
  switch (dt) {
  case DType::f32:
    return 4;
  case DType::f64:
    return 8;
  case DType::i32:
    return 4;
  case DType::i64:
    return 8;
  }
  return 0;
}

inline constexpr bool is_floating_dtype(DType dt) noexcept {
  return dt == DType::f32 || dt == DType::f64;
}

inline std::int64_t numel_from_shape(const std::vector<int64_t> &shape) {
  std::int64_t total = 1;
  for (const auto dim : shape) {
    if (dim < 0) {
      throw std::invalid_argument("tensor dimensions must be non-negative");
    }
    total *= dim;
  }
  return total;
}

inline std::vector<int64_t> default_strides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  int64_t running = 1;
  for (int index = static_cast<int>(shape.size()) - 1; index >= 0; --index) {
    strides[static_cast<std::size_t>(index)] = running;
    running *= shape[static_cast<std::size_t>(index)];
  }
  return strides;
}

inline bool is_default_contiguous(const std::vector<int64_t> &shape,
                                  const std::vector<int64_t> &stride) {
  return default_strides(shape) == stride;
}

std::shared_ptr<Storage> make_host_storage(std::size_t bytes,
                                           std::size_t alignment = 64);

struct TensorAutogradState {
  bool requires_grad{false};
  bool is_leaf{true};
  std::shared_ptr<DTensor> grad{};
  std::shared_ptr<AutogradNode> grad_fn{};
};

class DTensor {
public:
  DTensor() = default;
  DTensor(std::shared_ptr<Storage> storage, std::vector<int64_t> shape,
          std::vector<int64_t> stride, int64_t offset, DType dtype,
          bool is_contiguous = false, bool requires_grad = false,
          std::shared_ptr<TensorAutogradState> autograd_state = nullptr);

  const std::vector<int64_t> &shape() const noexcept { return shape_; }
  const std::vector<int64_t> &stride() const noexcept { return stride_; }
  int64_t offset() const noexcept { return offset_; }
  DType dtype() const noexcept { return dtype_; }
  bool is_contiguous() const noexcept { return is_contiguous_; }
  std::shared_ptr<Storage> storage() const noexcept { return storage_; }
  bool defined() const noexcept { return static_cast<bool>(storage_); }
  int32_t rank() const noexcept { return static_cast<int32_t>(shape_.size()); }
  int64_t numel() const { return numel_from_shape(shape_); }

  bool requires_grad() const noexcept;
  void set_requires_grad(bool value) noexcept;
  bool is_leaf() const noexcept;
  std::shared_ptr<DTensor> grad() const noexcept;
  void set_grad(std::shared_ptr<DTensor> grad) noexcept;
  void zero_grad() noexcept;
  std::shared_ptr<AutogradNode> grad_fn() const noexcept;
  void set_grad_fn(std::shared_ptr<AutogradNode> fn) noexcept;
  std::shared_ptr<TensorAutogradState> autograd_state() const noexcept {
    return autograd_state_;
  }

  void *data() noexcept;
  const void *data() const noexcept;

private:
  std::shared_ptr<Storage> storage_{};
  std::vector<int64_t> shape_{};
  std::vector<int64_t> stride_{};
  int64_t offset_{0};
  DType dtype_{DType::f32};
  bool is_contiguous_{false};
  std::shared_ptr<TensorAutogradState> autograd_state_{};
};

struct AutogradNode : std::enable_shared_from_this<AutogradNode> {
  virtual void backward(const DTensor &upstream) = 0;
  virtual ~AutogradNode() = default;
};

template <typename T> constexpr DType dtype_of() noexcept {
  if constexpr (std::is_same_v<T, int> || std::is_same_v<T, std::int32_t>) {
    return DType::i32;
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    return DType::i64;
  } else if constexpr (std::is_same_v<T, float>) {
    return DType::f32;
  } else if constexpr (std::is_same_v<T, double>) {
    return DType::f64;
  } else {
    static_assert(!sizeof(T), "unsupported tensor scalar type");
  }
}

template <typename T> class Tensor {
public:
  Tensor() = default;
  explicit Tensor(DTensor dt) : dt_(std::move(dt)) {
    if (dt_.dtype() != dtype_of<T>()) {
      throw std::invalid_argument("typed tensor dtype mismatch");
    }
  }

  const std::vector<int64_t> &shape() const noexcept { return dt_.shape(); }
  const std::vector<int64_t> &stride() const noexcept { return dt_.stride(); }
  int64_t offset() const noexcept { return dt_.offset(); }
  int32_t rank() const noexcept { return dt_.rank(); }
  int64_t numel() const { return dt_.numel(); }
  bool is_contiguous() const noexcept { return dt_.is_contiguous(); }

  T *data() noexcept { return static_cast<T *>(dt_.data()); }
  const T *data() const noexcept { return static_cast<const T *>(dt_.data()); }

  DTensor &as_dtensor() & noexcept { return dt_; }
  const DTensor &as_dtensor() const & noexcept { return dt_; }
  DTensor as_dtensor() && noexcept { return std::move(dt_); }

  bool requires_grad() const noexcept { return dt_.requires_grad(); }
  void set_requires_grad(bool value) noexcept { dt_.set_requires_grad(value); }

  std::shared_ptr<DTensor> grad() const noexcept { return dt_.grad(); }
  void zero_grad() noexcept { dt_.zero_grad(); }

private:
  DTensor dt_{};
};

} // namespace Tensor
