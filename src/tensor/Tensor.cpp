#include "Tensor.hpp"

#include <cstdlib>
#include <new>
#include <utility>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace Tensor {

namespace {

void validate_shape_and_stride(const std::vector<int64_t> &shape,
                               const std::vector<int64_t> &stride,
                               int64_t offset) {
  if (shape.size() != stride.size()) {
    throw std::invalid_argument("shape and stride rank mismatch");
  }
  if (offset < 0) {
    throw std::invalid_argument("tensor offset must be non-negative");
  }
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (shape[index] < 0) {
      throw std::invalid_argument("tensor dimensions must be non-negative");
    }
    if (stride[index] < 0) {
      throw std::invalid_argument("tensor strides must be non-negative");
    }
  }
}

void *aligned_alloc_host(std::size_t alignment, std::size_t size) {
#if defined(_WIN32)
  return _aligned_malloc(size, alignment);
#else
  void *ptr = nullptr;
  if (alignment < sizeof(void *)) {
    alignment = sizeof(void *);
  }
  if (posix_memalign(&ptr, alignment, size) == 0) {
    return ptr;
  }
  return nullptr;
#endif
}

} // namespace

DTensor::DTensor(std::shared_ptr<Storage> storage, std::vector<int64_t> shape,
                 std::vector<int64_t> stride, int64_t offset, DType dtype,
                 bool is_contiguous, bool requires_grad,
                 std::shared_ptr<TensorAutogradState> autograd_state)
    : storage_(std::move(storage)), shape_(std::move(shape)),
      stride_(std::move(stride)), offset_(offset), dtype_(dtype),
      is_contiguous_(is_contiguous),
      autograd_state_(std::move(autograd_state)) {
  validate_shape_and_stride(shape_, stride_, offset_);
  if (!storage_) {
    throw std::invalid_argument("tensor storage must be valid");
  }
  if (!autograd_state_) {
    autograd_state_ = std::make_shared<TensorAutogradState>();
    autograd_state_->requires_grad = requires_grad;
  } else {
    autograd_state_->requires_grad = autograd_state_->requires_grad || requires_grad;
  }
}

bool DTensor::requires_grad() const noexcept {
  return autograd_state_ && autograd_state_->requires_grad;
}

void DTensor::set_requires_grad(bool value) noexcept {
  if (!autograd_state_) {
    autograd_state_ = std::make_shared<TensorAutogradState>();
  }
  autograd_state_->requires_grad = value;
  if (!value) {
    autograd_state_->grad.reset();
    autograd_state_->grad_fn.reset();
    autograd_state_->is_leaf = true;
  }
}

bool DTensor::is_leaf() const noexcept {
  return !autograd_state_ || autograd_state_->is_leaf;
}

std::shared_ptr<DTensor> DTensor::grad() const noexcept {
  return autograd_state_ ? autograd_state_->grad : nullptr;
}

void DTensor::set_grad(std::shared_ptr<DTensor> grad) noexcept {
  if (!autograd_state_) {
    autograd_state_ = std::make_shared<TensorAutogradState>();
  }
  autograd_state_->grad = std::move(grad);
}

void DTensor::zero_grad() noexcept {
  if (autograd_state_) {
    autograd_state_->grad.reset();
  }
}

std::shared_ptr<AutogradNode> DTensor::grad_fn() const noexcept {
  return autograd_state_ ? autograd_state_->grad_fn : nullptr;
}

void DTensor::set_grad_fn(std::shared_ptr<AutogradNode> fn) noexcept {
  if (!autograd_state_) {
    autograd_state_ = std::make_shared<TensorAutogradState>();
  }
  autograd_state_->grad_fn = std::move(fn);
  autograd_state_->is_leaf = autograd_state_->grad_fn == nullptr;
}

void *DTensor::data() noexcept {
  if (!storage_) {
    return nullptr;
  }
  auto *base = static_cast<std::byte *>(storage_->data());
  return base + (offset_ * static_cast<int64_t>(dtype_size(dtype_)));
}

const void *DTensor::data() const noexcept {
  if (!storage_) {
    return nullptr;
  }
  auto *base = static_cast<const std::byte *>(storage_->data());
  return base + (offset_ * static_cast<int64_t>(dtype_size(dtype_)));
}

std::shared_ptr<Storage> make_host_storage(std::size_t bytes,
                                           std::size_t alignment) {
  const std::size_t alloc_bytes = bytes == 0 ? 1 : bytes;
  void *raw = aligned_alloc_host(alignment, alloc_bytes);
  if (!raw) {
    throw std::bad_alloc{};
  }

#if defined(_WIN32)
  auto deleter = [](void *ptr) { _aligned_free(ptr); };
#else
  auto deleter = [](void *ptr) { std::free(ptr); };
#endif

  return std::make_shared<Storage>(std::shared_ptr<void>(raw, deleter), alloc_bytes,
                                   alignment);
}

} // namespace Tensor
