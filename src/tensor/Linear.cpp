#include "tensor/Linear.hpp"

#include "api/Api.hpp"

#include <cmath>
#include <stdexcept>

namespace Tensor::nn {

namespace {

float *f32_data(DTensor &tensor) {
  return static_cast<float *>(tensor.data());
}

const float *f32_data(const DTensor &tensor) {
  return static_cast<const float *>(tensor.data());
}

} // namespace

Linear::Linear(int64_t in_features, int64_t out_features)
    : weight_(api::zeros({in_features, out_features}, DType::f32, true)),
      bias_(api::zeros({out_features}, DType::f32, true)) {
  if (in_features <= 0 || out_features <= 0) {
    throw std::invalid_argument("Linear requires positive feature dimensions");
  }

  float *weight_ptr = f32_data(weight_);
  const float scale = 1.0f / std::sqrt(static_cast<float>(in_features));
  for (int64_t row = 0; row < in_features; ++row) {
    for (int64_t col = 0; col < out_features; ++col) {
      const int64_t flat_index = row * out_features + col;
      const float centered = static_cast<float>((flat_index % 7) - 3) / 3.0f;
      weight_ptr[flat_index] = centered * scale;
    }
  }

  ops::fill(bias_, 0.0f);
}

DTensor Linear::forward(const DTensor &input) const {
  if (input.dtype() != DType::f32) {
    throw std::invalid_argument("Linear currently supports only f32 inputs");
  }
  if (input.rank() != 2) {
    throw std::invalid_argument("Linear expects rank-2 input");
  }
  if (input.shape()[1] != weight_.shape()[0]) {
    throw std::invalid_argument("Linear input width must match weight rows");
  }

  return ops::bias_add(ops::matmul(input, weight_), bias_);
}

std::vector<DTensor *> Linear::parameters() {
  return {&weight_, &bias_};
}

void SGD::zero_grad(const std::vector<DTensor *> &parameters) const {
  for (DTensor *parameter : parameters) {
    if (parameter != nullptr) {
      parameter->zero_grad();
    }
  }
}

void SGD::step(const std::vector<DTensor *> &parameters) const {
  for (DTensor *parameter : parameters) {
    if (parameter == nullptr) {
      continue;
    }
    if (parameter->dtype() != DType::f32) {
      throw std::invalid_argument("SGD currently supports only f32 parameters");
    }
    if (!parameter->grad()) {
      continue;
    }

    float *param_ptr = f32_data(*parameter);
    const float *grad_ptr = f32_data(*parameter->grad());
    for (int64_t index = 0; index < parameter->numel(); ++index) {
      param_ptr[index] -= learning_rate_ * grad_ptr[index];
    }
  }
}

} // namespace Tensor::nn
