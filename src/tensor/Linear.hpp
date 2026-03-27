#pragma once

#include "Ops.hpp"

#include <vector>

namespace Tensor::nn {

class Linear {
public:
  Linear(int64_t in_features, int64_t out_features);

  DTensor forward(const DTensor &input) const;

  DTensor &weight() noexcept { return weight_; }
  const DTensor &weight() const noexcept { return weight_; }
  DTensor &bias() noexcept { return bias_; }
  const DTensor &bias() const noexcept { return bias_; }

  std::vector<DTensor *> parameters();

private:
  DTensor weight_;
  DTensor bias_;
};

class SGD {
public:
  explicit SGD(float learning_rate) : learning_rate_(learning_rate) {}

  void zero_grad(const std::vector<DTensor *> &parameters) const;
  void step(const std::vector<DTensor *> &parameters) const;

private:
  float learning_rate_;
};

} // namespace Tensor::nn
