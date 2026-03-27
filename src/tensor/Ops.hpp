#pragma once

#include "Tensor.hpp"

#include <vector>

namespace Tensor::ops {

DTensor clone(const DTensor &tensor);
DTensor zeros_like(const DTensor &tensor);
DTensor ones_like(const DTensor &tensor);

void fill(DTensor &tensor, float value);
void copy(const DTensor &src, DTensor &dst);

DTensor add(const DTensor &lhs, const DTensor &rhs);
DTensor sub(const DTensor &lhs, const DTensor &rhs);
DTensor mul(const DTensor &lhs, const DTensor &rhs);
DTensor matmul(const DTensor &lhs, const DTensor &rhs);
DTensor sum(const DTensor &tensor);
DTensor mean(const DTensor &tensor);
DTensor relu(const DTensor &tensor);
DTensor clamp(const DTensor &tensor, float min_value, float max_value);
DTensor bias_add(const DTensor &value, const DTensor &bias);
DTensor mse_loss(const DTensor &prediction, const DTensor &target);

void backward(const DTensor &loss);

} // namespace Tensor::ops
