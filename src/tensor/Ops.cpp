#include "tensor/Ops.hpp"

#include "api/Api.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace Tensor::ops {

namespace {

void require_contiguous(const DTensor &tensor, const char *op_name) {
  if (!tensor.is_contiguous()) {
    throw std::invalid_argument(std::string(op_name) +
                                " only supports contiguous tensors");
  }
}

void require_f32(const DTensor &tensor, const char *op_name) {
  if (tensor.dtype() != DType::f32) {
    throw std::invalid_argument(std::string(op_name) +
                                " currently supports only f32 tensors");
  }
}

void require_same_shape(const DTensor &lhs, const DTensor &rhs, const char *op_name) {
  if (lhs.shape() != rhs.shape()) {
    throw std::invalid_argument(std::string(op_name) + " requires matching shapes");
  }
}

float *f32_data(DTensor &tensor) {
  return static_cast<float *>(tensor.data());
}

const float *f32_data(const DTensor &tensor) {
  return static_cast<const float *>(tensor.data());
}

DTensor make_f32_tensor(const std::vector<int64_t> &shape, bool requires_grad = false) {
  return api::zeros(shape, DType::f32, requires_grad);
}

void add_inplace_f32(DTensor &dst, const DTensor &src) {
  require_f32(dst, "add_inplace");
  require_f32(src, "add_inplace");
  require_contiguous(dst, "add_inplace");
  require_contiguous(src, "add_inplace");
  require_same_shape(dst, src, "add_inplace");

  float *dst_ptr = f32_data(dst);
  const float *src_ptr = f32_data(src);
  for (int64_t index = 0; index < dst.numel(); ++index) {
    dst_ptr[index] += src_ptr[index];
  }
}

void scale_inplace_f32(DTensor &tensor, float scale) {
  require_f32(tensor, "scale_inplace");
  require_contiguous(tensor, "scale_inplace");

  float *ptr = f32_data(tensor);
  for (int64_t index = 0; index < tensor.numel(); ++index) {
    ptr[index] *= scale;
  }
}

void accumulate_gradient(DTensor tensor, const DTensor &grad);

struct AddBackward final : AutogradNode {
  AddBackward(DTensor lhs_in, DTensor rhs_in)
      : lhs(std::move(lhs_in)), rhs(std::move(rhs_in)) {}

  void backward(const DTensor &upstream) override {
    if (lhs.requires_grad()) {
      accumulate_gradient(lhs, upstream);
    }
    if (rhs.requires_grad()) {
      accumulate_gradient(rhs, upstream);
    }
  }

  DTensor lhs;
  DTensor rhs;
};

struct SubBackward final : AutogradNode {
  SubBackward(DTensor lhs_in, DTensor rhs_in)
      : lhs(std::move(lhs_in)), rhs(std::move(rhs_in)) {}

  void backward(const DTensor &upstream) override {
    if (lhs.requires_grad()) {
      accumulate_gradient(lhs, upstream);
    }
    if (rhs.requires_grad()) {
      DTensor grad_rhs = clone(upstream);
      scale_inplace_f32(grad_rhs, -1.0f);
      accumulate_gradient(rhs, grad_rhs);
    }
  }

  DTensor lhs;
  DTensor rhs;
};

struct MulBackward final : AutogradNode {
  MulBackward(DTensor lhs_in, DTensor rhs_in)
      : lhs(std::move(lhs_in)), rhs(std::move(rhs_in)) {}

  void backward(const DTensor &upstream) override {
    if (lhs.requires_grad()) {
      DTensor grad_lhs = make_f32_tensor(lhs.shape());
      float *dst = f32_data(grad_lhs);
      const float *up = f32_data(upstream);
      const float *rhs_ptr = f32_data(rhs);
      for (int64_t index = 0; index < lhs.numel(); ++index) {
        dst[index] = up[index] * rhs_ptr[index];
      }
      accumulate_gradient(lhs, grad_lhs);
    }

    if (rhs.requires_grad()) {
      DTensor grad_rhs = make_f32_tensor(rhs.shape());
      float *dst = f32_data(grad_rhs);
      const float *up = f32_data(upstream);
      const float *lhs_ptr = f32_data(lhs);
      for (int64_t index = 0; index < rhs.numel(); ++index) {
        dst[index] = up[index] * lhs_ptr[index];
      }
      accumulate_gradient(rhs, grad_rhs);
    }
  }

  DTensor lhs;
  DTensor rhs;
};

struct SumBackward final : AutogradNode {
  explicit SumBackward(DTensor input_in) : input(std::move(input_in)) {}

  void backward(const DTensor &upstream) override {
    if (!input.requires_grad()) {
      return;
    }

    DTensor grad_input = make_f32_tensor(input.shape());
    const float scalar = f32_data(upstream)[0];
    float *dst = f32_data(grad_input);
    for (int64_t index = 0; index < input.numel(); ++index) {
      dst[index] = scalar;
    }
    accumulate_gradient(input, grad_input);
  }

  DTensor input;
};

struct MeanBackward final : AutogradNode {
  explicit MeanBackward(DTensor input_in) : input(std::move(input_in)) {}

  void backward(const DTensor &upstream) override {
    if (!input.requires_grad()) {
      return;
    }

    DTensor grad_input = make_f32_tensor(input.shape());
    const float scalar = f32_data(upstream)[0] /
                         static_cast<float>(std::max<int64_t>(input.numel(), 1));
    float *dst = f32_data(grad_input);
    for (int64_t index = 0; index < input.numel(); ++index) {
      dst[index] = scalar;
    }
    accumulate_gradient(input, grad_input);
  }

  DTensor input;
};

struct ReluBackward final : AutogradNode {
  explicit ReluBackward(DTensor input_in) : input(std::move(input_in)) {}

  void backward(const DTensor &upstream) override {
    if (!input.requires_grad()) {
      return;
    }

    DTensor grad_input = make_f32_tensor(input.shape());
    float *dst = f32_data(grad_input);
    const float *up = f32_data(upstream);
    const float *in = f32_data(input);
    for (int64_t index = 0; index < input.numel(); ++index) {
      dst[index] = in[index] > 0.0f ? up[index] : 0.0f;
    }
    accumulate_gradient(input, grad_input);
  }

  DTensor input;
};

struct ClampBackward final : AutogradNode {
  ClampBackward(DTensor input_in, float min_in, float max_in)
      : input(std::move(input_in)), min_value(min_in), max_value(max_in) {}

  void backward(const DTensor &upstream) override {
    if (!input.requires_grad()) {
      return;
    }

    DTensor grad_input = make_f32_tensor(input.shape());
    float *dst = f32_data(grad_input);
    const float *up = f32_data(upstream);
    const float *in = f32_data(input);
    for (int64_t index = 0; index < input.numel(); ++index) {
      const bool active = in[index] > min_value && in[index] < max_value;
      dst[index] = active ? up[index] : 0.0f;
    }
    accumulate_gradient(input, grad_input);
  }

  DTensor input;
  float min_value;
  float max_value;
};

struct MatmulBackward final : AutogradNode {
  MatmulBackward(DTensor lhs_in, DTensor rhs_in)
      : lhs(std::move(lhs_in)), rhs(std::move(rhs_in)) {}

  void backward(const DTensor &upstream) override {
    const int64_t m = lhs.shape()[0];
    const int64_t k = lhs.shape()[1];
    const int64_t n = rhs.shape()[1];

    const float *lhs_ptr = f32_data(lhs);
    const float *rhs_ptr = f32_data(rhs);
    const float *up_ptr = f32_data(upstream);

    if (lhs.requires_grad()) {
      DTensor grad_lhs = make_f32_tensor(lhs.shape());
      float *dst = f32_data(grad_lhs);
      for (int64_t row = 0; row < m; ++row) {
        for (int64_t inner = 0; inner < k; ++inner) {
          float acc = 0.0f;
          for (int64_t col = 0; col < n; ++col) {
            acc += up_ptr[row * n + col] * rhs_ptr[inner * n + col];
          }
          dst[row * k + inner] = acc;
        }
      }
      accumulate_gradient(lhs, grad_lhs);
    }

    if (rhs.requires_grad()) {
      DTensor grad_rhs = make_f32_tensor(rhs.shape());
      float *dst = f32_data(grad_rhs);
      for (int64_t inner = 0; inner < k; ++inner) {
        for (int64_t col = 0; col < n; ++col) {
          float acc = 0.0f;
          for (int64_t row = 0; row < m; ++row) {
            acc += lhs_ptr[row * k + inner] * up_ptr[row * n + col];
          }
          dst[inner * n + col] = acc;
        }
      }
      accumulate_gradient(rhs, grad_rhs);
    }
  }

  DTensor lhs;
  DTensor rhs;
};

struct BiasAddBackward final : AutogradNode {
  BiasAddBackward(DTensor value_in, DTensor bias_in)
      : value(std::move(value_in)), bias(std::move(bias_in)) {}

  void backward(const DTensor &upstream) override {
    if (value.requires_grad()) {
      accumulate_gradient(value, upstream);
    }

    if (!bias.requires_grad()) {
      return;
    }

    const int64_t rows = upstream.shape()[0];
    const int64_t cols = upstream.shape()[1];
    DTensor grad_bias = make_f32_tensor(bias.shape());
    float *dst = f32_data(grad_bias);
    const float *up = f32_data(upstream);
    for (int64_t col = 0; col < cols; ++col) {
      float acc = 0.0f;
      for (int64_t row = 0; row < rows; ++row) {
        acc += up[row * cols + col];
      }
      dst[col] = acc;
    }
    accumulate_gradient(bias, grad_bias);
  }

  DTensor value;
  DTensor bias;
};

void accumulate_gradient(DTensor tensor, const DTensor &grad) {
  if (!tensor.requires_grad()) {
    return;
  }

  if (tensor.is_leaf()) {
    if (!tensor.grad()) {
      tensor.set_grad(std::make_shared<DTensor>(clone(grad)));
    } else {
      add_inplace_f32(*tensor.grad(), grad);
    }
  }

  if (auto fn = tensor.grad_fn()) {
    fn->backward(grad);
  }
}

} // namespace

DTensor clone(const DTensor &tensor) {
  DTensor result = api::empty(tensor.shape(), tensor.dtype(), false);
  std::memcpy(result.data(), tensor.data(),
              static_cast<std::size_t>(tensor.numel()) * dtype_size(tensor.dtype()));
  return result;
}

DTensor zeros_like(const DTensor &tensor) {
  return api::zeros(tensor.shape(), tensor.dtype(), false);
}

DTensor ones_like(const DTensor &tensor) {
  DTensor result = zeros_like(tensor);
  require_f32(result, "ones_like");
  fill(result, 1.0f);
  return result;
}

void fill(DTensor &tensor, float value) {
  require_f32(tensor, "fill");
  require_contiguous(tensor, "fill");
  float *ptr = f32_data(tensor);
  for (int64_t index = 0; index < tensor.numel(); ++index) {
    ptr[index] = value;
  }
}

void copy(const DTensor &src, DTensor &dst) {
  if (src.dtype() != dst.dtype()) {
    throw std::invalid_argument("copy requires matching dtypes");
  }
  require_contiguous(src, "copy");
  require_contiguous(dst, "copy");
  require_same_shape(src, dst, "copy");
  std::memcpy(dst.data(), src.data(),
              static_cast<std::size_t>(src.numel()) * dtype_size(src.dtype()));
}

DTensor add(const DTensor &lhs, const DTensor &rhs) {
  require_contiguous(lhs, "add");
  require_contiguous(rhs, "add");
  require_f32(lhs, "add");
  require_f32(rhs, "add");
  require_same_shape(lhs, rhs, "add");

  const bool needs_grad = lhs.requires_grad() || rhs.requires_grad();
  DTensor result = make_f32_tensor(lhs.shape(), needs_grad);
  float *dst = f32_data(result);
  const float *lhs_ptr = f32_data(lhs);
  const float *rhs_ptr = f32_data(rhs);
  for (int64_t index = 0; index < lhs.numel(); ++index) {
    dst[index] = lhs_ptr[index] + rhs_ptr[index];
  }

  if (needs_grad) {
    result.set_grad_fn(std::make_shared<AddBackward>(lhs, rhs));
  }
  return result;
}

DTensor sub(const DTensor &lhs, const DTensor &rhs) {
  require_contiguous(lhs, "sub");
  require_contiguous(rhs, "sub");
  require_f32(lhs, "sub");
  require_f32(rhs, "sub");
  require_same_shape(lhs, rhs, "sub");

  const bool needs_grad = lhs.requires_grad() || rhs.requires_grad();
  DTensor result = make_f32_tensor(lhs.shape(), needs_grad);
  float *dst = f32_data(result);
  const float *lhs_ptr = f32_data(lhs);
  const float *rhs_ptr = f32_data(rhs);
  for (int64_t index = 0; index < lhs.numel(); ++index) {
    dst[index] = lhs_ptr[index] - rhs_ptr[index];
  }

  if (needs_grad) {
    result.set_grad_fn(std::make_shared<SubBackward>(lhs, rhs));
  }
  return result;
}

DTensor mul(const DTensor &lhs, const DTensor &rhs) {
  require_contiguous(lhs, "mul");
  require_contiguous(rhs, "mul");
  require_f32(lhs, "mul");
  require_f32(rhs, "mul");
  require_same_shape(lhs, rhs, "mul");

  const bool needs_grad = lhs.requires_grad() || rhs.requires_grad();
  DTensor result = make_f32_tensor(lhs.shape(), needs_grad);
  float *dst = f32_data(result);
  const float *lhs_ptr = f32_data(lhs);
  const float *rhs_ptr = f32_data(rhs);
  for (int64_t index = 0; index < lhs.numel(); ++index) {
    dst[index] = lhs_ptr[index] * rhs_ptr[index];
  }

  if (needs_grad) {
    result.set_grad_fn(std::make_shared<MulBackward>(lhs, rhs));
  }
  return result;
}

DTensor matmul(const DTensor &lhs, const DTensor &rhs) {
  require_contiguous(lhs, "matmul");
  require_contiguous(rhs, "matmul");
  require_f32(lhs, "matmul");
  require_f32(rhs, "matmul");
  if (lhs.rank() != 2 || rhs.rank() != 2) {
    throw std::invalid_argument("matmul requires rank-2 tensors");
  }
  if (lhs.shape()[1] != rhs.shape()[0]) {
    throw std::invalid_argument("matmul dimension mismatch");
  }

  const int64_t m = lhs.shape()[0];
  const int64_t k = lhs.shape()[1];
  const int64_t n = rhs.shape()[1];
  const bool needs_grad = lhs.requires_grad() || rhs.requires_grad();
  DTensor result = make_f32_tensor({m, n}, needs_grad);
  float *dst = f32_data(result);
  const float *lhs_ptr = f32_data(lhs);
  const float *rhs_ptr = f32_data(rhs);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int64_t inner = 0; inner < k; ++inner) {
        acc += lhs_ptr[row * k + inner] * rhs_ptr[inner * n + col];
      }
      dst[row * n + col] = acc;
    }
  }

  if (needs_grad) {
    result.set_grad_fn(std::make_shared<MatmulBackward>(lhs, rhs));
  }
  return result;
}

DTensor sum(const DTensor &tensor) {
  require_contiguous(tensor, "sum");
  require_f32(tensor, "sum");

  DTensor result = make_f32_tensor({1}, tensor.requires_grad());
  float total = 0.0f;
  const float *ptr = f32_data(tensor);
  for (int64_t index = 0; index < tensor.numel(); ++index) {
    total += ptr[index];
  }
  f32_data(result)[0] = total;

  if (tensor.requires_grad()) {
    result.set_grad_fn(std::make_shared<SumBackward>(tensor));
  }
  return result;
}

DTensor mean(const DTensor &tensor) {
  require_contiguous(tensor, "mean");
  require_f32(tensor, "mean");

  DTensor result = make_f32_tensor({1}, tensor.requires_grad());
  const float total = f32_data(sum(tensor))[0];
  f32_data(result)[0] = tensor.numel() == 0
                            ? 0.0f
                            : total / static_cast<float>(tensor.numel());

  if (tensor.requires_grad()) {
    result.set_grad_fn(std::make_shared<MeanBackward>(tensor));
  }
  return result;
}

DTensor relu(const DTensor &tensor) {
  require_contiguous(tensor, "relu");
  require_f32(tensor, "relu");

  DTensor result = make_f32_tensor(tensor.shape(), tensor.requires_grad());
  float *dst = f32_data(result);
  const float *src = f32_data(tensor);
  for (int64_t index = 0; index < tensor.numel(); ++index) {
    dst[index] = std::max(src[index], 0.0f);
  }

  if (tensor.requires_grad()) {
    result.set_grad_fn(std::make_shared<ReluBackward>(tensor));
  }
  return result;
}

DTensor clamp(const DTensor &tensor, float min_value, float max_value) {
  require_contiguous(tensor, "clamp");
  require_f32(tensor, "clamp");
  if (min_value > max_value) {
    throw std::invalid_argument("clamp requires min_value <= max_value");
  }

  DTensor result = make_f32_tensor(tensor.shape(), tensor.requires_grad());
  float *dst = f32_data(result);
  const float *src = f32_data(tensor);
  for (int64_t index = 0; index < tensor.numel(); ++index) {
    dst[index] = std::clamp(src[index], min_value, max_value);
  }

  if (tensor.requires_grad()) {
    result.set_grad_fn(std::make_shared<ClampBackward>(tensor, min_value, max_value));
  }
  return result;
}

DTensor bias_add(const DTensor &value, const DTensor &bias) {
  require_contiguous(value, "bias_add");
  require_contiguous(bias, "bias_add");
  require_f32(value, "bias_add");
  require_f32(bias, "bias_add");
  if (value.rank() != 2) {
    throw std::invalid_argument("bias_add expects a rank-2 value tensor");
  }
  if (!(bias.rank() == 1 || (bias.rank() == 2 && bias.shape()[0] == 1))) {
    throw std::invalid_argument("bias_add expects a rank-1 bias or shape {1, N}");
  }

  const int64_t cols = value.shape()[1];
  const int64_t bias_cols = bias.rank() == 1 ? bias.shape()[0] : bias.shape()[1];
  if (cols != bias_cols) {
    throw std::invalid_argument("bias_add output width must match bias size");
  }

  const int64_t rows = value.shape()[0];
  const bool needs_grad = value.requires_grad() || bias.requires_grad();
  DTensor result = make_f32_tensor(value.shape(), needs_grad);
  float *dst = f32_data(result);
  const float *value_ptr = f32_data(value);
  const float *bias_ptr = f32_data(bias);

  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      dst[row * cols + col] = value_ptr[row * cols + col] + bias_ptr[col];
    }
  }

  if (needs_grad) {
    result.set_grad_fn(std::make_shared<BiasAddBackward>(value, bias));
  }
  return result;
}

DTensor mse_loss(const DTensor &prediction, const DTensor &target) {
  DTensor diff = sub(prediction, target);
  return mean(mul(diff, diff));
}

void backward(const DTensor &loss) {
  require_f32(loss, "backward");
  if (loss.numel() != 1) {
    throw std::invalid_argument("backward expects a scalar loss tensor");
  }
  accumulate_gradient(loss, ones_like(loss));
}

} // namespace Tensor::ops
