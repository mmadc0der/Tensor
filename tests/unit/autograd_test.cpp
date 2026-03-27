#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "api/Api.hpp"
#include "tensor/Ops.hpp"

namespace {

Tensor::DTensor trainable_tensor(const std::vector<int64_t> &shape,
                                 const std::vector<float> &values) {
  auto tensor = Tensor::api::zeros(shape, Tensor::DType::f32, true);
  std::copy(values.begin(), values.end(), static_cast<float *>(tensor.data()));
  return tensor;
}

float scalar_loss_for_a00(float a00) {
  auto lhs = Tensor::api::zeros({2, 2}, Tensor::DType::f32, false);
  auto rhs = Tensor::api::zeros({2, 1}, Tensor::DType::f32, false);

  auto *lhs_ptr = static_cast<float *>(lhs.data());
  lhs_ptr[0] = a00;
  lhs_ptr[1] = 2.0f;
  lhs_ptr[2] = 3.0f;
  lhs_ptr[3] = 4.0f;

  auto *rhs_ptr = static_cast<float *>(rhs.data());
  rhs_ptr[0] = 5.0f;
  rhs_ptr[1] = 6.0f;

  auto out = Tensor::ops::matmul(lhs, rhs);
  auto loss = Tensor::ops::sum(out);
  return static_cast<const float *>(loss.data())[0];
}

} // namespace

TEST(Autograd, ElementwiseGradientsMatchHandCalculation) {
  auto lhs = trainable_tensor({2}, {1.0f, 2.0f});
  auto rhs = trainable_tensor({2}, {3.0f, 4.0f});

  auto loss = Tensor::ops::sum(Tensor::ops::add(Tensor::ops::mul(lhs, rhs), lhs));
  Tensor::ops::backward(loss);

  const auto *lhs_grad = static_cast<const float *>(lhs.grad()->data());
  const auto *rhs_grad = static_cast<const float *>(rhs.grad()->data());
  EXPECT_FLOAT_EQ(lhs_grad[0], 4.0f);
  EXPECT_FLOAT_EQ(lhs_grad[1], 5.0f);
  EXPECT_FLOAT_EQ(rhs_grad[0], 1.0f);
  EXPECT_FLOAT_EQ(rhs_grad[1], 2.0f);
}

TEST(Autograd, ReluMasksNegativeValues) {
  auto input = trainable_tensor({3}, {-2.0f, 0.5f, 3.0f});

  auto loss = Tensor::ops::sum(Tensor::ops::relu(input));
  Tensor::ops::backward(loss);

  const auto *grad = static_cast<const float *>(input.grad()->data());
  EXPECT_FLOAT_EQ(grad[0], 0.0f);
  EXPECT_FLOAT_EQ(grad[1], 1.0f);
  EXPECT_FLOAT_EQ(grad[2], 1.0f);
}

TEST(Autograd, MatmulGradientMatchesFiniteDifference) {
  auto lhs = trainable_tensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  auto rhs = trainable_tensor({2, 1}, {5.0f, 6.0f});

  auto loss = Tensor::ops::sum(Tensor::ops::matmul(lhs, rhs));
  Tensor::ops::backward(loss);

  const auto analytical = static_cast<const float *>(lhs.grad()->data())[0];
  constexpr float epsilon = 1.0e-3f;
  const float finite_difference =
      (scalar_loss_for_a00(1.0f + epsilon) - scalar_loss_for_a00(1.0f - epsilon)) /
      (2.0f * epsilon);

  EXPECT_NEAR(analytical, finite_difference, 1.0e-2f);
}

TEST(Autograd, LeafGradientsAccumulateAcrossBackwardCalls) {
  auto input = trainable_tensor({2}, {1.0f, 2.0f});

  Tensor::ops::backward(Tensor::ops::sum(input));
  Tensor::ops::backward(Tensor::ops::sum(input));

  const auto *grad = static_cast<const float *>(input.grad()->data());
  EXPECT_FLOAT_EQ(grad[0], 2.0f);
  EXPECT_FLOAT_EQ(grad[1], 2.0f);
}
