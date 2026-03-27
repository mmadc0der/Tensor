#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include "api/Api.hpp"
#include "tensor/Ops.hpp"

namespace {

Tensor::DTensor tensor_from_values(const std::vector<int64_t> &shape,
                                   const std::vector<float> &values) {
  auto tensor = Tensor::api::zeros(shape, Tensor::DType::f32, false);
  std::copy(values.begin(), values.end(), static_cast<float *>(tensor.data()));
  return tensor;
}

} // namespace

TEST(OpsForward, FillAndCopy) {
  auto src = Tensor::api::zeros({2, 2}, Tensor::DType::f32, false);
  auto dst = Tensor::api::zeros({2, 2}, Tensor::DType::f32, false);

  Tensor::ops::fill(src, 3.5f);
  Tensor::ops::copy(src, dst);

  const auto *ptr = static_cast<const float *>(dst.data());
  for (int index = 0; index < 4; ++index) {
    EXPECT_FLOAT_EQ(ptr[index], 3.5f);
  }
}

TEST(OpsForward, AddMulAndSum) {
  auto lhs = tensor_from_values({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  auto rhs = tensor_from_values({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});

  auto added = Tensor::ops::add(lhs, rhs);
  auto multiplied = Tensor::ops::mul(lhs, rhs);
  auto total = Tensor::ops::sum(multiplied);

  const auto *add_ptr = static_cast<const float *>(added.data());
  const auto *mul_ptr = static_cast<const float *>(multiplied.data());
  EXPECT_FLOAT_EQ(add_ptr[0], 6.0f);
  EXPECT_FLOAT_EQ(add_ptr[3], 12.0f);
  EXPECT_FLOAT_EQ(mul_ptr[0], 5.0f);
  EXPECT_FLOAT_EQ(mul_ptr[3], 32.0f);
  EXPECT_FLOAT_EQ(static_cast<const float *>(total.data())[0], 70.0f);
}

TEST(OpsForward, MatmulReferenceCase) {
  auto lhs = tensor_from_values({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto rhs = tensor_from_values({3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

  auto out = Tensor::ops::matmul(lhs, rhs);
  const auto *ptr = static_cast<const float *>(out.data());

  EXPECT_FLOAT_EQ(ptr[0], 58.0f);
  EXPECT_FLOAT_EQ(ptr[1], 64.0f);
  EXPECT_FLOAT_EQ(ptr[2], 139.0f);
  EXPECT_FLOAT_EQ(ptr[3], 154.0f);
}

TEST(OpsForward, ReluClampAndMean) {
  auto input = tensor_from_values({4}, {-2.0f, -0.5f, 1.5f, 5.0f});

  auto relu = Tensor::ops::relu(input);
  auto clipped = Tensor::ops::clamp(input, -1.0f, 2.0f);
  auto avg = Tensor::ops::mean(clipped);

  const auto *relu_ptr = static_cast<const float *>(relu.data());
  const auto *clamp_ptr = static_cast<const float *>(clipped.data());
  EXPECT_FLOAT_EQ(relu_ptr[0], 0.0f);
  EXPECT_FLOAT_EQ(relu_ptr[2], 1.5f);
  EXPECT_FLOAT_EQ(relu_ptr[3], 5.0f);
  EXPECT_FLOAT_EQ(clamp_ptr[0], -1.0f);
  EXPECT_FLOAT_EQ(clamp_ptr[3], 2.0f);
  EXPECT_FLOAT_EQ(static_cast<const float *>(avg.data())[0], 0.5f);
}

TEST(OpsForward, BiasAddBroadcastsAcrossRows) {
  auto value = tensor_from_values({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto bias = tensor_from_values({3}, {0.5f, -1.0f, 2.0f});

  auto out = Tensor::ops::bias_add(value, bias);
  const auto *ptr = static_cast<const float *>(out.data());

  EXPECT_FLOAT_EQ(ptr[0], 1.5f);
  EXPECT_FLOAT_EQ(ptr[1], 1.0f);
  EXPECT_FLOAT_EQ(ptr[2], 5.0f);
  EXPECT_FLOAT_EQ(ptr[3], 4.5f);
  EXPECT_FLOAT_EQ(ptr[4], 4.0f);
  EXPECT_FLOAT_EQ(ptr[5], 8.0f);
}
