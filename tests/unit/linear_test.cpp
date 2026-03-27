#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include "api/Api.hpp"
#include "tensor/Linear.hpp"

namespace {

Tensor::DTensor dataset_tensor(const std::vector<int64_t> &shape,
                               const std::vector<float> &values) {
  auto tensor = Tensor::api::zeros(shape, Tensor::DType::f32, false);
  std::copy(values.begin(), values.end(), static_cast<float *>(tensor.data()));
  return tensor;
}

float scalar_value(const Tensor::DTensor &tensor) {
  return static_cast<const float *>(tensor.data())[0];
}

} // namespace

TEST(Linear, ProducesExpectedOutputShape) {
  Tensor::nn::Linear linear(3, 2);
  auto input = dataset_tensor({4, 3}, {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f,
  });

  auto output = linear.forward(input);
  EXPECT_EQ(output.shape()[0], 4);
  EXPECT_EQ(output.shape()[1], 2);
}

TEST(Linear, BackwardPopulatesWeightAndBiasGradients) {
  Tensor::nn::Linear linear(1, 1);
  auto input = dataset_tensor({2, 1}, {1.0f, 2.0f});
  auto target = dataset_tensor({2, 1}, {3.0f, 5.0f});

  auto prediction = linear.forward(input);
  auto loss = Tensor::ops::mse_loss(prediction, target);
  Tensor::ops::backward(loss);

  ASSERT_NE(linear.weight().grad(), nullptr);
  ASSERT_NE(linear.bias().grad(), nullptr);
  EXPECT_EQ(linear.weight().grad()->shape()[0], 1);
  EXPECT_EQ(linear.weight().grad()->shape()[1], 1);
  EXPECT_EQ(linear.bias().grad()->shape()[0], 1);
}

TEST(Linear, LearnsSimpleAffineMapping) {
  Tensor::nn::Linear linear(1, 1);
  Tensor::nn::SGD optimizer(0.1f);

  auto input = dataset_tensor({4, 1}, {-2.0f, -1.0f, 1.0f, 2.0f});
  auto target = dataset_tensor({4, 1}, {-5.0f, -3.0f, 1.0f, 3.0f});

  const auto initial_loss = scalar_value(Tensor::ops::mse_loss(linear.forward(input), target));

  for (int step = 0; step < 200; ++step) {
    optimizer.zero_grad(linear.parameters());
    auto prediction = linear.forward(input);
    auto loss = Tensor::ops::mse_loss(prediction, target);
    Tensor::ops::backward(loss);
    optimizer.step(linear.parameters());
  }

  const auto final_prediction = linear.forward(input);
  const auto final_loss = scalar_value(Tensor::ops::mse_loss(final_prediction, target));
  const auto *pred_ptr = static_cast<const float *>(final_prediction.data());

  EXPECT_LT(final_loss, initial_loss * 0.01f);
  EXPECT_NEAR(pred_ptr[0], -5.0f, 0.25f);
  EXPECT_NEAR(pred_ptr[3], 3.0f, 0.25f);
}
