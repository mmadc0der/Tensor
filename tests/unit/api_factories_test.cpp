#include <gtest/gtest.h>
#include "api/Api.hpp"

TEST(Api, MakeScalar) {
    auto t = Tensor::api::make_scalar<int>(42);
    ASSERT_EQ(t.numel(), 1);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_EQ(*t.data(), 42);
}

TEST(Api, ZerosOnesEmpty) {
    {
        auto t = Tensor::api::zeros<float>({2,3});
        ASSERT_EQ(t.numel(), 6);
        for (int i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(t.data()[i], 0.0f);
    }
    {
        auto t = Tensor::api::ones<double>({2});
        ASSERT_EQ(t.numel(), 2);
        for (int i = 0; i < 2; ++i) EXPECT_DOUBLE_EQ(t.data()[i], 1.0);
    }
    {
        auto t = Tensor::api::empty<int>({4});
        ASSERT_EQ(t.numel(), 4);
        EXPECT_TRUE(t.is_contiguous());
    }
}


