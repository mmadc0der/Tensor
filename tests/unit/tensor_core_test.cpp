#include <gtest/gtest.h>

#include "api/Api.hpp"
#include "tensor/Tensor.hpp"

TEST(Core, DTypeSize) {
    EXPECT_EQ(Tensor::dtype_size(Tensor::DType::f32), 4u);
    EXPECT_EQ(Tensor::dtype_size(Tensor::DType::f64), 8u);
    EXPECT_EQ(Tensor::dtype_size(Tensor::DType::i32), 4u);
    EXPECT_EQ(Tensor::dtype_size(Tensor::DType::i64), 8u);
}

TEST(Core, DefaultStrides) {
    {
        std::vector<int64_t> shape{4};
        auto s = Tensor::default_strides(shape);
        ASSERT_EQ(s.size(), 1u);
        EXPECT_EQ(s[0], 1);
    }
    {
        std::vector<int64_t> shape{2,3,5};
        auto s = Tensor::default_strides(shape);
        ASSERT_EQ(s.size(), 3u);
        EXPECT_EQ(s[2], 1);
        EXPECT_EQ(s[1], 5);
        EXPECT_EQ(s[0], 3*5);
    }
}

TEST(Core, DTensorBasicConstruct) {
    auto st = Tensor::make_host_storage(4 * 8, 64);
    std::vector<int64_t> shape{4,8};
    auto stride = Tensor::default_strides(shape);
    Tensor::DTensor dt{std::move(st), shape, stride, 0, Tensor::DType::f32, true, false};
    EXPECT_EQ(dt.rank(), 2);
    EXPECT_EQ(dt.numel(), 32);
    EXPECT_TRUE(dt.is_contiguous());
    EXPECT_FALSE(dt.requires_grad());
}

TEST(Core, RequiresGradStateIsSharedAcrossCopies) {
    auto base = Tensor::api::zeros({2, 2}, Tensor::DType::f32, true);
    Tensor::DTensor alias = base;

    EXPECT_TRUE(base.requires_grad());
    EXPECT_TRUE(alias.requires_grad());
    EXPECT_TRUE(base.is_leaf());
    EXPECT_TRUE(alias.is_leaf());
}

