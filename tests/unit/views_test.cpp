#include <gtest/gtest.h>
#include "api/Api.hpp"

using namespace Tensor;
using namespace Tensor::api;

TEST(Views, ReshapeContiguousSameNumel) {
    auto t = zeros<float>({2,3});
    auto &dt = t.as_dtensor();
    auto r = reshape(dt, {3,2});
    EXPECT_EQ(r.shape()[0], 3);
    EXPECT_EQ(r.shape()[1], 2);
    EXPECT_TRUE(r.is_contiguous());
}

TEST(Views, PermuteBasic) {
    auto t = zeros<float>({2,3,4});
    auto &dt = t.as_dtensor();
    auto p = permute(dt, {2,1,0});
    EXPECT_EQ(p.rank(), 3);
    EXPECT_FALSE(p.is_contiguous());
    EXPECT_EQ(p.shape()[0], 4);
    EXPECT_EQ(p.shape()[1], 3);
    EXPECT_EQ(p.shape()[2], 2);
}


