#include <benchmark/benchmark.h>
#include <cstdint>
#include "api/Api.hpp"
#include "tensor/Linear.hpp"
#include "tensor/Ops.hpp"

// Avoid broad using-directives to prevent symbol ambiguity on MSVC

static void BM_CreateZeros(benchmark::State& state) {
    int64_t n = state.range(0);
    for (auto _ : state) {
        auto t = ::Tensor::api::zeros<float>({n, n});
        benchmark::DoNotOptimize(t.data());
    }
    state.SetComplexityN(n * n);
}
BENCHMARK(BM_CreateZeros)->RangeMultiplier(2)->Range(64, 4096)->Complexity();

static void BM_Reshape(benchmark::State& state) {
    int64_t n = state.range(0);
    auto t = ::Tensor::api::zeros<float>({n, n});
    for (auto _ : state) {
        auto r = ::Tensor::api::reshape(t.as_dtensor(), {n*n});
        benchmark::DoNotOptimize(r.data());
    }
}
BENCHMARK(BM_Reshape)->RangeMultiplier(2)->Range(64, 4096);

static void BM_Permute(benchmark::State& state) {
    int64_t n = state.range(0);
    auto t = ::Tensor::api::zeros<float>({n, n, 4});
    for (auto _ : state) {
        auto p = ::Tensor::api::permute(t.as_dtensor(), {2,1,0});
        benchmark::DoNotOptimize(p.data());
    }
}
BENCHMARK(BM_Permute)->RangeMultiplier(2)->Range(64, 1024);

static void BM_Matmul(benchmark::State& state) {
    const int64_t m = state.range(0);
    const int64_t k = state.range(1);
    const int64_t n = state.range(2);
    auto lhs = ::Tensor::api::zeros<float>({m, k});
    auto rhs = ::Tensor::api::zeros<float>({k, n});
    ::Tensor::ops::fill(lhs.as_dtensor(), 1.0f);
    ::Tensor::ops::fill(rhs.as_dtensor(), 1.0f);

    for (auto _ : state) {
        auto out = ::Tensor::ops::matmul(lhs.as_dtensor(), rhs.as_dtensor());
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * m * k * n);
}
BENCHMARK(BM_Matmul)->Args({128, 256, 64})->Args({1, 768, 256});

static void BM_LinearForward(benchmark::State& state) {
    const int64_t batch = state.range(0);
    const int64_t in_features = state.range(1);
    const int64_t out_features = state.range(2);
    ::Tensor::nn::Linear linear(in_features, out_features);
    auto input = ::Tensor::api::zeros<float>({batch, in_features});
    ::Tensor::ops::fill(input.as_dtensor(), 1.0f);

    for (auto _ : state) {
        auto out = linear.forward(input.as_dtensor());
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * batch * in_features * out_features);
}
BENCHMARK(BM_LinearForward)->Args({1, 768, 256})->Args({1, 256, 32});

BENCHMARK_MAIN();


