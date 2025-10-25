#include <benchmark/benchmark.h>
#include "api/Api.hpp"

using namespace Tensor;
using namespace Tensor::api;

static void BM_CreateZeros(benchmark::State& state) {
    int64_t n = state.range(0);
    for (auto _ : state) {
        auto t = zeros<float>({n, n});
        benchmark::DoNotOptimize(t.data());
    }
    state.SetComplexityN(n * n);
}
BENCHMARK(BM_CreateZeros)->RangeMultiplier(2)->Range(64, 4096)->Complexity();

static void BM_Reshape(benchmark::State& state) {
    int64_t n = state.range(0);
    auto t = zeros<float>({n, n});
    for (auto _ : state) {
        auto r = reshape(t.as_dtensor(), {n*n});
        benchmark::DoNotOptimize(r.data());
    }
}
BENCHMARK(BM_Reshape)->RangeMultiplier(2)->Range(64, 4096);

static void BM_Permute(benchmark::State& state) {
    int64_t n = state.range(0);
    auto t = zeros<float>({n, n, 4});
    for (auto _ : state) {
        auto p = permute(t.as_dtensor(), {2,1,0});
        benchmark::DoNotOptimize(p.data());
    }
}
BENCHMARK(BM_Permute)->RangeMultiplier(2)->Range(64, 1024);

BENCHMARK_MAIN();


