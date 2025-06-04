#pragma once
#include "ATen/Config.h"
#include "ATen/core/Half.h"

// 这段代码实现了一个‌累加类型选择器‌（AccumulateType），用于在数值计算中根据输入类型（T）和设备环境（CPU/CUDA）自动选择最优的累加类型，以平衡精度与性能

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_fp16.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif

namespace at {

template <typename T, bool is_cuda>
struct AccumulateType { };

#if defined(__CUDACC__) || defined(__HIPCC__)
template <> struct AccumulateType<half, true> { using type = float; };
#endif
template <> struct AccumulateType<Half, true> { using type = float; };
template <> struct AccumulateType<float, true> { using type = float; };
template <> struct AccumulateType<double, true> { using type = double; };
template <> struct AccumulateType<int8_t, true> { using type = int64_t; };
template <> struct AccumulateType<uint8_t, true> { using type = int64_t; };
template <> struct AccumulateType<char, true> { using type = int64_t; };
template <> struct AccumulateType<int16_t, true> { using type = int64_t; };
template <> struct AccumulateType<int32_t, true> { using type = int64_t; };
template <> struct AccumulateType<int64_t, true> { using type = int64_t; };
template <> struct AccumulateType<float, false> { using type = double; };
template <> struct AccumulateType<double, false> { using type = double; };
template <> struct AccumulateType<int8_t, false> { using type = int64_t; };
template <> struct AccumulateType<uint8_t, false> { using type = int64_t; };
template <> struct AccumulateType<char, false> { using type = int64_t; };
template <> struct AccumulateType<int16_t, false> { using type = int64_t; };
template <> struct AccumulateType<int32_t, false> { using type = int64_t; };
template <> struct AccumulateType<int64_t, false> { using type = int64_t; };

template<typename T, bool is_cuda>
using acc_type = typename AccumulateType<T, is_cuda>::type;

}  // namespace at
