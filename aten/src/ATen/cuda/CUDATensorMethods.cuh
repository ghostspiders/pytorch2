#pragma once

#include "ATen/Tensor.h"
#include "ATen/core/Half.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace at {  // ATen库的核心命名空间

// 模板特化：获取Tensor中半精度浮点(__half)数据的指针
template <>
inline __half* Tensor::data() const {
  // 通过reinterpret_cast将Half类型指针转换为__half类型指针
  return reinterpret_cast<__half*>(data<Half>());
}

} // namespace at
