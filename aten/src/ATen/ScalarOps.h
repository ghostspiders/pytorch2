#pragma once  // 防止头文件被重复包含

#include <c10/core/Scalar.h>  // 包含Scalar类型的定义
#include "ATen/Tensor.h"      // 包含Tensor类型的定义

// 这个命名空间放在c10中，因为我们使用ADL(参数依赖查找)来查找其中的函数
namespace c10 {

// 注意：这个函数本应该是Scalar::toTensor，但目前无法在不涉及派生类型(不属于核心部分)的情况下实现
// 将Scalar标量转换为Tensor张量
inline at::Tensor scalar_to_tensor(Scalar s) {
  if (s.isFloatingPoint()) {  // 如果是浮点类型
    // 创建CPU双精度张量
    return at::CPU(kDouble).scalarTensor(s);
  } else {  // 否则(整数类型)
    AT_ASSERT(s.isIntegral());  // 断言确保是整数类型
    // 创建CPU长整型张量
    return at::CPU(kLong).scalarTensor(s);
  }
}

}  // namespace c10结束