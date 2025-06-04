#pragma once

// 包含ATen工具函数
#include "ATen/Utils.h"

// 包含随机数生成器基类
#include "ATen/core/Generator.h"

// 包含异常处理工具
#include "c10/util/Exception.h"

namespace at {

/**
 * @brief 检查并转换生成器类型
 * 
 * 该函数用于验证生成器对象是否为特定类型，并在验证失败时抛出错误
 * 
 * @tparam T 期望的目标生成器类型（如CPUGeneratorImpl, CUDAGeneratorImpl等）
 * @param expr 需要检查的生成器指针（可能为null）
 * @param defaultValue 当expr为空时使用的默认生成器
 * @return T* 类型转换后的生成器指针
 * 
 * 功能说明：
 * 1. 如果输入指针expr为空，则使用defaultValue替代
 * 2. 尝试将生成器对象动态转换为目标类型T
 * 3. 转换成功：返回类型正确的指针
 * 4. 转换失败：抛出类型不匹配错误
 * 
 * 使用场景：
 * 在需要特定类型生成器的操作中（如CPU/CUDA随机数生成），
 * 确保获取到正确类型的生成器对象
 */
template <typename T>
static inline T* check_generator(Generator* expr, Generator* defaultValue) {
  // 处理空指针情况：使用默认生成器
  if (!expr)
    expr = defaultValue;
  
  // 尝试动态类型转换
  if (auto result = dynamic_cast<T*>(expr))
    return result;
  
  // 类型转换失败时抛出详细错误信息
  AT_ERROR("Expected a '", typeid(T).name(), 
           "' but found '", typeid(expr).name(), "'");
}

} // namespace at