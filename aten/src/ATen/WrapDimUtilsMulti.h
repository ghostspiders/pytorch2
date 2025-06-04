#pragma once

#include "ATen/core/TensorImpl.h"  // 包含TensorImpl类定义
#include "ATen/WrapDimUtils.h"     // 包含维度包装工具函数
#include <sstream>                 // 字符串流处理
#include <bitset>                  // 位集合容器

namespace at {

// 此文件独立存在是为了解决Windows平台上bitset与运算符重载的奇怪交互问题

constexpr size_t dim_bitset_size = 64;  // 定义维度位集合的最大尺寸为64位

/**
 * 将维度列表转换为位集合(bitset)表示
 * @param dims   要处理的维度列表(IntList类型)
 * @param ndims  张量的总维度数
 * @return       表示已选维度的位集合
 *
 * 功能说明:
 * 1. 检查总维度数是否超过支持的最大值(64维)
 * 2. 将每个维度包装为正索引(处理负维度)
 * 3. 检查是否有重复维度
 * 4. 在位集合中标记选中的维度
 */
static inline std::bitset<dim_bitset_size> dim_list_to_bitset(IntList dims, int64_t ndims) {
  // 检查维度数是否超过最大值
  AT_CHECK(ndims <= (int64_t) dim_bitset_size, 
           "only tensors with up to ", dim_bitset_size, " dims are supported");
  
  std::bitset<dim_bitset_size> seen;  // 创建位集合(默认所有位为0)
  
  for (size_t i = 0; i < dims.size(); i++) {
    // 包装维度(处理负值)
    size_t dim = maybe_wrap_dim(dims[i], ndims);
    
    // 检查维度是否重复
    AT_CHECK(!seen[dim], "dim ", dim, " appears multiple times in the list of dims");
    
    // 在位集合中标记该维度
    seen[dim] = true;
  }
  
  return seen;
}

} // namespace at