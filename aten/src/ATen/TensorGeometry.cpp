#include <ATen/TensorGeometry.h>  // 包含张量几何形状相关定义
#include <ATen/TensorUtils.h>     // 包含张量工具函数
#include <ATen/ATen.h>            // ATen核心头文件

namespace at {

/**
 * 检查张量是否是连续存储的(contiguous)
 * 
 * 张量连续存储意味着:
 * 1. 元素在内存中顺序排列
 * 2. 步长(strides)满足特定条件:
 *    - 最后一个维度的步长为1
 *    - 每个维度的步长等于后一个维度步长×后一个维度大小
 * 
 * 特殊情况:
 * - 当张量元素数量(numel)为0时，视为连续存储
 * 
 * @return bool - 返回true表示张量是连续存储的，false表示不是
 */
bool TensorGeometry::is_contiguous() const {
  // 处理空张量的特殊情况
  if (numel_ == 0) {
    return true;  // 空张量总是视为连续的
  }
  
  // 调用ATen工具函数检查连续性
  return at::geometry_is_contiguous(sizes_, strides_);
}

} // namespace at
