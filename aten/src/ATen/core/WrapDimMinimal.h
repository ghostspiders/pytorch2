// 防止头文件重复包含的编译器指令
#pragma once

// 引入异常处理工具
#include "c10/util/Exception.h"

namespace at {  // 属于 at 命名空间

/**
 * 处理维度索引的包装/规范化
 * @param dim 输入的维度索引（可能为负数）
 * @param dim_post_expr 实际的最大维度值（例如 tensor.dim()）
 * @param wrap_scalar 是否允许标量（0维）的包装处理
 * @return 规范化后的正数维度索引
 */
static inline int64_t maybe_wrap_dim(
    int64_t dim, 
    int64_t dim_post_expr, 
    bool wrap_scalar = true) 
{
  // 处理0维张量的特殊情况
  if (dim_post_expr <= 0) {
    // 如果不允许包装标量则抛出异常
    AT_CHECK(wrap_scalar, 
             "dimension specified as ", dim, 
             " but tensor has no dimensions");
    // 强制设置为1维，使后续计算范围变为[-1, 0]
    dim_post_expr = 1;  
  }

  // 计算合法维度的取值范围
  int64_t min = -dim_post_expr;       // 最小允许值（负索引）
  int64_t max = dim_post_expr - 1;    // 最大允许值
  
  // 检查维度是否越界
  AT_CHECK(
      dim >= min && dim <= max,
      "Dimension out of range (expected to be in range of [",
      min, ", ", max, "], but got ", dim, ")");

  // 处理负数索引：将 -1 转换为最后一维
  if (dim < 0) dim += dim_post_expr;
  
  return dim;  // 返回规范化后的维度
}

}  // namespace at