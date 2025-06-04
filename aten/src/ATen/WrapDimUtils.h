#pragma once

#include "ATen/core/WrapDimMinimal.h"  // 包含维度包装的基础函数
#include "ATen/core/TensorImpl.h"      // 包含TensorImpl类定义

namespace at {

// 包装单个维度，基于TensorImpl的维度数
static inline int64_t maybe_wrap_dim(int64_t dim, TensorImpl *tensor) {
  return maybe_wrap_dim(dim, tensor->dim());  // 调用基础函数，传入张量的维度数
}

// 包装单个维度，基于TensorList中第一个张量的维度数
static inline int64_t maybe_wrap_dim(int64_t dim, TensorList tensors) {
  if (tensors.size() == 0) {
    // 空TensorList无法包装，依赖底层实现抛出错误
    return dim;
  }
  return maybe_wrap_dim(dim, tensors[0].dim());  // 使用第一个张量的维度数
}

// 包装单个维度，基于张量大小列表中的第一个张量维度数
static inline int64_t maybe_wrap_dim(int64_t dim, const std::vector<std::vector<int64_t>> & tensor_sizes) {
  if (tensor_sizes.size() == 0) {
    // 空列表无法包装，依赖底层实现抛出错误
    return dim;
  }
  return maybe_wrap_dim(dim, tensor_sizes[0].size());  // 使用第一个张量的大小维度数
}

// 包装多个维度，基于给定的维度表达式(dim_post_expr)
static inline void maybe_wrap_dims(std::vector<int64_t>& dims, int64_t dim_post_expr) {
  if (dim_post_expr <= 0) {
    dim_post_expr = 1; // 确保范围至少为[-1, 0]
  }
  int64_t min = -dim_post_expr;          // 最小允许值
  int64_t max = dim_post_expr - 1;       // 最大允许值
  for (auto& dim : dims) {
    // 检查维度是否在合法范围内
    AT_CHECK(
        dim >= min && dim <= max,
        "Dimension out of range (expected to be in range of [",
        min, ", ", max, "], but got ", dim, ")");
    // 处理负维度
    if (dim < 0) dim += dim_post_expr;
  }
}

// 为保持向后兼容性而保留的特殊维度包装逻辑
// 传统行为：只跳过大小为[0]的空张量
static inline int64_t legacy_cat_wrap_dim(int64_t dim, const std::vector<std::vector<int64_t>>& tensor_sizes) {
  for (auto& sizes : tensor_sizes) {
    if (sizes == std::vector<int64_t>({0})) {
      continue;  // 跳过大小为[0]的张量
    }
    return maybe_wrap_dim(dim, sizes.size());  // 使用第一个非[0]张量的维度数
  }
  return dim;  // 如果所有张量都是[0]，直接返回原维度
}

// 同上，但接受TensorList作为输入
static inline int64_t legacy_cat_wrap_dim(int64_t dim, TensorList tensors) {
  for (auto& tensor : tensors) {
    if (tensor.dim() == 1 && tensor.sizes()[0] == 0) {
      continue;  // 跳过1维且大小为0的张量
    }
    return maybe_wrap_dim(dim, tensor.dim());  // 使用第一个符合条件的张量维度
  }
  return dim;  // 如果所有张量都符合跳过条件，直接返回原维度
}

// 包装向量中的所有维度
static inline void wrap_all_dims(std::vector<int64_t>& dims_to_wrap, int64_t tensor_total_dims) {
  for (size_t i = 0; i < dims_to_wrap.size(); i++) {
    // 对每个维度单独包装
    dims_to_wrap[i] = maybe_wrap_dim(dims_to_wrap[i], tensor_total_dims);
  }
}

} // namespace at