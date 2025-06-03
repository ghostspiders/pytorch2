#pragma once

#include "ATen/Tensor.h"
#include "c10/util/Exception.h"

#include <functional>
#include <sstream>
#include <tuple>

namespace at {

// 声明两个核心形状推断函数（定义在其他文件中）
CAFFE2_API std::vector<int64_t> infer_size(IntList a, IntList b);  // 推断广播后的尺寸
CAFFE2_API std::tuple<std::vector<int64_t>, std::vector<int64_t>>
inferExpandGeometry(  // 推断扩展后的形状和步长
    IntList tensor_sizes,
    IntList tensor_strides,
    IntList sizes);

// 检查张量是否已定义的辅助函数
// 使用reference_wrapper避免不必要的张量拷贝
inline void check_defined(std::initializer_list<std::reference_wrapper<const Tensor>> tensors, const char *api_name) {
  for (auto& t : tensors) {
    if (!t.get().defined()) {  // 如果发现未定义的张量
      AT_ERROR(api_name, "(...) called with an undefined Tensor");  // 抛出错误
    }
  }
}

/* 就地扩展函数系列：将张量扩展至目标张量的形状 */

// 单张量扩展版本
inline std::tuple<Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand) {
  if (tensor.sizes().equals(to_expand.sizes())) {  // 如果形状已匹配
    return std::make_tuple(to_expand);  // 直接返回原张量
  }
  // 否则调用expand进行隐式广播（implicit=true表示允许广播语义）
  return std::make_tuple(to_expand.expand(tensor.sizes(), /*implicit=*/true)); 
}

// 带API名称检查的版本（便于错误追踪）
inline std::tuple<Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand, const char *api_name) {
  check_defined({tensor, to_expand}, api_name);  // 前置检查
  return expand_inplace(tensor, to_expand);
}

// 双张量扩展版本
inline std::tuple<Tensor, Tensor> expand_inplace(const Tensor &tensor, 
                                                const Tensor &to_expand1, 
                                                const Tensor &to_expand2) {
  if (tensor.sizes().equals(to_expand1.sizes()) && 
      tensor.sizes().equals((to_expand2.sizes()))) {
    return std::make_tuple(to_expand1, to_expand2);
  }
  // 两个张量都扩展到目标形状
  return std::make_tuple(
      to_expand1.expand(tensor.sizes(), /*implicit=*/true),
      to_expand2.expand(tensor.sizes(), /*implicit=*/true));
}

// 带检查的双张量版本
inline std::tuple<Tensor, Tensor> expand_inplace(const Tensor &tensor, 
                                                const Tensor &to_expand1, 
                                                const Tensor &to_expand2,
                                                const char *api_name) {
  check_defined({tensor, to_expand1, to_expand2}, api_name);
  return expand_inplace(tensor, to_expand1, to_expand2);
}

/* 异地扩展函数系列：自动推断广播后的形状 */

// 双张量异地扩展
inline std::tuple<Tensor, Tensor> expand_outplace(const Tensor &to_expand1, 
                                                 const Tensor &to_expand2) {
  if (to_expand1.sizes().equals(to_expand2.sizes())) {
    return std::make_tuple(to_expand1, to_expand2);
  }
  // 自动推断广播后的形状
  auto expanded_size = infer_size(to_expand1.sizes(), to_expand2.sizes());
  return std::make_tuple(
      to_expand1.expand(expanded_size, /*implicit=*/true),
      to_expand2.expand(expanded_size, /*implicit=*/true));
}

// 带检查的版本
inline std::tuple<Tensor, Tensor> expand_outplace(const Tensor &to_expand1,
                                                 const Tensor &to_expand2, 
                                                 const char *api_name) {
  check_defined({to_expand1, to_expand2}, api_name);
  return expand_outplace(to_expand1, to_expand2);
}

// 三张量异地扩展
inline std::tuple<Tensor, Tensor, Tensor> expand_outplace(const Tensor &to_expand1,
                                                         const Tensor &to_expand2,
                                                         const Tensor &to_expand3) {
  if (to_expand1.sizes().equals(to_expand2.sizes()) && 
      to_expand1.sizes().equals(to_expand3.sizes())) {
    return std::make_tuple(to_expand1, to_expand2, to_expand3);
  }
  // 分两步推断广播形状
  auto expanded_size12 = infer_size(to_expand1.sizes(), to_expand2.sizes());
  auto expanded_size = infer_size(expanded_size12, to_expand3.sizes());
  return std::make_tuple(
      to_expand1.expand(expanded_size, /*implicit=*/true),
      to_expand2.expand(expanded_size, /*implicit=*/true),
      to_expand3.expand(expanded_size, /*implicit=*/true));
}

// 带检查的三张量版本
inline std::tuple<Tensor, Tensor, Tensor> expand_outplace(const Tensor &to_expand1,
                                                         const Tensor &to_expand2,
                                                         const Tensor &to_expand3,
                                                         const char *api_name) {
  check_defined({to_expand1, to_expand2, to_expand3}, api_name);
  return expand_outplace(to_expand1, to_expand2, to_expand3);
}

/* 指定目标形状的扩展 */

inline std::tuple<Tensor> expand_size(const Tensor &to_expand, IntList sizes) {
  if(to_expand.sizes().equals(sizes)) {
    return std::make_tuple(to_expand);
  }
  return std::make_tuple(to_expand.expand(sizes, /*implicit=*/true));
}

inline std::tuple<Tensor> expand_size(const Tensor &to_expand, 
                                     IntList sizes, 
                                     const char *api_name) {
  check_defined({to_expand}, api_name);
  return expand_size(to_expand, sizes);
}

/* 张量列表的广播扩展 */

inline std::vector<Tensor> expand_outplace(TensorList to_expand) {
  // 处理可能包含未定义张量的列表
  bool first = true;
  std::vector<int64_t> sizes;
  
  // 第一步：计算所有张量的广播后形状
  for (size_t i = 0; i < to_expand.size(); ++i) {
    if (!to_expand[i].defined()) continue;  // 跳过未定义张量
    
    if (first) {
      sizes = to_expand[i].sizes().vec();  // 初始化形状
      first = false;
    } else {
      sizes = infer_size(sizes, to_expand[i].sizes());  // 逐步推断
    }
  }

  // 第二步：执行实际扩展操作
  std::vector<Tensor> result(to_expand.size());
  for (size_t i = 0; i < to_expand.size(); ++i) {
    if (!to_expand[i].defined()) continue;
    
    if (to_expand[i].sizes().equals(sizes)) {
      result[i] = to_expand[i];  // 无需扩展
    } else {
      result[i] = to_expand[i].expand(sizes, /*implicit=*/true);  // 执行扩展
    }
  }
  return result;
}

/* 广播相关的工具函数 */

// 通过求和将张量缩减到目标形状
// 前置条件：is_expandable_to(shape, tensor.sizes())必须为true
static inline Tensor sum_to(Tensor tensor, const IntList shape) {
  if (shape.size() == 0) {  // 标量情况
    return tensor.sum();
  }
  
  // 计算需要缩减的维度
  c10::SmallVector<int64_t, 8> reduce_dims;
  const at::IntList sizes = tensor.sizes();
  const int64_t leading_dims = sizes.size() - shape.size();
  
  // 前导维度必须缩减
  for (int64_t i = 0; i < leading_dims; ++i) {
    reduce_dims.push_back(i);
  }
  
  // 处理需要广播的维度（size=1的目标维度）
  for (int64_t i = leading_dims; i < static_cast<int64_t>(sizes.size()); ++i) {
    if (shape[i - leading_dims] == 1 && sizes[i] > 1) {
      reduce_dims.push_back(i);
    }
  }
  
  // 执行求和操作（保持维度以便后续view）
  if (!reduce_dims.empty()) {
    tensor = tensor.sum(reduce_dims, /*keepdim=*/true);
  }
  
  // 调整形状（处理前导维度情况）
  return leading_dims > 0 ? tensor.view(shape) : tensor;
}

// 检查形状是否可广播到目标形状
static inline bool is_expandable_to(IntList shape, IntList desired) {
  int ndim = shape.size();
  int target_dim = desired.size();
  
  if (ndim > target_dim) {  // 维度数超过目标
    return false;
  }
  
  // 从尾部开始逐维度检查
  for (int i = 0; i < ndim; i++) {
    int64_t size = shape[ndim - i - 1];
    int64_t target = desired[target_dim - i - 1];
    
    // 不满足广播规则（size != target 且 size != 1）
    if (size != target && size != 1) {
      return false;
    }
  }
  return true;
}

} // namespace at