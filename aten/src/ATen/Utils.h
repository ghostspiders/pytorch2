#pragma once

#include "ATen/core/ATenGeneral.h"
#include <c10/core/StorageImpl.h>
#include "ATen/core/UndefinedTensorImpl.h"

#include <c10/core/ScalarType.h>
#include "ATen/Formatting.h"
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <sstream>
#include <typeinfo>
#include <numeric>

// 编译器特定属性处理
#if defined(__clang__)
// 忽略浮点除零检查（Clang）
#define __ubsan_ignore_float_divide_by_zero__ __attribute__((no_sanitize("float-divide-by-zero")))
// 忽略vptr检查（Clang）
#define __ubsan_ignore_vptr__ __attribute__((no_sanitize("vptr")))
#else
// 其他编译器不定义这些属性
#define __ubsan_ignore_float_divide_by_zero__
#define __ubsan_ignore_vptr__
#endif

namespace at {

// ASAN内存检测辅助函数声明
CAFFE2_API int _crash_if_asan(int);

/**
 * 检查Storage对象是否符合预期设备类型和数据类型
 * @param expr 要检查的Storage对象
 * @param name 参数名称（用于错误信息）
 * @param pos 参数位置（用于错误信息）
 * @param device_type 期望的设备类型
 * @param data_type 期望的数据类型
 * @return 通过检查的Storage引用
 * @throws 当设备类型或数据类型不匹配时抛出错误
 */
static inline const Storage& checked_storage(
    const Storage& expr,
    const char* name,
    int pos,
    DeviceType device_type,
    DataType data_type) {
  if (expr.device_type() != device_type) {
    AT_ERROR(
        "Expected object of device type ",
        device_type,
        " but got device type ",
        expr.data_ptr().device().type(),
        " for argument #",
        pos,
        " '",
        name,
        "'");
  }
  if (expr.dtype().id() != data_type) {
    AT_ERROR(
        "Expected object of data type ",
        data_type,
        " but got data type ",
        expr.dtype().id(),
        " for argument #",
        pos,
        " '",
        name,
        "'");
  }
  return expr;
}

/**
 * 解包Tensor并验证其属性和空值状态
 * @param expr 要检查的Tensor对象
 * @param name 参数名称（用于错误信息）
 * @param pos 参数位置（用于错误信息）
 * @param allowNull 是否允许空Tensor
 * @param backend 期望的后端类型
 * @param scalar_type 期望的标量类型
 * @return TensorImpl指针（当allowNull且未定义时返回nullptr）
 * @throws 当后端或标量类型不匹配时抛出错误
 */
static inline TensorImpl* checked_tensor_unwrap(const Tensor& expr, const char * name, int pos, bool allowNull, Backend backend, ScalarType scalar_type) {
  if(allowNull && !expr.defined()) {
    return nullptr;
  }
  if (tensorTypeIdToBackend(expr.type_id()) != backend) {
    AT_ERROR("Expected object of backend ", backend, " but got backend ", tensorTypeIdToBackend(expr.type_id()),
             " for argument #", pos, " '", name, "'");
  }
  if (expr.scalar_type() != scalar_type) {
    AT_ERROR("Expected object of scalar type ", scalar_type, " but got scalar type ", expr.scalar_type(),
             " for argument #", pos, " '", name, "'");
  }
  return expr.unsafeGetTensorImpl();
}

/**
 * 将Tensor列表解包为TensorImpl指针向量并验证属性
 * @param tensors Tensor列表
 * @param name 参数名称（用于错误信息）
 * @param pos 参数位置（用于错误信息）
 * @param backend 期望的后端类型
 * @param scalar_type 期望的标量类型
 * @return TensorImpl指针组成的向量
 * @throws 当任何元素的后端或标量类型不匹配时抛出详细错误
 */
static inline std::vector<TensorImpl*> checked_tensor_list_unwrap(ArrayRef<Tensor> tensors, const char * name, int pos, Backend backend, ScalarType scalar_type) {
  std::vector<TensorImpl*> unwrapped;
  unwrapped.reserve(tensors.size());
  for (unsigned int i = 0; i < tensors.size(); ++i) {
    const auto& expr = tensors[i];
    if (tensorTypeIdToBackend(expr.type_id()) != backend) {
      AT_ERROR("Expected object of backend ", backend, " but got backend ", tensorTypeIdToBackend(expr.type_id()),
               " for sequence element ", i, " in sequence argument at position #", pos, " '", name, "'");
    }
    if (expr.scalar_type() != scalar_type) {
      AT_ERROR("Expected object of scalar type ", scalar_type, " but got scalar type ", expr.scalar_type(),
               " for sequence element ", i , " in sequence argument at position #", pos, " '", name, "'");
    }
    unwrapped.emplace_back(expr.unsafeGetTensorImpl());
  }
  return unwrapped;
}

/**
 * 检查整数列表是否符合预期大小和内容
 * @tparam N 期望的列表大小
 * @param list 要检查的整数列表
 * @param name 参数名称（用于错误信息）
 * @param pos 参数位置（用于错误信息）
 * @param def 默认值（当列表为空时使用）
 * @return 包含N个元素的std::array<int64_t, N>
 * @throws 当列表大小不符合预期时抛出错误
 * 
 * 特殊处理：
 * 1. 空列表时使用默认值
 * 2. 单元素列表且N>1时，复制该元素填充整个数组
 */
template <size_t N>
std::array<int64_t, N> check_intlist(ArrayRef<int64_t> list, const char * name, int pos, ArrayRef<int64_t> def={}) {
  if (list.empty()) {
    list = def;
  }
  auto res = std::array<int64_t, N>();
  if (list.size() == 1 && N > 1) {
    res.fill(list[0]);
    return res;
  }
  if (list.size() != N) {
    AT_ERROR("Expected a list of ", N, " ints but got ", list.size(), " for argument #", pos, " '", name, "'");
  }
  std::copy_n(list.begin(), N, res.begin());
  return res;
}

/// 计算整数列表的总和
inline int64_t sum_intlist(ArrayRef<int64_t> list) {
  return std::accumulate(list.begin(), list.end(), 0ll);
}

/// 计算整数列表的乘积
inline int64_t prod_intlist(ArrayRef<int64_t> list) {
  return std::accumulate(list.begin(), list.end(), 1ll, std::multiplies<int64_t>());
}

} // namespace at