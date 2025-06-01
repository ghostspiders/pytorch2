#pragma once

#include <c10/core/dispatch/DeviceId.h>
#include <c10/core/dispatch/LayoutId.h>
#include <c10/util/typeid.h>

#include <vector>
#include <functional>
#include <sstream>
#include <c10/util/Array.h>

namespace c10 {

namespace details {

// 张量参数调度键结构体，表示用于函数调度的张量特征
// 注意：这个调度键结构还不是最终版本，会继续修改，不要依赖它
struct TensorParameterDispatchKey final {
  DeviceTypeId deviceTypeId;    // 设备类型ID (CPU/GPU等)
  LayoutId layoutId;            // 布局ID (如密集/稀疏张量)
  // TODO 将这个TypeIdentifier移到c10命名空间
  caffe2::TypeIdentifier dataType;  // 数据类型 (如float/int等)
};

// 重载==操作符，比较两个TensorParameterDispatchKey是否相等
inline constexpr bool operator==(const TensorParameterDispatchKey& lhs, const TensorParameterDispatchKey& rhs) {
  return lhs.deviceTypeId == rhs.deviceTypeId && lhs.layoutId == rhs.layoutId && lhs.dataType == rhs.dataType;
}

// 重载<<操作符，用于输出TensorParameterDispatchKey
inline std::ostream& operator<<(std::ostream& stream, const TensorParameterDispatchKey& key) {
  return stream << "TensorKey(" << key.deviceTypeId << ", " << key.layoutId.value() << ", " << key.dataType << ")";
}

}  // namespace details
}  // namespace c10

namespace std {
  // 为TensorParameterDispatchKey特化std::hash
  template<>
  struct hash<c10::details::TensorParameterDispatchKey> {
    // TODO 常量表达式哈希计算
    size_t operator()(const c10::details::TensorParameterDispatchKey& obj) const {
      return std::hash<c10::DeviceTypeId>()(obj.deviceTypeId) ^ 
             std::hash<c10::LayoutId>()(obj.layoutId) ^ 
             std::hash<caffe2::TypeIdentifier>()(obj.dataType);
    }
  };
}  // namespace std

namespace c10 {
/**
 * 调度键(DispatchKey)编码函数调用参数的运行时类型标识，
 * 指定可以动态调度的特征。
 *
 * 直观地说，给定一个函数签名如f(Tensor, int)，参数的
 * 有效调度键可能是[CPUFloatTensor](注意'f'不包含在
 * 调度键中，且'int'的运行时类型不考虑调度(因为它是平凡的)。
 *
 * 调度键支持相等性测试并且是可哈希的。
 *
 * @tparam num_dispatch_args 可调度参数的数量
 */
template<size_t num_dispatch_args>
struct DispatchKey final {
  guts::array<details::TensorParameterDispatchKey, num_dispatch_args> argTypes;  // 参数类型数组
};

// 重载==操作符，比较两个DispatchKey是否相等
template<size_t num_dispatch_args>
inline constexpr bool operator==(const DispatchKey<num_dispatch_args> &lhs, const DispatchKey<num_dispatch_args>& rhs) {
  // TODO: 使用AVX指令集来加速这个相等性测试
  return lhs.argTypes == rhs.argTypes;
}

// 重载<<操作符，用于输出DispatchKey
template<size_t num_dispatch_args>
inline std::ostream& operator<<(std::ostream& stream, const DispatchKey<num_dispatch_args>& key) {
  stream << "DispatchKey(";
  if (num_dispatch_args > 0) {
      stream << "DispatchKey(" << key.argTypes[0];
      for (size_t i = 1; i < num_dispatch_args; ++i) {
          stream << ", " << key.argTypes[i];
      }
      stream << ")";
  }
  return stream << ")";
}

}  // namespace c10

namespace std {
  // 为DispatchKey特化std::hash
  template<size_t num_dispatch_args>
  struct hash<c10::DispatchKey<num_dispatch_args>> {
    // TODO 常量表达式哈希计算
    size_t operator()(const c10::DispatchKey<num_dispatch_args>& obj) const {
      size_t hash_value = 0;
      for (const auto& argType : obj.argTypes) {
        hash_value *= 10883; // 使用质数进行哈希混合
        hash_value += std::hash<c10::details::TensorParameterDispatchKey>()(argType);
      }
      return hash_value;
    }
  };
}  // namespace std