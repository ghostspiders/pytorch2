#pragma once

#include "ATen/ATen.h"

namespace at {
namespace native {

// 注意：[有符号和无符号类型之间的隐式转换]
// C和C++有一套隐式转换规则：
// 1. 将有符号整型值转换为无符号整型值总是有效的（采用模运算处理）
// 2. 但将负浮点数值转换为无符号整型是未定义行为(UB)！
// 这意味着：
//   (double)-1 -> (int64_t)-1 -> (uint8_t)255 是安全的
//   (double)-1 -> (uint8_t)<任意值> 是未定义行为
// 这个设计非常不合理，我们不应该遵循这些规则。
//
// 下面的结构体确保所有无符号类型（目前只有uint8_t）在转换时
// 都会通过int64_t进行中间转换，保证负值能正确回绕处理。
//
// 注意：从double到有符号整型的转换中，如果值超出目标类型范围也是UB，
// 但这个问题不能简单地通过int64_t中间转换解决，因为当值过大时
// int64_t -> <较小有符号类型> 的转换仍然是UB。
// 这种情况我们只能接受，但它比上述无符号转换的问题要少得多。
//
// 参考：
//   https://en.cppreference.com/w/cpp/language/implicit_conversion
//   关键段落："浮点数-整型转换"

// 类型转换模板
template <typename T>
struct inter_copy_type {
  using type = T;  // 默认情况下直接使用原类型
};

// uint8_t类型的特化
template <>
struct inter_copy_type<uint8_t> {
  using type = int64_t;  // 对uint8_t使用int64_t作为中间转换类型
};

// 类型别名模板
template <typename T>
using inter_copy_type_t = typename inter_copy_type<T>::type;

} // namespace native
} // namespace at