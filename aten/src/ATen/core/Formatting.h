#pragma once  // 防止头文件被重复包含

#include <c10/core/Scalar.h>      // 包含Scalar类型的定义
#include <ATen/core/Tensor.h>     // 包含Tensor核心定义
#include <ATen/core/TensorMethods.h> // 包含Tensor方法
#include <ATen/core/Type.h>       // 包含Type类型定义
#include <iostream>               // 标准输入输出流

namespace c10 {
// 声明Backend类型的输出流操作符重载
// CAFFE2_API宏用于指定符号的可见性
CAFFE2_API std::ostream& operator<<(std::ostream& out, Backend b);
}

namespace at {

// 声明Type类型的输出流操作符重载
CAFFE2_API std::ostream& operator<<(std::ostream& out, const Type& t);
// 声明Tensor打印函数，可以指定每行最大字符数
CAFFE2_API std::ostream& print(
    std::ostream& stream,
    const Tensor& tensor,
    int64_t linesize);

// 内联的Tensor输出流操作符重载，默认每行80个字符
static inline std::ostream& operator<<(std::ostream & out, const Tensor & t) {
  return print(out, t, 80);  // 调用print函数，默认行宽80
}

// 内联的Tensor打印函数，默认输出到std::cout，行宽80
static inline void print(const Tensor & t, int64_t linesize=80) {
  print(std::cout, t, linesize);
}

// 内联的Scalar类型输出流操作符重载
// 根据Scalar是否为浮点类型选择不同的输出方式
static inline std::ostream& operator<<(std::ostream & out, Scalar s) {
  // 如果是浮点数则转换为double输出，否则转换为long输出
  return out << (s.isFloatingPoint() ? s.toDouble() : s.toLong());
}

} // namespace at