#pragma once

#include <ATen/Type.h>
#include <ATen/core/Half.h>
#include <c10/util/Exception.h>

// 基础宏定义：根据给定的枚举类型选择对应的C++类型，并执行后续代码
#define AT_PRIVATE_CASE_TYPE(enum_type, type, ...) \
  case enum_type: {                                \  // 匹配特定枚举类型
    using scalar_t = type;                         \  // 定义类型别名
    return __VA_ARGS__();                          \  // 执行可变参数中的代码
  }

// 分派浮点类型（float/double）的宏
// TYPE: 输入类型对象
// NAME: 操作名称（用于错误提示）
// ...: 要执行的代码块
#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                          \
  [&] {                                                                      \  // 使用lambda捕获上下文
    const at::Type& the_type = TYPE;                                         \  // 获取类型引用
    switch (the_type.scalarType()) {                                         \  // 根据标量类型进行分派
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \  // 处理double类型
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \  // 处理float类型
      default:                                                               \  // 默认情况
        AT_ERROR(#NAME, " not implemented for '", the_type.toString(), "'"); \  // 抛出未实现错误
    }                                                                        \
  }()

// 分派浮点类型和半精度浮点(Half)的宏
#define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)                 \
  [&] {                                                                      \
    const at::Type& the_type = TYPE;                                         \
    switch (the_type.scalarType()) {                                         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)      \  // 增加半精度浮点支持
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", the_type.toString(), "'"); \
    }                                                                        \
  }()

// 分派浮点和复数类型的宏
#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...)              \
  [&] {                                                                      \
    const at::Type& the_type = TYPE;                                         \
    switch (the_type.scalarType()) {                                         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)  \  // 双精度复数
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)    \  // 单精度复数
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexHalf, std::complex<at::Half>, __VA_ARGS__)  \  // 半精度复数
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", the_type.toString(), "'"); \
    }                                                                        \
  }()

// 分派整数类型的宏
#define AT_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...)                          \
  [&] {                                                                      \
    const at::Type& the_type = TYPE;                                         \
    switch (the_type.scalarType()) {                                         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)       \  // 无符号8位整数
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)        \  // 有符号8位整数
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)        \  // 32位整数
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)       \  // 64位整数
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)      \  // 16位整数
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", the_type.toString(), "'"); \
    }                                                                        \
  }()

// 分派所有基本类型的宏（不包括半精度和复数）
#define AT_DISPATCH_ALL_TYPES(TYPE, NAME, ...)                               \
  [&] {                                                                      \
    const at::Type& the_type = TYPE;                                         \
    switch (the_type.scalarType()) {                                         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)      \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", the_type.toString(), "'"); \
    }                                                                        \
  }()

// 分派所有基本类型和半精度浮点的宏
#define AT_DISPATCH_ALL_TYPES_AND_HALF(TYPE, NAME, ...)                      \
  [&] {                                                                      \
    const at::Type& the_type = TYPE;                                         \
    switch (the_type.scalarType()) {                                         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)      \  // 半精度浮点
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", the_type.toString(), "'"); \
    }                                                                        \
  }()

// 分派复数类型的宏
#define AT_DISPATCH_COMPLEX_TYPES(TYPE, NAME, ...)                           \
  [&] {                                                                      \
    const at::Type& the_type = TYPE;                                         \
    switch (the_type.scalarType()) {                                         \
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)    \  // 单精度复数
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)  \  // 双精度复数
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", the_type.toString(), "'"); \
    }                                                                        \
  }()

// 分派所有基本类型和复数类型的宏
#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, ...)                   \
  [&] {                                                                      \
    const at::Type& the_type = TYPE;                                         \
    switch (the_type.scalarType()) {                                         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)  \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", the_type.toString(), "'"); \
    }                                                                        \
  }()

// 分派所有类型（基本类型、半精度浮点和复数）的宏
#define AT_DISPATCH_ALL_TYPES_AND_HALF_AND_COMPLEX(TYPE, NAME, ...)          \
  [&] {                                                                      \
    const at::Type& the_type = TYPE;                                         \
    switch (the_type.scalarType()) {                                         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)      \  // 半精度浮点
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)    \  // 单精度复数
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)  \  // 双精度复数
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", the_type.toString(), "'"); \
    }                                                                        \
  }()