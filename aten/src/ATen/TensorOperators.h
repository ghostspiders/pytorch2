#pragma once

#include <c10/core/Scalar.h>  // 标量类型支持
#include "ATen/Tensor.h"      // 张量基类
#include "ATen/Type.h"        // 类型系统支持

#include <string>
#include <stdexcept>

namespace at {

// 右值赋值运算符重载 (拷贝语义)
inline Tensor & Tensor::operator=(Tensor const & rhs) && {
  return copy_(rhs);  // 调用拷贝操作
}

// 右值赋值运算符重载 (移动语义)  
inline Tensor & Tensor::operator=(Tensor && rhs) && {
  return copy_(rhs);  // 调用拷贝操作
}

// 右值赋值运算符重载 (标量值)
inline Tensor & Tensor::operator=(Scalar v) && {
  return fill_(v);  // 调用填充操作
}

// 一元负号运算符重载
inline Tensor Tensor::operator-() const {
  return neg();  // 返回负值张量
}

// 复合赋值运算符重载 += (张量版本)
inline Tensor& Tensor::operator+=(const Tensor & other) {
  return add_(other);  // 原地加法
}

// 复合赋值运算符重载 += (标量版本)  
inline Tensor& Tensor::operator+=(Scalar other) {
  return add_(other);  // 原地加标量
}

// 复合赋值运算符重载 -= (张量版本)
inline Tensor& Tensor::operator-=(const Tensor & other) {
  return sub_(other);  // 原地减法
}

// 复合赋值运算符重载 -= (标量版本)
inline Tensor& Tensor::operator-=(Scalar other) {
  return sub_(other);  // 原地减标量
}

// 复合赋值运算符重载 *= (张量版本)
inline Tensor& Tensor::operator*=(const Tensor & other) {
  return mul_(other);  // 原地乘法
}

// 复合赋值运算符重载 *= (标量版本)
inline Tensor& Tensor::operator*=(Scalar other) {
  return mul_(other);  // 原地乘标量
}

// 复合赋值运算符重载 /= (张量版本)
inline Tensor& Tensor::operator/=(const Tensor & other) {
  return div_(other);  // 原地除法
}

// 复合赋值运算符重载 /= (标量版本)
inline Tensor& Tensor::operator/=(Scalar other) {
  return div_(other);  // 原地除标量
}

// 下标运算符重载 (标量索引)
inline Tensor Tensor::operator[](Scalar index) const {
  AT_CHECK(
      index.isIntegral(),  // 检查是否为整数类型
      "Can only index tensors with integral scalars");
  return select(0, index.toLong());  // 在第0维选择
}

// 下标运算符重载 (张量索引)
inline Tensor Tensor::operator[](Tensor index) const {
  // 检查索引张量是否有效
  AT_CHECK(index.defined(), "Can only index with tensors that are defined");
  AT_CHECK(
      index.dim() == 0,  // 检查是否为标量张量
      "Can only index with tensors that are scalars (zero-dim)");
  // 转换为标量后调用标量版本
  return this->operator[](index.item());
}

// 下标运算符重载 (整数索引)
inline Tensor Tensor::operator[](int64_t index) const {
  return select(0, index);  // 在第0维选择
}

// 定义所有支持的二元操作宏
#define AT_FORALL_BINARY_OPS(_) \
_(+,x.add(y), y.add(x)) \               // 加法
_(*,x.mul(y), y.mul(x)) \              // 乘法
_(-,x.sub(y), ::at::empty(y.sizes(), y.options()).fill_(x).sub_(y)) \  // 减法
_(/,x.div(y), ::at::empty(y.sizes(), y.options()).fill_(x).div_(y)) \  // 除法
_(%,x.remainder(y), ::at::empty(y.sizes(), y.options()).fill_(x).remainder_(y)) \  // 取模
_(<,x.lt(y), y.gt(x)) \                // 小于
_(<=,x.le(y), y.ge(x)) \               // 小于等于
_(>,x.gt(y),y.lt(x)) \                 // 大于
_(>=,x.ge(y), y.le(x)) \               // 大于等于
_(==,x.eq(y), y.eq(x)) \               // 等于
_(!=,x.ne(y), y.ne(x))                 // 不等于

// 定义运算符的宏
#define DEFINE_OPERATOR(op,body,reverse_scalar_body) \
static inline Tensor operator op(const Tensor & x, const Tensor & y) { \
  return body; \                      // 张量-张量操作
} \
static inline Tensor operator op(const Tensor & x, Scalar y) { \
  return body; \                      // 张量-标量操作
} \
static inline Tensor operator op(Scalar x, const Tensor & y) { \
  return reverse_scalar_body; \       // 标量-张量操作(需要反转操作数)
}

// 使用宏生成所有二元运算符重载
AT_FORALL_BINARY_OPS(DEFINE_OPERATOR)

// 清理宏定义
#undef DEFINE_OPERATOR
#undef AT_FORALL_BINARY_OPS

} // namespace at