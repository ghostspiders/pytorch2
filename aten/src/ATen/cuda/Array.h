#pragma once  // 防止头文件重复包含

// 定义可在CUDA kernel中使用的固定大小数组类型

#include <c10/macros/Macros.h>  // 包含C10的宏定义（如设备/主机函数修饰符）

namespace at { namespace cuda {  // ATen库的CUDA命名空间

/**
 * 固定大小的数组模板类，可在主机和设备代码中使用
 * @tparam T 数组元素类型
 * @tparam size 数组固定大小
 */
template <typename T, int size>
struct Array {
  T data[size];  // 内联存储数组数据

  // 常量下标访问运算符（主机/设备均可调用）
  C10_HOST_DEVICE T operator[](int i) const {
    return data[i];
  }

  // 非常量下标访问运算符（主机/设备均可调用）
  C10_HOST_DEVICE T& operator[](int i) {
    return data[i];
  }

  // 默认构造函数（支持HIP和CUDA的host-device调用）
  C10_HIP_HOST_DEVICE Array() = default;
  
  // 默认拷贝构造函数
  C10_HIP_HOST_DEVICE Array(const Array&) = default;
  
  // 默认赋值运算符
  C10_HIP_HOST_DEVICE Array& operator=(const Array&) = default;

  /**
   * 用指定值填充整个数组的构造函数
   * @param x 用于填充数组的值
   */
  C10_HOST_DEVICE Array(T x) {
    for (int i = 0; i < size; i++) {
      data[i] = x;  // 初始化每个元素
    }
  }
};

}} // namespace at::cuda