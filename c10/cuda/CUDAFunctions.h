#pragma once

// 此头文件提供常用CUDA API函数的C++封装
// 使用C++封装的好处是可以在出错时抛出异常，
// 而不是显式传递错误码，这使得API更加自然易用
//
// 命名规范与torch.cuda保持一致

#include <cuda_runtime_api.h>  // CUDA运行时API头文件

#include <c10/macros/Macros.h>  // PyTorch宏定义
#include <c10/Device.h>        // 设备相关定义

namespace c10 {
namespace cuda {

// 获取当前系统中可用的CUDA设备数量
inline DeviceIndex device_count() {
  int count;  // 存储设备数量的临时变量
  // 调用CUDA API获取设备数量，并通过宏检查错误
  C10_CUDA_CHECK(cudaGetDeviceCount(&count));
  // 将结果转换为类型安全的DeviceIndex并返回
  return static_cast<DeviceIndex>(count);
}

// 获取当前活动的CUDA设备索引
inline DeviceIndex current_device() {
  int cur_device;  // 存储当前设备索引的临时变量
  // 调用CUDA API获取当前设备，并通过宏检查错误
  C10_CUDA_CHECK(cudaGetDevice(&cur_device));
  // 将结果转换为类型安全的DeviceIndex并返回
  return static_cast<DeviceIndex>(cur_device);
}

// 设置当前活动的CUDA设备
inline void set_device(DeviceIndex device) {
  // 调用CUDA API设置设备，将DeviceIndex转换为int，并通过宏检查错误
  C10_CUDA_CHECK(cudaSetDevice(static_cast<int>(device)));
}

}} // namespace c10::cuda