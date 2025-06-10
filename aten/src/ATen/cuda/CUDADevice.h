// 防止头文件被重复包含
#pragma once

// 引入ATen CUDA异常处理头文件
#include "ATen/cuda/Exceptions.h"
// 引入CUDA运行时API头文件
#include "cuda.h"

// 定义ATen库的命名空间
namespace at {
namespace cuda {

// 内联函数：通过指针获取对应的CUDA设备
// 参数:
//   void* ptr - 指向CUDA内存的指针
// 返回值:
//   Device - 包含设备类型和设备编号的结构体
inline Device getDeviceFromPtr(void* ptr) {
  // 定义CUDA指针属性结构体
  struct cudaPointerAttributes attr;
  
  // 调用CUDA API获取指针属性，并检查错误
  // AT_CUDA_CHECK是ATen的CUDA错误检查宏
  AT_CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
  
  // 返回Device结构体，包含:
  //   - 设备类型设置为CUDA (DeviceType::CUDA)
  //   - 设备编号从attr.device转换而来(转为int16_t)
  return {DeviceType::CUDA, static_cast<int16_t>(attr.device)};
}

}} // 结束命名空间 at::cuda