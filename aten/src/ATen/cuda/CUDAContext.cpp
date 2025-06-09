#include "ATen/cuda/CUDAContext.h"  // ATen CUDA上下文头文件
#include "THC/THCGeneral.hpp"       // THC通用功能头文件

namespace at { namespace cuda {  // ATen库的CUDA命名空间

/* 设备信息相关函数 */

// 获取当前设备的warp大小（CUDA架构基本执行单元）
int warp_size() {
  return getCurrentDeviceProperties()->warpSize;  // 典型值：32（NVIDIA GPU）
}

// 获取当前CUDA设备的属性结构体
cudaDeviceProp* getCurrentDeviceProperties() {
  // 通过全局THC状态获取当前设备属性
  return THCState_getCurrentDeviceProperties(at::globalContext().getTHCState());
}

// 获取指定设备的属性结构体
// 参数：device - 设备ID
cudaDeviceProp* getDeviceProperties(int64_t device) {
  return THCState_getDeviceProperties(
    at::globalContext().getTHCState(), 
    (int)device  // 转换为int类型
  );
}

// 获取CUDA设备内存分配器
Allocator* getCUDADeviceAllocator() {
  return at::globalContext().getTHCState()->cudaDeviceAllocator;
}

/* CUDA库句柄管理 */

// 获取当前CUDA稀疏矩阵计算句柄(cusparse)
cusparseHandle_t getCurrentCUDASparseHandle() {
  return THCState_getCurrentSparseHandle(at::globalContext().getTHCState());
}

// 获取当前CUDA基础线性代数子程序句柄(cublas) 
cublasHandle_t getCurrentCUDABlasHandle() {
  return THCState_getCurrentBlasHandle(at::globalContext().getTHCState());
}

} // namespace cuda
} // namespace at