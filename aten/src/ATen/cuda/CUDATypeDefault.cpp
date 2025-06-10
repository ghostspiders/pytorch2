#include <ATen/cuda/CUDATypeDefault.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADevice.h>
#include <ATen/CUDAGenerator.h>
namespace at {  // ATen库的核心命名空间

// 获取CUDA设备默认的内存分配器
Allocator* CUDATypeDefault::allocator() const {
  return cuda::getCUDADeviceAllocator();  // 返回CUDA设备内存分配器实例
}

// 根据指针数据获取对应的设备信息
Device CUDATypeDefault::getDeviceFromPtr(void * data) const {
  return cuda::getDeviceFromPtr(data);  // 查询指针所属的CUDA设备
}

// 创建CUDA随机数生成器
std::unique_ptr<Generator> CUDATypeDefault::generator() const {
  // 创建并返回一个CUDA专用的随机数生成器实例
  return std::unique_ptr<Generator>(new CUDAGenerator(&at::globalContext()));
}

} // namespace at
