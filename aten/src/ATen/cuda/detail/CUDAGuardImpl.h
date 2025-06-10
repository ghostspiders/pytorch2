#pragma once

#include <c10/impl/DeviceGuardImplInterface.h>  // 设备守卫实现接口
#include <c10/macros/Macros.h>                  // 宏定义

#include <ATen/cuda/Exceptions.h>  // CUDA异常处理
#include <ATen/cuda/CUDAStream.h>  // CUDA流定义

#include <cuda_runtime_api.h>      // CUDA运行时API

namespace at {
namespace cuda {
namespace impl {

// CUDA设备守卫实现类，继承自设备守卫接口
struct CUDAGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  // 静态常量，表示设备类型为CUDA
  static constexpr DeviceType static_type = DeviceType::CUDA;
  
  // 默认构造函数
  CUDAGuardImpl() {}
  
  // 带设备类型的构造函数，确保只能是CUDA类型
  CUDAGuardImpl(DeviceType t) {
    AT_ASSERT(t == DeviceType::CUDA);
  }
  
  // 获取设备类型
  DeviceType type() const override {
    return DeviceType::CUDA;
  }
  
  // 交换当前设备并返回旧设备
  Device exchangeDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::CUDA);  // 确保是CUDA设备
    Device old_device = getDevice();          // 获取当前设备
    if (old_device.index() != d.index()) {    // 如果设备不同
      AT_CUDA_CHECK(cudaSetDevice(d.index())); // 设置新设备并检查错误
    }
    return old_device;  // 返回旧设备
  }
  
  // 获取当前设备
  Device getDevice() const override {
    int device;
    AT_CUDA_CHECK(cudaGetDevice(&device));  // 获取当前设备索引并检查错误
    return Device(DeviceType::CUDA, device); // 返回设备对象
  }
  
  // 设置当前设备
  void setDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::CUDA);  // 确保是CUDA设备
    AT_CUDA_CHECK(cudaSetDevice(d.index()));  // 设置设备并检查错误
  }
  
  // 不检查错误的设备设置(noexcept)
  void uncheckedSetDevice(Device d) const noexcept override {
    cudaSetDevice(d.index());  // 直接设置设备，不检查错误
  }
  
  // 获取指定设备的当前流(noexcept)
  Stream getStream(Device d) const noexcept override {
    return at::cuda::getCurrentCUDAStream().unwrap();  // 获取当前流并解包
  }
  
  // 交换当前流并返回旧流(不改变当前设备)(noexcept)
  Stream exchangeStream(Stream s) const noexcept override {
    CUDAStream cs(s);  // 构造CUDA流对象
    // 获取指定设备的当前流
    auto old_stream = at::cuda::getCurrentCUDAStream(s.device().index());
    at::cuda::setCurrentCUDAStream(cs);  // 设置新流
    return old_stream.unwrap();          // 返回旧流
  }
};

}}} // namespace at::cuda::impl