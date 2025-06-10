#include <ATen/cuda/detail/CUDAHooks.h>

#include <ATen/CUDAGenerator.h>
#include <ATen/Context.h>
#include <ATen/RegisterCUDA.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/cuda/CuFFTPlanCache.h>
#include <c10/util/Exception.h>

#include "THC/THC.h"  // Torch CUDA库头文件
#include <THC/THCGeneral.hpp>

#if AT_CUDNN_ENABLED()
#include "ATen/cudnn/cudnn-wrapper.h"  // cuDNN包装器
#endif

#include <cuda.h>  // CUDA主头文件

#include <cstddef>
#include <functional>
#include <memory>

namespace at {
namespace cuda {
namespace detail {

// 初始化CUDA状态(THCState)
// 注意: 删除器是动态的，因为需要它存在于单独的编译单元中
std::unique_ptr<THCState, void (*)(THCState*)> CUDAHooks::initCUDA() const {
  THCState* thc_state = THCState_alloc();  // 分配THC状态

  THCudaInit(thc_state);  // 初始化CUDA状态
  return std::unique_ptr<THCState, void (*)(THCState*)>(
      thc_state, [](THCState* p) {
        if (p)
          THCState_free(p);  // 自定义删除器释放THC状态
      });
}

// 初始化CUDA随机数生成器
std::unique_ptr<Generator> CUDAHooks::initCUDAGenerator(Context* context) const {
  return std::unique_ptr<Generator>(new CUDAGenerator(context));
}

// 检查系统是否支持CUDA
bool CUDAHooks::hasCUDA() const {
  int count;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err == cudaErrorInsufficientDriver) {  // 驱动不足
    return false;
  }
  return true;
}

// 检查是否编译了MAGMA支持
bool CUDAHooks::hasMAGMA() const {
#ifdef USE_MAGMA
  return true;
#else
  return false;
#endif
}

// 检查是否支持cuDNN
bool CUDAHooks::hasCuDNN() const {
  return AT_CUDNN_ENABLED();
}

// 获取当前设备索引
int64_t CUDAHooks::current_device() const {
  int device;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) {
    return device;
  }
  return -1;  // 获取失败返回-1
}

// 获取固定内存分配器
Allocator* CUDAHooks::getPinnedMemoryAllocator() const {
  return at::cuda::getPinnedMemoryAllocator();
}

// 注册CUDA类型
void CUDAHooks::registerCUDATypes(Context* context) const {
  register_cuda_types(context);
}

// 检查是否编译了cuDNN支持
bool CUDAHooks::compiledWithCuDNN() const {
  return AT_CUDNN_ENABLED();
}

// 检查是否编译了MIOpen支持(ROCm)
bool CUDAHooks::compiledWithMIOpen() const {
  return AT_ROCM_ENABLED();
}

// 检查是否支持带空洞卷积(dilated convolution)
bool CUDAHooks::supportsDilatedConvolutionWithCuDNN() const {
#if AT_CUDNN_ENABLED()
  cudaDeviceProp* prop =
      THCState_getCurrentDeviceProperties(globalContext().getTHCState());
  // cuDNN 6.0+ 且计算能力>=5.0，或 cuDNN 6.1+ 支持空洞卷积
  return (
      (CUDNN_VERSION >= (6021)) ||
      (CUDNN_VERSION >= (6000) && prop->major >= 5));
#else
  return false;
#endif
}

// 获取cuDNN版本
long CUDAHooks::versionCuDNN() const {
#if AT_CUDNN_ENABLED()
  return CUDNN_VERSION;
#else
  AT_ERROR("Cannot query CuDNN version if ATen_cuda is not built with CuDNN");
#endif
}

// 获取cuDNN BatchNorm最小epsilon值
double CUDAHooks::batchnormMinEpsilonCuDNN() const {
#if AT_CUDNN_ENABLED()
  return CUDNN_BN_MIN_EPSILON;
#else
  AT_ERROR(
      "Cannot query CUDNN_BN_MIN_EPSILON if ATen_cuda is not built with CuDNN");
#endif
}

// 获取cuFFT计划缓存最大大小
int64_t CUDAHooks::cuFFTGetPlanCacheMaxSize() const {
#ifndef __HIP_PLATFORM_HCC__
  return at::native::detail::cufft_get_plan_cache_max_size_impl();
#else
  AT_ERROR("cuFFT with HIP is not supported");  // HIP不支持cuFFT
#endif
}

// 设置cuFFT计划缓存最大大小
void CUDAHooks::cuFFTSetPlanCacheMaxSize(int64_t max_size) const {
#ifndef __HIP_PLATFORM_HCC__
  at::native::detail::cufft_set_plan_cache_max_size_impl(max_size);
#else
  AT_ERROR("cuFFT with HIP is not supported");
#endif
}

// 获取cuFFT计划缓存当前大小
int64_t CUDAHooks::cuFFTGetPlanCacheSize() const {
#ifndef __HIP_PLATFORM_HCC__
  return at::native::detail::cufft_get_plan_cache_size_impl();
#else
  AT_ERROR("cuFFT with HIP is not supported");
#endif
}

// 清空cuFFT计划缓存
void CUDAHooks::cuFFTClearPlanCache() const {
#ifndef __HIP_PLATFORM_HCC__
  at::native::detail::cufft_clear_plan_cache_impl();
#else
  AT_ERROR("cuFFT with HIP is not supported");
#endif
}

// 获取GPU数量
int CUDAHooks::getNumGPUs() const {
  int count;
  auto err = cudaGetDeviceCount(&count);
  if (err == cudaErrorNoDevice) {  // 没有设备
    return 0;
  } else if (err != cudaSuccess) {  // 其他错误
    AT_ERROR(
        "CUDA error (", static_cast<int>(err), "): ", cudaGetErrorString(err));
  }
  return count;
}

// 注册CUDA钩子
using at::CUDAHooksRegistry;
using at::RegistererCUDAHooksRegistry;

REGISTER_CUDA_HOOKS(CUDAHooks);

} // namespace detail
} // namespace cuda
} // namespace at