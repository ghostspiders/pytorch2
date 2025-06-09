#pragma once  // 防止头文件重复包含

// ATen核心头文件
#include "ATen/core/ATenGeneral.h"  // ATen通用定义
#include "ATen/Context.h"           // ATen上下文
#include "ATen/cuda/CUDAStream.h"   // CUDA流管理
#include "ATen/cuda/Exceptions.h"   // CUDA异常处理
#include "c10/cuda/CUDAFunctions.h" // CUDA基础功能

// 标准库
#include <cstdint>  // 标准整数类型

// CUDA库头文件
#include "cuda_runtime_api.h"  // CUDA运行时API
#include "cusparse.h"          // CUDA稀疏矩阵库
#include "cublas_v2.h"         // CUDA BLAS库(v2)

namespace at {
namespace cuda {

/*
 * ATen的统一CUDA接口
 *
 * 注意与CUDAHooks的区别：
 * - CUDAHooks: 用于CPU/CUDA混合编译的运行时分发接口
 * - CUDAContext: 专用于CUDA编译环境的统一功能接口
 *
 * 使用原则：
 * - 仅在CUDA编译环境中使用的文件应使用CUDAContext
 * - 需要兼容CPU/CUDA环境的文件使用CUDAHooks
 *
 * 本接口不关联具体类，各模块自行管理状态
 * CUDA上下文全局唯一
 */

/* 设备信息相关接口 */

// 获取系统中可用GPU数量
inline int64_t getNumGPUs() {
    return c10::cuda::device_count();  // 调用c10库的设备计数函数
}

/**
 * 检查CUDA是否可用
 * 可能编译时支持CUDA但运行时无可用设备
 * @return bool  true表示有可用CUDA设备
 */
inline bool is_available() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorInsufficientDriver) {  // 驱动不兼容
      return false;
    }
    return count > 0;  // 有至少一个可用设备
}

// 获取当前设备属性结构体（包含计算能力等参数）
CAFFE2_API cudaDeviceProp* getCurrentDeviceProperties();

// 获取当前设备的warp大小（典型值32）
CAFFE2_API int warp_size();

// 获取指定设备的属性结构体
CAFFE2_API cudaDeviceProp* getDeviceProperties(int64_t device);

// 获取CUDA设备内存分配器
CAFFE2_API Allocator* getCUDADeviceAllocator();

/* CUDA数学库句柄 */

// 获取当前线程的cuSPARSE稀疏矩阵计算句柄
CAFFE2_API cusparseHandle_t getCurrentCUDASparseHandle();

// 获取当前线程的cuBLAS线性代数计算句柄
CAFFE2_API cublasHandle_t getCurrentCUDABlasHandle();

} // namespace cuda
} // namespace at