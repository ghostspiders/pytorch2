#pragma once  // 防止头文件重复包含

// CUDA相关头文件
#include <cuda.h>          // CUDA驱动API
#include <cuda_runtime.h>  // CUDA运行时API
#include <cuda_fp16.h>     // CUDA半精度浮点支持

// 动态链接库导出/导入宏定义
#ifdef _WIN32  // Windows平台处理
# if defined(ATen_cuda_EXPORTS) || defined(caffe2_gpu_EXPORTS) || defined(CAFFE2_CUDA_BUILD_MAIN_LIB)
#  define AT_CUDA_API __declspec(dllexport)  // 编译为DLL时导出符号
# else
#  define AT_CUDA_API __declspec(dllimport)  // 使用DLL时导入符号
# endif
#elif defined(__GNUC__)  // GCC/Clang平台处理
# if defined(ATen_cuda_EXPORTS) || defined(caffe2_gpu_EXPORTS)
#  define AT_CUDA_API __attribute__((__visibility__("default")))  // 设置符号可见性
# else
#  define AT_CUDA_API  // 非导出情况保持默认
# endif
#else  // 其他平台
# define AT_CUDA_API  // 默认无修饰
#endif