#include <ATen/detail/CUDAHooksInterface.h>  // CUDA钩子接口头文件
#include <ATen/Generator.h>                  // 随机数生成器头文件

// TODO: 不需要保留整个头文件，可以全部移到cpp文件中

namespace at { namespace cuda { namespace detail {

// CUDAHooksInterface 的实际实现
struct CUDAHooks : public at::CUDAHooksInterface {
  CUDAHooks(at::CUDAHooksArgs) {}  // 构造函数
  
  // 初始化CUDA状态，返回THCState智能指针
  std::unique_ptr<THCState, void(*)(THCState*)> initCUDA() const override;
  
  // 初始化CUDA随机数生成器
  std::unique_ptr<Generator> initCUDAGenerator(Context*) const override;
  
  // 检查系统是否有CUDA支持
  bool hasCUDA() const override;
  
  // 检查是否安装了MAGMA库
  bool hasMAGMA() const override;
  
  // 检查是否安装了CuDNN库
  bool hasCuDNN() const override;
  
  // 获取当前CUDA设备索引
  int64_t current_device() const override;
  
  // 获取固定内存(pinned memory)分配器
  Allocator* getPinnedMemoryAllocator() const override;
  
  // 注册CUDA类型
  void registerCUDATypes(Context*) const override;
  
  // 检查是否编译时链接了CuDNN
  bool compiledWithCuDNN() const override;
  
  // 检查是否编译时链接了MIOpen
  bool compiledWithMIOpen() const override;
  
  // 检查当前CuDNN是否支持空洞卷积
  bool supportsDilatedConvolutionWithCuDNN() const override;
  
  // 获取CuDNN版本号
  long versionCuDNN() const override;
  
  // 获取CuDNN BatchNorm允许的最小epsilon值
  double batchnormMinEpsilonCuDNN() const override;
  
  // 获取cuFFT计划缓存的最大大小
  int64_t cuFFTGetPlanCacheMaxSize() const override;
  
  // 设置cuFFT计划缓存的最大大小
  void cuFFTSetPlanCacheMaxSize(int64_t max_size) const override;
  
  // 获取当前cuFFT计划缓存的大小
  int64_t cuFFTGetPlanCacheSize() const override;
  
  // 清空cuFFT计划缓存
  void cuFFTClearPlanCache() const override;
  
  // 获取系统中GPU的数量
  int getNumGPUs() const override;
};

}}} // 命名空间结束 at::cuda::detail