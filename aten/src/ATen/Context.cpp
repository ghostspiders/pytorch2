#include "ATen/Config.h"
#include "Context.h"
#include <ATen/core/TensorOptions.h>
#include <thread>
#include <mutex>
#include <sstream>
#include <string>
#include <stdexcept>

#include "ATen/CPUGenerator.h"  // CPU随机数生成器
#include "ATen/RegisterCPU.h"   // 注册CPU相关类型
#include "ATen/Tensor.h"       // Tensor核心类
#include <ATen/cpu/FlushDenormal.h>  // 处理非正规浮点数

#include "TH/TH.h"  // 引入TH库，用于USE_LAPACK定义

namespace at {

// 默认错误处理函数，抛出运行时错误
static inline void errorHandler(const char * msg, void * data) {
  throw std::runtime_error(msg);
}

// 参数错误处理函数，抛出包含参数信息的运行时错误
static inline void argErrorHandler(int arg, const char * msg, void * data) {
  std::stringstream new_error;
  new_error << "invalid argument " << arg << ": " << msg;
  throw std::runtime_error(new_error.str());
}

// Context构造函数
Context::Context()
: next_id(static_cast<size_t>(TypeID::NumOptions))  // 初始化下一个类型ID
, thc_state(nullptr, [](THCState* p){ /* no-op */ } )  // 初始化CUDA状态(空操作)
, thh_state(nullptr, [](THHState* p){ /* no-op */ } )   // 初始化HIP状态(空操作)
{
  // 设置默认错误处理器
  THSetDefaultErrorHandler(errorHandler,nullptr);
  THSetDefaultArgErrorHandler(argErrorHandler,nullptr);

  // 注册CPU随机数生成器
  generator_registry[static_cast<int>(DeviceType::CPU)]
    .reset(new CPUGenerator(this));
    
  // 注册CPU相关类型
  register_cpu_types(this);
}

// 获取全局Context实例(单例模式)
// 注意：如果在静态生命周期对象的析构函数中调用可能会有问题
Context & globalContext() {
  static Context globalContext_;  // 静态局部变量，保证只初始化一次
  return globalContext_;
}

// 检查用户是否启用了CuDNN
// 注意：这只是用户设置，不代表CuDNN实际可用
bool Context::userEnabledCuDNN() const {
  return enabled_cudnn;
}

// 设置用户是否启用CuDNN
void Context::setUserEnabledCuDNN(bool e) {
  enabled_cudnn = e;
}

// 检查是否使用确定性CuDNN算法
bool Context::deterministicCuDNN() const {
  return deterministic_cudnn;
}

// 设置是否使用确定性CuDNN算法
void Context::setDeterministicCuDNN(bool b) {
  deterministic_cudnn = b;
}

// 检查是否启用CuDNN基准测试模式
bool Context::benchmarkCuDNN() const {
  return benchmark_cudnn;
}

// 设置是否启用CuDNN基准测试模式
void Context::setBenchmarkCuDNN(bool b) {
  benchmark_cudnn = b;
}

// 检查是否支持MKL
bool Context::hasMKL() const {
#if AT_MKL_ENABLED()  // 根据编译配置决定
  return true;
#else
  return false;
#endif
}

// 检查是否支持LAPACK
bool Context::hasLAPACK() const {
#ifdef USE_LAPACK  // 根据TH头文件中的定义决定
  return true;
#else
  return false;
#endif
}

// 设置是否刷新非正规浮点数(denormal)到零
bool Context::setFlushDenormal(bool on) {
  return at::cpu::set_flush_denormal(on);
}

// 根据TensorOptions获取对应的类型接口
TypeExtendedInterface& getType(TensorOptions options) {
  return globalContext().getType(
            options.backend(),          // 后端类型(CPU/CUDA等)
            typeMetaToScalarType(options.dtype()),  // 数据类型转换
            options.is_variable());    // 是否是变量
}

// 根据Tensor实现获取对应的类型接口
TypeExtendedInterface& getType(const TensorImpl* impl) {
  Backend backend = tensorTypeIdToBackend(impl->type_id());
  return globalContext().getType(
            backend, 
            typeMetaToScalarType(impl->dtype()), 
            impl->is_variable());
}

// 根据Tensor对象获取对应的类型接口
TypeExtendedInterface& getType(const Tensor& t) {
  return getType(t.unsafeGetTensorImpl());
}

// 根据TensorOptions获取传统的TH分发器
LegacyTHDispatcher& getLegacyTHDispatcher(TensorOptions options) {
  return globalContext().getLegacyTHDispatcher(
            options.backend(), 
            typeMetaToScalarType(options.dtype()));
}

// 根据Tensor实现获取传统的TH分发器
LegacyTHDispatcher& getLegacyTHDispatcher(const TensorImpl* impl) {
  Backend backend = tensorTypeIdToBackend(impl->type_id());
  return globalContext().getLegacyTHDispatcher(
            backend, 
            typeMetaToScalarType(impl->dtype()));
}

// 获取CPU分配器
Allocator* getCPUAllocator() {
  return getTHDefaultAllocator();  // 返回TH默认分配器
}

// 传统设备类型初始化结构体
struct LegacyDeviceTypeInit : public LegacyDeviceTypeInitInterface {
  LegacyDeviceTypeInit(LegacyDeviceTypeInitArgs) {}
  
  // 初始化CPU
  void initCPU() const override {
    globalContext();  // 确保全局Context已初始化
  }
  
  // 延迟初始化CUDA
  void initCUDA() const override {
    globalContext().lazyInitCUDA();
  }
  
  // 延迟初始化HIP
  void initHIP() const override {
    globalContext().lazyInitHIP();
  }
  
  // 延迟初始化复数支持
  void initComplex() const override {
    globalContext().lazyInitComplex();
  }
};

// 注册传统类型初始化器
REGISTER_LEGACY_TYPE_INIT(LegacyDeviceTypeInit);

} // namespace at