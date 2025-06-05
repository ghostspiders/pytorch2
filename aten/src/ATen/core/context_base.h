#pragma once  // 防止头文件重复包含

#include <array>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <unordered_map>

#include <ATen/core/ATenGeneral.h>  // ATen核心宏定义
#include <c10/core/Allocator.h>     // 分配器接口
#include <c10/util/typeid.h>        // 类型信息工具
#include <c10/util/Exception.h>     // 异常处理
#include <c10/util/Registry.h>      // 注册表模板

namespace caffe2 {
class Event;  // 前向声明事件类
} // namespace caffe2

namespace at {  // ATen命名空间

class BaseContext;  // 前向声明

/**
 * Caffe2中Context类的虚接口
 *
 * Context定义了在特定设备上运行算子所需的所有功能。
 * 具体Context类需要实现BaseContext中的所有纯虚函数。
 */
class CAFFE2_API BaseContext {
 public:
  virtual ~BaseContext() noexcept {}  // 虚析构函数

  // 核心接口 ====================================================
  virtual Device device() const = 0;  // 获取当前设备
  virtual DeviceType device_type() const = 0;  // 获取设备类型

  // 设备流控制
  virtual void SwitchToDevice(int /*stream_id*/) = 0;  // 切换到指定流
  inline void SwitchToDevice() { SwitchToDevice(0); }  // 默认流重载

  // 事件同步
  virtual void WaitEvent(const caffe2::Event& ev) = 0;  // 等待事件完成
  virtual void Record(caffe2::Event* ev, const char* err_msg = nullptr) const = 0;  // 记录事件

  // 计算同步
  virtual void FinishDeviceComputation() = 0;  // 完成设备计算

  // 内存拷贝接口 ================================================
  // 同设备拷贝
  virtual void CopyBytesSameDevice(size_t nbytes, const void* src, void* dst) = 0;
  // CPU到设备拷贝
  virtual void CopyBytesFromCPU(size_t nbytes, const void* src, void* dst) = 0;
  // 设备到CPU拷贝
  virtual void CopyBytesToCPU(size_t nbytes, const void* src, void* dst) = 0;

  // 模板化拷贝方法（仅支持基础类型）------------------------------
  template <typename T>
  inline void CopySameDevice(size_t n, const T* src, T* dst) {
    static_assert(std::is_fundamental<T>::value, "需要基础类型");
    CopyBytesSameDevice(n * sizeof(T), src, dst);
  }

  template <typename T>
  inline void CopyFromCPU(size_t n, const T* src, T* dst) {
    static_assert(std::is_fundamental<T>::value, "需要基础类型");
    CopyBytesFromCPU(n * sizeof(T), src, dst);
  }

  template <typename T>
  inline void CopyToCPU(size_t n, const T* src, T* dst) {
    static_assert(std::is_fundamental<T>::value, "需要基础类型");
    CopyBytesToCPU(n * sizeof(T), src, dst);
  }

  // 高级类型支持 ================================================
  virtual bool SupportsNonFundamentalTypes() const { return false; }  // 是否支持非基础类型

  inline void EnforceMetaCopyOK() {  // 检查类型支持
    AT_ASSERTM(SupportsNonFundamentalTypes(), "需要基础类型");
  }

  // 基于TypeMeta的拷贝方法 -------------------------------------
  void CopyItemsSameDevice(const caffe2::TypeMeta& meta, size_t n, const void* src, void* dst) {
    if (meta.copy()) {  // 优先使用类型特定的拷贝函数
      EnforceMetaCopyOK();
      meta.copy()(src, dst, n);
    } else {  // 回退到字节拷贝
      CopyBytesSameDevice(n * meta.itemsize(), src, dst);
    }
  }

  void CopyItemsFromCPU(const caffe2::TypeMeta& meta, size_t n, const void* src, void* dst) {
    if (meta.copy()) {
      EnforceMetaCopyOK();
      meta.copy()(src, dst, n);
    } else {
      CopyBytesFromCPU(n * meta.itemsize(), src, dst);
    }
  }

  void CopyItemsToCPU(const caffe2::TypeMeta& meta, size_t n, const void* src, void* dst) {
    if (meta.copy()) {
      EnforceMetaCopyOK();
      meta.copy()(src, dst, n);
    } else {
      CopyBytesToCPU(n * meta.itemsize(), src, dst);
    }
  }
};

// 上下文注册系统 ================================================
C10_DECLARE_TYPED_REGISTRY(  // 声明上下文注册表
    ContextRegistry,
    at::DeviceType,       // 键类型：设备类型
    at::BaseContext,     // 值类型：上下文基类
    std::unique_ptr,    // 包装类型：独占指针
    at::Device);       // 构造参数类型

// 上下文注册宏
#define REGISTER_CONTEXT(type, ...) \
  C10_REGISTER_TYPED_CLASS(ContextRegistry, type, __VA_ARGS__)

// 上下文工厂函数
inline std::unique_ptr<at::BaseContext> CreateContext(const at::Device& device) {
  return at::ContextRegistry()->Create(device.type(), device);
}

// 跨设备拷贝系统 ================================================
using CopyBytesFunction = void (*)(  // 拷贝函数指针类型
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device);

// 拷贝函数注册器
struct CAFFE2_API _CopyBytesFunctionRegisterer {
  _CopyBytesFunctionRegisterer(
      DeviceType from,            // 源设备类型
      DeviceType to,              // 目标设备类型
      CopyBytesFunction func_sync,   // 同步拷贝函数
      CopyBytesFunction func_async = nullptr);  // 异步拷贝函数（可选）
};

// 拷贝函数注册宏
#define REGISTER_COPY_BYTES_FUNCTION(from, to, ...)           \
  namespace {                                                 \
  static _CopyBytesFunctionRegisterer C10_ANONYMOUS_VARIABLE( \
      g_copy_function)(from, to, __VA_ARGS__);                \
  }

// 全局拷贝函数
CAFFE2_API void CopyBytes(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device,
    bool async);  // 是否异步执行

} // namespace at

// Caffe2兼容层 ================================================
namespace caffe2 {
using at::BaseContext;      // 复用ATen的上下文基类
using at::CreateContext;   // 复用上下文创建函数
} // namespace caffe2