#include <ATen/core/context_base.h>  // ATen核心上下文基类
#include <c10/util/Logging.h>        // 日志工具

namespace at {  // ATen命名空间

// 定义设备类型上下文注册表
// 模板参数说明：
// - ContextRegistry: 注册表名称
// - at::DeviceType: 键类型（设备类型）
// - at::BaseContext: 注册对象类型（基础上下文类）
// - std::unique_ptr: 包装器类型（所有权管理）
// - at::Device: 构造参数类型
C10_DEFINE_TYPED_REGISTRY(
    ContextRegistry,
    at::DeviceType,
    at::BaseContext,
    std::unique_ptr,
    at::Device);

// 字节拷贝函数矩阵（三维数组）：
// 第一维度[2]: 同步/异步模式（0=同步，1=异步）
// 第二维度[COMPILE_TIME_MAX_DEVICE_TYPES]: 源设备类型
// 第三维度[COMPILE_TIME_MAX_DEVICE_TYPES]: 目标设备类型
static CopyBytesFunction g_copy_bytes[2][COMPILE_TIME_MAX_DEVICE_TYPES]
                                     [COMPILE_TIME_MAX_DEVICE_TYPES];

// 字节拷贝函数注册器实现
_CopyBytesFunctionRegisterer::_CopyBytesFunctionRegisterer(
    DeviceType fromType,       // 源设备类型
    DeviceType toType,         // 目标设备类型
    CopyBytesFunction func_sync,   // 同步拷贝函数
    CopyBytesFunction func_async)  // 异步拷贝函数
{
  auto from = static_cast<int>(fromType);  // 设备类型转数组索引
  auto to = static_cast<int>(toType);
  
  // 如果未提供异步函数，默认使用同步函数
  if (!func_async) {
    func_async = func_sync;
  }

  // 检查是否重复注册
  CHECK(
      g_copy_bytes[0][from][to] == nullptr &&
      g_copy_bytes[1][from][to] == nullptr)
      << "Duplicate registration for device type pair "
      << c10::DeviceTypeName(fromType) << ", " << c10::DeviceTypeName(toType);

  // 注册函数指针
  g_copy_bytes[0][from][to] = func_sync;   // 同步槽位
  g_copy_bytes[1][from][to] = func_async;  // 异步槽位
}

// 实际字节拷贝函数
void CopyBytes(
    size_t nbytes,        // 要拷贝的字节数
    const void* src,      // 源数据指针
    Device src_device,    // 源设备
    void* dst,           // 目标指针
    Device dst_device,   // 目标设备
    bool async)         // 是否异步模式
{
  // 获取对应的函数指针
  auto ptr = g_copy_bytes[async ? 1 : 0]             // 选择同步/异步槽位
                    [static_cast<int>(src_device.type())]  // 源设备索引
                    [static_cast<int>(dst_device.type())]; // 目标设备索引

  // 检查函数是否已注册
  CAFFE_ENFORCE(
      ptr,
      "No function found for copying from ",
      c10::DeviceTypeName(src_device.type()),
      " to ",
      c10::DeviceTypeName(dst_device.type()));

  // 执行拷贝操作
  ptr(nbytes, src, src_device, dst, dst_device);
}

} // namespace at

namespace caffe2 {

// 待办：重命名上下文头文件
// 计划将 context.h -> context_cpu.h
// context_base.h -> context.h

} // namespace caffe2