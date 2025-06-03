#include <ATen/CPUTypeDefault.h>

#include <ATen/Context.h>
#include <ATen/CPUGenerator.h>

namespace at {  // ATen核心命名空间

// 获取CPU默认内存分配器
Allocator* CPUTypeDefault::allocator() const {
  return getCPUAllocator();  // 返回全局CPU内存分配器单例
}

// 根据数据指针判断设备类型（始终返回CPU）
Device CPUTypeDefault::getDeviceFromPtr(void * data) const {
  return DeviceType::CPU;  // CPU设备特有实现，无需实际检查指针
}

// 创建CPU随机数生成器
std::unique_ptr<Generator> CPUTypeDefault::generator() const {
  // 创建新的CPU生成器实例，并传入全局上下文
  return std::unique_ptr<Generator>(new CPUGenerator(&at::globalContext()));
}

} // namespace at

