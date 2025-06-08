#include <ATen/core/LegacyDeviceTypeInit.h>

namespace at {

// 定义LegacyDeviceTypeInit的注册表，用于管理不同设备类型的初始化接口
// 注册表参数说明：
// - LegacyDeviceTypeInitRegistry: 注册表名称
// - LegacyDeviceTypeInitInterface: 注册的接口类型 
// - LegacyDeviceTypeInitArgs: 构造接口时传入的参数类型
C10_DEFINE_REGISTRY(
    LegacyDeviceTypeInitRegistry,
    LegacyDeviceTypeInitInterface,
    LegacyDeviceTypeInitArgs)

// 获取Legacy设备类型初始化接口的单例对象
const LegacyDeviceTypeInitInterface& getLegacyDeviceTypeInit() {
  // 静态变量，保存单例对象
  static std::unique_ptr<LegacyDeviceTypeInitInterface> legacy_device_type_init;
  
  // 用于保证线程安全的once_flag
  static std::once_flag once;
  
  // 使用std::call_once保证线程安全的延迟初始化
  std::call_once(once, [] {
    // 尝试从注册表创建实例
    legacy_device_type_init = LegacyDeviceTypeInitRegistry()->Create(
        "LegacyDeviceTypeInit", 
        LegacyDeviceTypeInitArgs{});
    
    // 如果注册表中没有对应的实现，创建一个默认的空接口实例
    if (!legacy_device_type_init) {
      legacy_device_type_init =
          std::unique_ptr<LegacyDeviceTypeInitInterface>(
              new LegacyDeviceTypeInitInterface());
    }
  });
  
  // 返回单例引用
  return *legacy_device_type_init;
}

} // namespace at