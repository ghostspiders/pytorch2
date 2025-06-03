#pragma once
#include <ATen/TypeDefault.h>

namespace at {  // ATen核心命名空间

// CPU设备默认类型实现类（继承自TypeDefault基类）
// CAFFE2_API宏确保跨DLL的符号可见性
struct CAFFE2_API CPUTypeDefault : public TypeDefault {
  // 构造函数初始化Tensor类型标识和状态标志
  // type_id: 张量类型标识符
  // is_variable: 是否为可求导变量
  // is_undefined: 是否为未定义类型
  CPUTypeDefault(TensorTypeId type_id, bool is_variable, bool is_undefined)
      : TypeDefault(type_id, is_variable, is_undefined) {}
  
  // 获取CPU内存分配器（纯虚函数实现）
  Allocator* allocator() const override;
  
  // 从数据指针获取设备类型（CPU实现直接返回CPU设备）
  Device getDeviceFromPtr(void * data) const override;
  
  // 创建CPU随机数生成器实例
  std::unique_ptr<Generator> generator() const override;
};

} // namespace at
