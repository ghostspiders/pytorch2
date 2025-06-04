#pragma once

// 基础类型定义头文件
#include "ATen/TypeDefault.h"
// 生成器检查相关头文件
#include "ATen/CheckGenerator.h"

// Windows平台特殊处理：避免Type宏冲突
#ifdef _MSC_VER
#ifdef Type
#undef Type
#endif
#endif

namespace at {

// UndefinedType最终类：表示未定义张量类型，继承自TypeDefault
struct UndefinedType final : public TypeDefault {
  // 显式构造函数
  explicit UndefinedType();
  
  // 重写TypeDefault的虚函数
  
  // 获取标量类型（返回Undefined）
  virtual ScalarType scalarType() const override;
  // 获取类型元信息
  virtual caffe2::TypeMeta typeMeta() const override;
  // 获取后端类型（返回Undefined）
  virtual Backend backend() const override;
  
  // 内存分配相关接口（未实现）
  virtual Allocator* allocator() const override;
  // 从数据指针获取设备信息（未实现）
  virtual Device getDeviceFromPtr(void* data) const override;
  
  // 存储空间相关接口（未实现）
  virtual Storage storage(bool resizable = false) const override;
  virtual Storage storage(size_t size, bool resizable = false) const override;
  virtual Storage storageFromBlob(void * data, int64_t size, 
                               const std::function<void(void*)> & deleter) const override;
  virtual Storage storageWithAllocator(int64_t size, Allocator* allocator) const override;
  
  // 随机数生成器接口（未实现）
  virtual std::unique_ptr<Generator> generator() const override;
  
  // 类型描述信息
  virtual const char * toString() const override;
  // 获取元素字节大小（未实现）
  virtual size_t elementSizeInBytes() const override;
  
  // 类型转换接口
  virtual Type & toBackend(Backend b) const override;  // 转换为其他后端类型
  virtual Type & toScalarType(ScalarType s) const override;  // 转换为其他标量类型
  
  // 类型标识
  virtual TypeID ID() const override;
  
  // 底层TH指针转换接口（未实现）
  virtual Storage unsafeStorageFromTH(void * th_pointer, bool retain) const override;
  virtual Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const override;
};

} // namespace at