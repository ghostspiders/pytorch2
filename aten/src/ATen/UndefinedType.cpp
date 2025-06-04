#include "ATen/UndefinedType.h"
#include "c10/util/Exception.h"

namespace at {

// UndefinedType构造函数：初始化一个未定义类型的Type对象
UndefinedType::UndefinedType()
    : TypeDefault(UndefinedTensorId(), /*is_variable=*/false, /*is_undefined=*/true) {}

// 获取未定义类型的标量类型（返回Undefined）
ScalarType UndefinedType::scalarType() const {
  return ScalarType::Undefined;
}

// 获取未定义类型的类型元信息
caffe2::TypeMeta UndefinedType::typeMeta() const {
  return scalarTypeToTypeMeta(scalarType());
}

// 获取未定义类型的后端类型（返回Undefined）
Backend UndefinedType::backend() const {
  return Backend::Undefined;
}

// 以下所有方法对于未定义类型都是无效操作，调用时会抛出错误

// 获取分配器 - 未实现
Allocator* UndefinedType::allocator() const {
  AT_ERROR("allocator not defined for UndefinedType");
}

// 从指针获取设备 - 未实现
Device UndefinedType::getDeviceFromPtr(void*) const {
  AT_ERROR("getDeviceFromPtr not defined for UndefinedType");
}

// 创建存储空间 - 未实现
Storage UndefinedType::storage(bool resizable) const {
  AT_ERROR("storage not defined for UndefinedType");
}
Storage UndefinedType::storage(size_t size, bool resizable) const {
  AT_ERROR("storage(size_t) not defined for UndefinedType");
}

// 从数据blob创建存储空间 - 未实现
Storage UndefinedType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
  AT_ERROR("storageFromBlob not defined for UndefinedType");
}

// 从TH指针不安全地创建存储空间 - 未实现
Storage UndefinedType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  AT_ERROR("unsafeStorageFromTH not defined for UndefinedType");
}

// 使用指定分配器创建存储空间 - 未实现
Storage UndefinedType::storageWithAllocator(int64_t size, Allocator* allocator) const {
  AT_ERROR("storageWithAllocator not defined for UndefinedType");
}

// 从TH指针不安全地创建张量 - 未实现
Tensor UndefinedType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  AT_ERROR("unsafeTensorFromTH not defined for UndefinedType");
}

// 创建随机数生成器 - 未实现
std::unique_ptr<Generator> UndefinedType::generator() const {
  AT_ERROR("generator not defined for UndefinedType");
}

// 返回类型名称字符串
const char * UndefinedType::toString() const {
  return "UndefinedType";
}

// 获取类型ID
TypeID UndefinedType::ID() const {
  return TypeID::Undefined;
}

// 获取元素字节大小 - 未实现
size_t UndefinedType::elementSizeInBytes() const {
  AT_ERROR("elementSizeInBytes not defined for UndefinedType");
}

// 转换为其他后端类型
Type & UndefinedType::toBackend(Backend b) const {
  if (b == Backend::Undefined) {
    return TypeDefault::toBackend(b);  // 如果是转换为Undefined，允许转换
  }
  AT_ERROR("toBackend not implemented for UndefinedType to non-UndefinedType");  // 其他转换不允许
}

// 转换为其他标量类型
Type & UndefinedType::toScalarType(ScalarType s) const {
  if (s == ScalarType::Undefined) {
    return TypeDefault::toScalarType(s);  // 如果是转换为Undefined，允许转换
  }
  AT_ERROR("toScalarType not implemented for UndefinedType to non-UndefinedType");  // 其他转换不允许
}

}  // namespace at