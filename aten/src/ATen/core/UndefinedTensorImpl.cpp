#include "ATen/core/UndefinedTensorImpl.h"
#include "c10/util/Exception.h"

namespace at {

// 未定义张量的实现类
// 注意：这里是否应该使用全局上下文？能否通过某种方式传入上下文？
UndefinedTensorImpl::UndefinedTensorImpl()
    : TensorImpl(
          UndefinedTensorId(),       // 使用未定义张量ID
          caffe2::TypeMeta(),        // 空的类型元信息
          nullptr,                  // 无存储空间
          /* is variable */ false    // 不是变量
      ) {
}

// 获取张量维度大小（未实现，抛出错误）
IntList UndefinedTensorImpl::sizes() const {
  AT_ERROR("sizes() called on undefined Tensor");  // 不能在未定义张量上调用sizes()
}

// 获取特定维度的大小（未实现，抛出错误）
int64_t UndefinedTensorImpl::size(int64_t d) const {
  AT_ERROR("size(dim) called on an undefined Tensor");  // 不能在未定义张量上调用size(dim)
}

// 获取特定维度的步长（未实现，抛出错误）
int64_t UndefinedTensorImpl::stride(int64_t d) const {
  AT_ERROR("stride(dim) called on an undefined Tensor");  // 不能在未定义张量上调用stride(dim)
}

// 获取张量维度数（未实现，抛出错误）
int64_t UndefinedTensorImpl::dim() const {
  AT_ERROR("dim() called on undefined Tensor");  // 不能在未定义张量上调用dim()
}

// 获取存储对象（未实现，抛出错误）
const Storage& UndefinedTensorImpl::storage() const {
  AT_ERROR("storage() called on undefined Tensor");  // 不能在未定义张量上调用storage()
}

// 获取存储偏移量（未实现，抛出错误）
int64_t UndefinedTensorImpl::storage_offset() const {
  AT_ERROR("storage_offset() called on an undefined Tensor");  // 不能在未定义张量上调用storage_offset()
}

// 获取所有维度的步长（未实现，抛出错误）
IntList UndefinedTensorImpl::strides() const {
  AT_ERROR("strides() called on undefined Tensor");  // 不能在未定义张量上调用strides()
}

// 全局唯一的未定义张量实例（单例模式）
UndefinedTensorImpl UndefinedTensorImpl::_singleton;

}  // namespace at