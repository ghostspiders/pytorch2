#include <ATen/core/TensorImpl.h>

#include <c10/core/Backend.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/WrapDimMinimal.h>
#include "c10/util/Optional.h"

#include <ATen/core/VariableHooksInterface.h>

namespace at {

// 获取梯度张量(非常量版本) - 未实现抛出错误
Tensor& TensorImpl::grad() {
  AT_ERROR("grad is not implemented for Tensor");
}

// 获取梯度张量(常量版本) - 未实现抛出错误
const Tensor& TensorImpl::grad() const {
  AT_ERROR("grad is not implemented for Tensor");
}

// 构造函数1: 根据类型ID、数据类型、分配器和是否为变量创建TensorImpl
TensorImpl::TensorImpl(TensorTypeId type_id, const caffe2::TypeMeta& data_type, Allocator *allocator, bool is_variable)
    : TensorImpl({}, type_id, data_type, is_variable) {
  // 变量、未定义张量和稀疏张量没有存储
  if (!is_variable && type_id != UndefinedTensorId() && data_type.id() != caffe2::TypeIdentifier::uninitialized()
      && type_id != SparseCPUTensorId() && type_id != SparseCUDATensorId()) {
    storage_ = Storage(data_type, 0, allocator, true);  // 创建空存储
  }
}

// 构造函数2: 从现有存储创建TensorImpl
TensorImpl::TensorImpl(Storage&& storage, TensorTypeId type_id, bool is_variable)
    : TensorImpl(std::move(storage), type_id, storage.dtype(), is_variable) {}

// 构造函数3: 核心实现(从存储、类型ID、数据类型和是否为变量创建TensorImpl)
TensorImpl::TensorImpl(Storage&& storage, TensorTypeId type_id, const caffe2::TypeMeta& data_type, bool is_variable)
    : storage_(std::move(storage)),  // 移动存储
      sizes_{0},                     // 初始化大小为0
      storage_offset_(0),            // 存储偏移为0
      numel_(0),                     // 元素数量为0
      data_type_(data_type),         // 设置数据类型
      type_id_(type_id),             // 设置类型ID
      is_variable_(is_variable) {    // 设置是否为变量
  strides_.push_back(1);             // 初始化步长为1
}

// 获取张量各维度大小
IntList TensorImpl::sizes() const {
  return sizes_;
}

// 获取张量各维度步长
IntList TensorImpl::strides() const {
  return strides_;
}

// 计算张量是否是连续的
bool TensorImpl::compute_contiguous() const {
  bool is_contiguous = true;
  if (is_empty())  // 空张量视为连续
    return is_contiguous;
  
  int64_t z = 1;  // 用于计算期望步长
  // 从最后一个维度开始检查
  for (int64_t d = dim() - 1; d >= 0; d--) {
    if (size(d) != 1) {  // 忽略大小为1的维度
      if (stride(d) == z) {  // 步长符合连续布局
        z *= size(d);        // 更新期望步长
      } else {
        is_contiguous = false;  // 不连续
        break;
      }
    }
  }
  return is_contiguous;
}

// 释放张量资源(主要是存储)
void TensorImpl::release_resources() {
  if (storage_) {
    storage_ = {};  // 释放存储
  }
}

// 获取张量维度数
int64_t TensorImpl::dim() const {
  return sizes_.size();
}

// 获取指定维度的大小(带维度检查)
int64_t TensorImpl::size(int64_t d) const {
  d = at::maybe_wrap_dim(d, dim(), false);  // 处理负维度
  return sizes_[d];
}

// 获取指定维度的步长(带维度检查)
int64_t TensorImpl::stride(int64_t d) const {
  d = at::maybe_wrap_dim(d, dim(), false);  // 处理负维度
  return strides_[d];
}

// 可能将张量降为0维(如果满足条件)
TensorImpl* TensorImpl::maybe_zero_dim(bool condition_when_zero_dim) {
  bool set_zero_dim = condition_when_zero_dim && this->sizes().size() == 1 && this->size(0) == 1;
  if (set_zero_dim) {
    resize_dim(0);  // 调整为0维
  }
  return this;
}

// 获取存储(常量版本)
const Storage& TensorImpl::storage() const {
  return storage_;
}

// 删除PlacementDeleteContext的静态函数
static void deletePlacementDeleteContext(void* ptr) {
  delete static_cast<PlacementDeleteContext*>(ptr);
}

// 创建带有placement删除器的DataPtr
at::DataPtr PlacementDeleteContext::makeDataPtr(
    at::DataPtr&& data_ptr,
    PlacementDtor placement_dtor,
    size_t size,
    at::Device device) {
  auto* ptr = data_ptr.get();
  return {ptr,
          new PlacementDeleteContext(std::move(data_ptr), placement_dtor, size),
          &deletePlacementDeleteContext,
          device};
}

} // namespace at