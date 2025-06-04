#pragma once

#include <ATen/Type.h>
#include <ATen/WrapDimUtils.h>

namespace at {

// 张量几何结构，描述张量的形状、步长等基本信息
struct CAFFE2_API TensorGeometry {
  // 默认构造函数，初始化存储偏移为0
  TensorGeometry() : storage_offset_(0) {}

  // 通过大小列表构造张量几何
  // 计算连续内存布局的默认步长
  explicit TensorGeometry(IntList sizes)
    : sizes_(sizes.vec())         // 初始化大小向量
    , strides_(sizes.size())      // 初始化步长向量
    , storage_offset_(0) {        // 初始化存储偏移为0
      int64_t dim = sizes.size();
      int64_t expected_stride = 1;
      // 从最后一个维度开始计算步长
      for (int64_t i = dim - 1; i >= 0; i--) {
        strides_[i] = expected_stride;  // 设置当前维度步长
        expected_stride *= sizes_[i];   // 计算下一个维度的步长
      }
      numel_ = expected_stride;  // 总元素数量
  }

  // 通过现有张量构造张量几何
  explicit TensorGeometry(const Tensor& t)
    : sizes_(t.sizes().vec())       // 复制张量大小
    , strides_(t.strides().vec())   // 复制张量步长
    , storage_offset_(t.storage_offset())  // 复制存储偏移
    , numel_(t.numel()) {}         // 复制元素总数

  // 检查张量是否是连续内存布局
  bool is_contiguous() const;

  // 返回张量维度数
  int64_t dim() const { return sizes_.size(); }
  
  // 返回指定维度的大小
  int64_t size(int64_t dim) const {
    dim = maybe_wrap_dim(dim, this->dim());  // 处理负维度
    return sizes_.at(static_cast<size_t>(dim));
  }
  
  // 返回所有维度的大小
  IntList sizes() const { return IntList{ sizes_ }; }
  
  // 返回指定维度的步长
  int64_t stride(int64_t dim) const {
    dim = maybe_wrap_dim(dim, this->dim());  // 处理负维度
    return strides_.at(static_cast<size_t>(dim));
  }
  
  // 返回所有维度的步长
  IntList strides() const { return IntList{ strides_ }; }
  
  // 返回存储偏移量
  int64_t storage_offset() const { return storage_offset_; }
  
  // 返回元素总数
  int64_t numel() const { return numel_; }

  // 交换两个维度的位置（转置操作）
  TensorGeometry transpose(int64_t dim0, int64_t dim1) {
    TensorGeometry r = *this; // 创建副本
    // 检查维度是否有效
    AT_CHECK(dim0 < dim(), "transpose: dim0=", dim0, " out of range (dim=", dim(), ")")
    AT_CHECK(dim1 < dim(), "transpose: dim1=", dim1, " out of range (dim=", dim(), ")")
    // 交换大小和步长
    std::swap(r.sizes_[dim0], r.sizes_[dim1]);
    std::swap(r.strides_[dim0], r.strides_[dim1]);
    return r;
  }

  std::vector<int64_t> sizes_;      // 各维度大小
  std::vector<int64_t> strides_;    // 各维度步长
  int64_t storage_offset_;          // 存储偏移量
  int64_t numel_;                   // 元素总数
};

} // namespace at