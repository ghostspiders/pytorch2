#pragma once  // 防止头文件重复包含

#include "ATen/Tensor.h"  // 包含Tensor基础类
#include "ATen/core/TensorImpl.h"  // 包含Tensor实现基类
#include "c10/util/Exception.h"  // 包含异常处理

namespace at {

// 稀疏张量实现类，继承自TensorImpl
struct CAFFE2_API SparseTensorImpl : public TensorImpl {
  // 采用COO(坐标格式)存储，包含indices和values两部分
  
  /*
   * 类成员变量说明：
   * 
   * 不变量(INVARIANTS):
   * - sparse_dim: 稀疏维度数，范围[0, len(shape)]，满足 sparse_dim + dense_dim = len(shape)
   * - dense_dim: 稠密维度数，范围[0, len(shape)]，满足 sparse_dim + dense_dim = len(shape)
   * - _indices.shape: 维度为2，形状为(sparse_dim, nnz)
   * - _values.shape: 维度为1 + dense_dim，形状为(nnz, shape[sparse_dim:])
   */
  
  std::vector<int64_t> size_;  // 稀疏张量的实际大小(稠密化后的尺寸)
  int64_t sparse_dim_ = 0;     // 稀疏维度数量
  int64_t dense_dim_ = 0;      // 稠密维度数量
  Tensor indices_;             // 索引张量(总是Long类型)
  Tensor values_;              // 值张量
  
  // 是否已合并(coalesced)标志位
  // 合并的稀疏张量意味着：每个索引最多出现一次，且索引是排序的
  // 这使得转换为CSR格式非常容易
  bool coalesced_ = false;

public:
  // 构造函数声明
  explicit SparseTensorImpl(at::TensorTypeId, const caffe2::TypeMeta&);

  // 基本属性访问方法
  int64_t nnz() const { return values_.size(0); }       // 非零元素数量
  int64_t sparse_dim() const { return sparse_dim_; }    // 获取稀疏维度数
  int64_t dense_dim() const { return dense_dim_; }      // 获取稠密维度数
  bool coalesced() const { return coalesced_; }         // 是否已合并
  Tensor indices() const { return indices_; }           // 获取索引张量
  Tensor values() const { return values_; }             // 获取值张量

  // 重写TensorImpl的虚函数
  IntList sizes() const override;      // 获取尺寸
  IntList strides() const override;    // 获取步长(稀疏张量不支持)
  bool is_contiguous() const override; // 是否连续(稀疏张量不支持)
  int64_t size(int64_t d) const override;  // 获取指定维度大小
  int64_t stride(int64_t d) const override; // 获取指定维度步长(不支持)
  void resize_dim(int64_t ndim) override;  // 调整维度数(不支持)
  void set_size(int64_t dim, int64_t new_size) override;  // 设置维度大小(不支持)
  void set_stride(int64_t dim, int64_t new_stride) override;  // 设置步长(不支持)
  void set_storage_offset(int64_t storage_offset) override;  // 设置存储偏移(不支持)

  int64_t dim() const override;  // 获取总维度数
  TensorImpl* maybe_zero_dim(bool condition_when_zero_dim) override;  // 可能转换为0维
  const Storage& storage() const override;  // 获取存储(不支持)
  int64_t storage_offset() const override;  // 获取存储偏移(不支持)

  // 原始调整大小方法(不维护稀疏维度/稠密维度与indices/values的关系)
  void raw_resize_(int64_t sparse_dim, int64_t dense_dim, IntList size) {
    size_ = size.vec();  // 设置新尺寸
    sparse_dim_ = sparse_dim;  // 设置稀疏维度数
    dense_dim_ = dense_dim;  // 设置稠密维度数
    refresh_numel();  // 刷新元素总数
  }

  /*
   * 调整稀疏张量大小的方法(resize_)
   * 
   * 支持以下情况：
   * 1. 保持稠密维度数不变，且不缩小任何稠密维度大小
   * 2. 保持稀疏维度数不变，且不缩小任何稀疏维度大小
   * 3. 当稀疏张量的nnz为0时，可以自由改变稀疏和稠密维度的形状
   * 
   * 不支持以下情况(会抛出错误)：
   * 1. 尝试在非空稀疏张量上改变稀疏维度数(会使得存储的索引无效)
   * 2. 尝试在非空稀疏张量上改变稠密维度数(行为与稠密张量不一致)
   * 3. 尝试在非空稀疏张量上缩小任何稠密维度大小(行为与稠密张量不一致)
   * 4. 尝试在非空稀疏张量上缩小任何稀疏维度大小(可能导致索引越界)
   */
  void resize_(int64_t sparse_dim, int64_t dense_dim, IntList size) {
    // 检查维度总数是否正确
    AT_CHECK(sparse_dim + dense_dim == size.size(), 
             "维度总数必须是稀疏维度(", sparse_dim, ") + 稠密维度(", dense_dim, "), 但获取 ", size.size());
    
    // 非空张量的额外检查
    if (nnz() > 0) {
      // 错误提示信息
      auto alt_options_msg = "可以尝试以下选项:\n\
1. 如果需要这个尺寸的空稀疏张量，调用`x = torch.sparse_coo_tensor(size)`\n\
2. 如果需要调整这个张量大小，有以下选项:\n\
    1. 对于稀疏和稠密维度，保持它们的数量不变且大小不缩小，然后重试\n\
    2. 或者，从这个稀疏张量创建具有正确索引和值的新稀疏张量";

      // 检查稀疏维度数是否改变
      AT_CHECK(sparse_dim == sparse_dim_,
        "在非空稀疏张量上改变稀疏维度数(从", sparse_dim_, "到", sparse_dim, ")不被支持\n", alt_options_msg);

      // 检查稠密维度数是否改变
      AT_CHECK(dense_dim == dense_dim_,
        "在非空稀疏张量上改变稠密维度数(从", dense_dim_, "到", dense_dim, ")不被支持\n", alt_options_msg);

      // 检查是否缩小了稀疏维度
      bool shrinking_sparse_dims = false;
      auto sparse_size_original = sizes().slice(0, sparse_dim);
      auto sparse_size_new = size.slice(0, sparse_dim);
      for (int i = 0; i < sparse_dim; i++) {
        if (sparse_size_new[i] < sparse_size_original[i]) {
          shrinking_sparse_dims = true;
          break;
        }
      }

      // 检查是否缩小了稠密维度
      bool shrinking_dense_dim = false;
      auto dense_size_original = sizes().slice(sparse_dim);
      auto dense_size_new = size.slice(sparse_dim);
      for (int i = 0; i < dense_dim; i++) {
        if (dense_size_new[i] < dense_size_original[i]) {
          shrinking_dense_dim = true;
          break;
        }
      }

      // 检查并抛出相应错误
      AT_CHECK(!shrinking_sparse_dims,
        "在非空稀疏张量上缩小稀疏维度大小(从", sparse_size_original, "到", sparse_size_new, ")不被支持\n", alt_options_msg);

      AT_CHECK(!shrinking_dense_dim,
        "在非空稀疏张量上缩小稠密维度大小(从", dense_size_original, "到", dense_size_new, ")不被支持\n", alt_options_msg);
    }

    // 如果尺寸有变化，调整indices和values的大小
    if ((!size.equals(size_)) || (sparse_dim != sparse_dim_) || (dense_dim != dense_dim_)) {
      auto nnz = values().size(0);
      std::vector<int64_t> values_size = {nnz};
      auto dense_size = size.slice(sparse_dim);
      values_size.insert(values_size.end(), dense_size.begin(), dense_size.end());
      values_.resize_(values_size);  // 调整values大小
      indices_.resize_({sparse_dim, nnz});  // 调整indices大小
    }

    // 更新成员变量
    size_ = size.vec();
    sparse_dim_ = sparse_dim;
    dense_dim_ = dense_dim;
    refresh_numel();  // 刷新元素总数
  }

  // 调整大小并清空张量的方法
  void resize_and_clear_(int64_t sparse_dim, int64_t dense_dim, IntList size) {
    AT_CHECK(sparse_dim + dense_dim == size.size(), 
             "维度总数必须是稀疏维度(", sparse_dim, ") + 稠密维度(", dense_dim, "), 但获取 ", size.size());

    size_ = size.vec();
    sparse_dim_ = sparse_dim;
    dense_dim_ = dense_dim;

    // 创建空的indices和values
    auto empty_indices = at::empty({sparse_dim, 0}, indices().options());
    std::vector<int64_t> values_size = {0};
    auto dense_size = sizes().slice(sparse_dim);
    values_size.insert(values_size.end(), dense_size.begin(), dense_size.end());
    auto empty_values = at::empty(values_size, values().options());
    
    // 设置空的indices和values
    set_indices_and_values_unsafe(empty_indices, empty_values);
    refresh_numel();  // 刷新元素总数
  }

  // 设置合并标志
  void set_coalesced(bool coalesced) { coalesced_ = coalesced; }

  // 内部使用的非零元素数调整方法(不暴露给Python前端)
  void set_nnz_and_narrow(int64_t new_nnz) {
    AT_ASSERT(new_nnz <= nnz());
    indices_ = indices_.narrow(1, 0, new_nnz);  // 缩小indices
    values_ = values_.narrow(0, 0, new_nnz);    // 缩小values
  }

  // 直接设置indices和values(不检查索引是否越界)
  // 注意：这个方法不安全，应该只在确定索引不越界的情况下使用
  void set_indices_and_values_unsafe(const Tensor& indices, const Tensor& values);

private:
  // 获取设备类型的慢速路径实现
  int64_t get_device_slow() const override {
    return values_.get_device();
  }
};

} // namespace at