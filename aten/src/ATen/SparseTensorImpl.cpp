#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/core/LegacyTypeDispatch.h>

namespace at {

namespace {
  // 将稀疏张量类型ID转换为设备类型
  DeviceType sparseTensorIdToDeviceType(TensorTypeId type_id) {
    if (type_id == SparseCPUTensorId()) {
      return kCPU;  // CPU稀疏张量
    } else if (type_id == SparseCUDATensorId()) {
      return kCUDA;  // CUDA稀疏张量
    } else {
      AT_ERROR("无法使用非稀疏张量类型ID构造稀疏张量", type_id);
    }
  }
}

/*
 * 稀疏张量实现类(SparseTensorImpl)说明：
 *
 * 空稠密张量默认是大小为[0]的1维张量
 * (注意，它不是0维张量，因为0维张量是标量且有一个元素)
 *
 * 因此，空稀疏张量也应该是大小为[0]的1维张量。
 * 我们有 dim == sparse_dim + dense_dim；对于稀疏张量，
 * 我们设空稀疏张量的 sparse_dim == 1 和 dense_dim == 0。
 * (这里有一定自由度，但由于是稀疏维度，要求 sparse_dim > 0 是合理的)
 *
 * 这意味着我们为这种空张量分配一个[1,0]大小的索引张量和一个[0]大小的值张量
 */
SparseTensorImpl::SparseTensorImpl(at::TensorTypeId type_id, const caffe2::TypeMeta& data_type)
    : TensorImpl(type_id, data_type, nullptr, false)  // 调用基类构造函数
    , size_{0}  // 大小为0
    , sparse_dim_(1)  // 稀疏维度为1
    , dense_dim_(0)  // 稠密维度为0
    // 初始化索引张量：大小为[1,0]的Long类型张量
    , indices_(at::empty({1, 0}, at::initialTensorOptions()
                  .device(sparseTensorIdToDeviceType(type_id))
                  .dtype(ScalarType::Long)))
    // 初始化值张量：大小为[0]的指定数据类型张量
    , values_(at::empty({0}, at::initialTensorOptions()
                  .device(sparseTensorIdToDeviceType(type_id))
                  .dtype(data_type))) {}

// 返回张量大小
IntList SparseTensorImpl::sizes() const {
  return size_;
}

// 稀疏张量不支持步长，抛出错误
IntList SparseTensorImpl::strides() const {
  AT_ERROR("稀疏张量没有步长(strides)");
}

// 稀疏张量不支持连续性判断，抛出错误
bool SparseTensorImpl::is_contiguous() const {
  AT_ERROR("稀疏张量不支持is_contiguous");
}

// 获取指定维度的大小
int64_t SparseTensorImpl::size(int64_t d) const {
  d = at::maybe_wrap_dim(d, dim(), false);  // 处理负维度
  return size_[d];
}

// 稀疏张量不支持步长，抛出错误
int64_t SparseTensorImpl::stride(int64_t d) const {
  AT_ERROR("稀疏张量没有步长(strides)");
}

// 稀疏张量不支持维度重设，抛出错误
void SparseTensorImpl::resize_dim(int64_t ndim) {
  AT_ERROR("稀疏张量不支持resize_dim");
}

// 稀疏张量不支持大小设置，抛出错误
void SparseTensorImpl::set_size(int64_t dim, int64_t new_size) {
  AT_ERROR("稀疏张量不支持set_size");
}

// 稀疏张量不支持步长设置，抛出错误
void SparseTensorImpl::set_stride(int64_t dim, int64_t new_stride) {
  AT_ERROR("稀疏张量不支持set_stride");
}

// 稀疏张量不支持存储偏移设置，抛出错误
void SparseTensorImpl::set_storage_offset(int64_t storage_offset) {
  AT_ERROR("稀疏张量不支持set_storage_offset");
}

// 返回总维度数(稀疏维度+稠密维度)
int64_t SparseTensorImpl::dim() const {
  return sparse_dim_ + dense_dim_;
}

// 检查是否可以转换为0维张量
TensorImpl* SparseTensorImpl::maybe_zero_dim(bool condition_when_zero_dim) {
  AT_CHECK(condition_when_zero_dim == (dim() == 0),
           "尝试在SparseTensorImpl上执行maybe_zero_dim到", condition_when_zero_dim,
           "但SparseTensor的dim()是", dim(), "稀疏张量不支持",
           "通过maybe_zero_dim改变维度");
  return this;
}

// 稀疏张量没有存储概念，抛出错误
const Storage& SparseTensorImpl::storage() const {
  AT_ERROR("稀疏张量没有存储(storage)");
}

// 稀疏张量没有存储偏移，抛出错误
int64_t SparseTensorImpl::storage_offset() const {
  AT_ERROR("稀疏张量没有存储偏移");
}

// 不安全地设置索引和值张量
void SparseTensorImpl::set_indices_and_values_unsafe(const Tensor& indices, const Tensor& values) {
  AT_ASSERT(!indices.is_variable() && !values.is_variable());  // 必须是普通张量

  // 检查索引和值张量的类型和布局
  AT_CHECK(!indices.is_sparse(), "期望索引是稠密张量，但获取的索引布局为", indices.layout());
  AT_CHECK(!values.is_sparse(), "期望值是稠密张量，但获取的值布局为", values.layout());

  // 检查值张量类型是否匹配
  AT_CHECK(values.type().toSparse() == legacyTensorType(*this), "值类型必须匹配稀疏张量类型");
  // 检查索引张量类型
  AT_CHECK(indices.type().scalarType() == kLong, "索引必须是int64张量");
  // 检查后端是否匹配
  AT_CHECK(indices.type().backend() == values.type().backend(), 
           "索引的后端(", indices.type().backend(), ")必须匹配值的后端(", values.type().backend(), ")");
  // 检查设备是否匹配
  AT_CHECK(!indices.is_cuda() || indices.get_device() == values.get_device(), 
           "索引的设备(", indices.get_device(), ")必须匹配值的设备(", values.get_device(), ")");

  // 检查维度
  AT_CHECK(indices.dim() == 2, "索引必须是 sparse_dim x nnz，但获取的是: ", indices.sizes());
  AT_CHECK(indices.size(1) == values.size(0), 
           "索引和值必须有相同的nnz，但从索引获取的nnz: ", indices.size(1), 
           ", 从值获取的nnz: ", values.size(0));
  AT_CHECK(indices.size(0) == sparse_dim_, 
           "索引的第一维度不正确，期望 ", sparse_dim_, ", 获取 ", indices.size(0));
  AT_CHECK(values.dim() == dense_dim_ + 1, 
           "值的维度数不正确，期望 ", dense_dim_ + 1, ", 获取 ", values.dim());

  // 检查值张量大小
  auto dense_size_original = sizes().slice(sparse_dim_);
  std::vector<int64_t> expected_values_size_vec = {values.size(0)};
  expected_values_size_vec.insert(expected_values_size_vec.end(), dense_size_original.begin(), dense_size_original.end());
  IntList expected_values_size(expected_values_size_vec);
  auto new_values_size = values.sizes();
  AT_CHECK(
    std::equal(expected_values_size.begin(), expected_values_size.end(), new_values_size.begin()),
    "值的大小不正确，期望 ", expected_values_size, ", 获取 ", new_values_size
  );

  // 设置新的索引和值
  indices_ = indices;
  values_ = values;

  // 重置合并标志
  coalesced_ = false;
}

} // namespace at