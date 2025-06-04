#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>

namespace at { namespace sparse {

// 以下类型定义主要用于文档说明目的
using SparseTensor = Tensor;  // 稀疏张量类型
using LongTensor = Tensor;    // 长整型张量类型  
using IntTensor = Tensor;     // 整型张量类型
using SparseType = Type;      // 稀疏类型

/* 
 * 获取稀疏张量实现对象的内部工具函数
 * 用于访问SparseTensorImpl的特殊字段
 * 注意：
 * - 仅用于实现稀疏张量的底层setter/getter
 * - 确保调用开销低(可能被频繁调用)
 */
inline SparseTensorImpl* get_sparse_impl(const SparseTensor& self) {
  AT_ASSERTM(!self.is_variable(), "_internal_get_SparseTensorImpl: 不能是变量");
  AT_ASSERTM(self.is_sparse(), "_internal_get_SparseTensorImpl: 不是稀疏张量");
  return static_cast<SparseTensorImpl*>(self.unsafeGetTensorImpl());
}

/*
 * 直接将indices和values放入稀疏张量(不拷贝数据)
 * 原名为THSTensor_(_move)
 */
inline void alias_into_sparse(const SparseTensor& self, const LongTensor& indices, const Tensor& values) {
  get_sparse_impl(self)->set_indices_and_values_unsafe(indices, values);
}

/*
 * 拷贝indices和values数据到稀疏张量
 * 原名为THSTensor_(_set)
 * 参数non_blocking表示是否使用非阻塞拷贝
 */
inline void copy_into_sparse(const SparseTensor& self, const LongTensor& indices, const Tensor& values, bool non_blocking) {
  alias_into_sparse(self, 
                   self._indices().type().copy(indices, non_blocking),
                   self._values().type().copy(values, non_blocking));
}

// TODO: 将此函数加入公共API
inline bool is_same_tensor(const Tensor& lhs, const Tensor& rhs) {
  return lhs.unsafeGetTensorImpl() == rhs.unsafeGetTensorImpl();
}

/* 
 * 检查两个稀疏张量是否具有相同的稀疏结构
 * 即稀疏维度和稠密维度数是否相同
 */
inline bool is_same_density(const SparseTensor& self, const SparseTensor& src) {
  return self.sparse_dim() == src.sparse_dim() && 
         self.dense_dim() == src.dense_dim();
}

/*
 * 创建一个新的values张量
 * 保持原有values的维度结构，但改变非零元素数量(nnz)
 * TODO: 未来应将其公开为ATen API
 * 注意：不保留原数据
 */
inline Tensor new_values_with_size_of(const Tensor& values, int64_t nnz) {
  std::vector<int64_t> size = values.sizes().vec();
  size[0] = nnz;  // 第一个维度设置为新的nnz
  return at::empty(size, values.options());
}

/*
 * [扁平化稀疏索引说明]
 * 将n维稀疏索引张量展平为1维索引
 * 例如：
 *   input = [[2, 4, 0],  // 2行3列的索引矩阵
 *           [3, 1, 10]]
 *   full_size = [2, 12]   // 完整张量尺寸
 *   output = [2*12+3, 4*12+1, 0*12+10] = [27, 49, 10]
 *
 * 参数force_clone强制返回克隆结果
 */
inline LongTensor flatten_indices(const Tensor& indices, IntList full_size, bool force_clone = false) {
  int64_t sparse_dim = indices.size(0);  // 获取稀疏维度数
  
  // 单稀疏维度的简单情况
  if (sparse_dim == 1) {
    return force_clone ? indices.squeeze(0).clone() : indices.squeeze(0);
  } 
  // 多稀疏维度情况
  else {
    // 计算各维度的乘数因子
    std::vector<int64_t> indices_mult_cpu_vec(sparse_dim);
    int64_t mult = 1;
    for (int64_t i = sparse_dim - 1; i >= 0; i--) {
      indices_mult_cpu_vec[i] = mult;
      mult *= full_size[i];
    }
    
    // 创建乘数张量(必须使用阻塞拷贝，避免数据竞争)
    auto indices_mult_cpu = indices.type().cpu()
                           .tensorFromBlob(indices_mult_cpu_vec.data(), {sparse_dim, 1});
    auto indices_mult = indices_mult_cpu.to(indices.device(), /*non_blocking=*/false);
    
    // 通过乘法和求和实现展平(比matmul更快)
    return indices.mul(indices_mult).sum(0);
  }
}

/*
 * 部分维度展平稀疏索引
 * 仅展平指定的维度，结果索引可能未合并
 * 如果输入索引已合并，展平后的索引仍保持排序
 *
 * 参数：
 *   indices: 稀疏张量索引
 *   sizes: 稀疏张量尺寸
 *   dims_to_flatten: 要展平的维度列表
 *
 * 示例1(全展平):
 *   indices = [[2, 4, 0],
 *             [3, 1, 3]]
 *   sizes = [2, 12]
 *   dims_to_flatten = [0, 1]
 *   new_indices = [2*12+3, 4*12+1, 0*12+3] = [27, 49, 3]
 *
 * 示例2(部分展平):
 *   dims_to_flatten = [1]
 *   new_indices = [3, 1, 3] // 未合并
 */
inline LongTensor flatten_indices_by_dims(const LongTensor& indices, 
                                        const IntList& sizes, 
                                        const IntList& dims_to_flatten) {
  LongTensor new_indices = at::zeros({indices.size(1)}, indices.options());
  for (auto d : dims_to_flatten) {
    new_indices.mul_(sizes[d]);  // 乘以当前维度大小
    new_indices.add_(indices.select(0, d));  // 加上当前维度索引
  }
  return new_indices;
}

}} // namespace at::sparse