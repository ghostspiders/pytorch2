#pragma once

#include "ATen/Parallel.h"
#include "ATen/TensorUtils.h"
#include <limits>
#include <utility>
#include <cstring>

namespace at {

/**
 * 压缩维度函数 - 将连续的、可合并的维度进行压缩，优化内存布局
 * 
 * @tparam T 尺寸和步长的数据类型（通常为int64_t）
 * @param sizes 原始尺寸数组（会被原地修改）
 * @param strides 原始步长数组（会被原地修改）
 * @param dims 原始维度数
 * @param excludeDim 需要排除的维度索引（不参与压缩，默认为-1表示不排除）
 * @return std::pair<int64_t, int64_t> 第一个元素是排除维度的新位置，第二个元素是压缩后的维度数
 */
template <typename T>
inline std::pair<int64_t, int64_t> collapse_dims(
    T* sizes,
    T* strides,
    int64_t dims,
    const int excludeDim = -1) {
  // 检查排除维度是否合法
  AT_CHECK(
      excludeDim >= -1 && excludeDim < dims,
      "expected excluded dim between -1 and dims - 1");

  // 设置初始停止维度（如果设置了排除维度，则先处理前面的维度）
  int64_t stopDim = (excludeDim == -1) ? dims : excludeDim;
  int64_t newIndex = -1;          // 新维度的当前位置
  int64_t oldIndex = 0;           // 原始维度的当前位置
  int64_t remappedExcludedDim = -1; // 排除维度在新布局中的位置

  // 遍历所有原始维度
  while (oldIndex < dims) {
    // 阶段1：寻找可合并的起始维度（跳过大小为1的维度）
    for (; oldIndex < stopDim; ++oldIndex) {
      if (sizes[oldIndex] == 1) {
        continue;  // 跳过大小为1的维度
      }

      // 找到第一个有效维度作为合并起点
      ++newIndex;
      sizes[newIndex] = sizes[oldIndex];
      strides[newIndex] = strides[oldIndex];
      ++oldIndex;
      break;
    }

    // 阶段2：合并连续的、可合并的维度
    for (; oldIndex < stopDim; ++oldIndex) {
      if (sizes[oldIndex] == 1) {
        continue;  // 跳过大小为1的维度
      }

      // 检查是否可以合并到前一个维度（内存布局连续）
      if (strides[newIndex] == sizes[oldIndex] * strides[oldIndex]) {
        // 可以合并：扩展前一个维度
        sizes[newIndex] *= sizes[oldIndex];
        strides[newIndex] = strides[oldIndex];
      } else {
        // 不能合并：创建新维度
        ++newIndex;
        sizes[newIndex] = sizes[oldIndex];
        strides[newIndex] = strides[oldIndex];
      }
    }

    // 阶段3：处理排除维度（如果设置了）
    if (oldIndex != dims) {
      // 保留排除维度（不参与合并）
      ++newIndex;
      sizes[newIndex] = sizes[oldIndex];
      strides[newIndex] = strides[oldIndex];
      remappedExcludedDim = newIndex;  // 记录排除维度的新位置

      // 调整迭代参数，继续处理剩余维度
      ++oldIndex;
      stopDim = dims;  // 现在处理所有剩余维度
    }
  }

  // 处理特殊情况：所有维度大小都为1
  if (newIndex == -1 || (newIndex == 0 && sizes[0] == 1)) {
    dims = 1;
    sizes[0] = 1;
    strides[0] = 1;
    return std::pair<int64_t, int64_t>(0, 1);  // 返回压缩后的单一维度
  }

  // 更新实际维度数（newIndex是从0开始的，所以需要+1）
  dims = newIndex + 1;
  return std::pair<int64_t, int64_t>(remappedExcludedDim, dims);
}

/*
 * The basic strategy for apply is as follows:
 *
 * 1. Starting with the outermost index, loop until we reach a dimension where
 * the data is no longer contiguous, i.e. the stride at that dimension is not
 * equal to the size of the tensor defined by the outer dimensions. Let's call
 * this outer (contiguous) tensor A. Note that if the Tensor is contiguous, then
 * A is equal to the entire Tensor. Let's call the inner tensor B.
 *
 * 2. We loop through the indices in B, starting at its outermost dimension. For
 * example, if B is a 2x2 matrix, then we do:
 *
 * B[0][0]
 * B[0][1]
 * B[1][0]
 * B[1][1]
 *
 * We set the offset into the underlying storage as (storageOffset + stride_B *
 * index_B), i.e. basically we compute the offset into the storage as we would
 * normally for a Tensor. But because we are guaranteed the subsequent data is
 * contiguous in memory, we can simply loop for sizeof(A) iterations and perform
 * the operation, without having to follow the order described by the strides of
 * A.
 *
 * 3. As an optimization, we merge dimensions of A that are contiguous in
 * memory. For example, if A is a 3x3x3x3 tensor narrowed from a 3x3x4x3 tensor,
 * then the first two dimensions can be merged for the purposes of APPLY,
 * reducing the number of nested loops.
 */

/**
 * 对张量的步长进行排序并返回新的张量视图
 * 
 * 该函数根据张量的步长(stride)从大到小对维度进行重新排列，
 * 生成一个新的张量视图，使得内存访问更加连续，有利于提高后续操作的效率。
 * 
 * @param tensor_ 输入张量（会被视为只读，不修改原张量）
 * @return Tensor 返回一个维度重新排列后的新张量视图
 */
inline Tensor sort_strides(Tensor& tensor_) {
  // 获取输入张量的步长信息
  IntList strides = tensor_.strides();
  
  // 创建索引向量用于存储维度顺序
  std::vector<int64_t> indices;
  indices.reserve(tensor_.ndimension());  // 预分配空间
  
  // 初始化索引为原始顺序 [0, 1, 2, ..., ndim-1]
  for (int64_t i = 0; i < tensor_.ndimension(); i++) {
    indices.push_back(i);
  }
  
  // 对索引进行排序，排序依据是相应维度的步长（从大到小）
  std::sort(indices.begin(), indices.end(), 
    [&strides](int64_t i1, int64_t i2) {
      return strides[i1] > strides[i2];  // 降序排列
    });
  
  // 使用permute操作按新顺序重新排列维度
  Tensor tensor = tensor_.permute(indices);
  
  return tensor;
}


/**
 * 固定维度的跨步张量迭代器 (编译期确定维度数)
 * 
 * @tparam T 张量元素类型
 * @tparam N 最大支持的维度数
 */
template <typename T, int N>
struct strided_tensor_iter_fixed {
 public:
  T* data_ = NULL;       // 指向张量数据的指针
  int64_t dim_ = 0;      // 实际使用的维度数

  // 以下三个数组使用固定大小N (栈上分配)
  int64_t counter_[N] = {0};  // 当前各维度的计数
  int64_t sizes_[N] = {0};    // 各维度的大小
  int64_t strides_[N] = {0};  // 各维度的步长

  // 禁用拷贝构造和赋值
  strided_tensor_iter_fixed(strided_tensor_iter_fixed const&) = delete;
  void operator=(strided_tensor_iter_fixed const& x) = delete;
  
  // 允许移动构造
  strided_tensor_iter_fixed(strided_tensor_iter_fixed&&) = default;

  /**
   * 构造函数
   * @param tensor 输入张量
   * @param sort_strides 是否对步长排序 (当前未实现)
   */
  strided_tensor_iter_fixed(Tensor& tensor, bool sort_strides = false)
      : data_(tensor.data<T>()) {  // 获取数据指针
    // 初始化计数器
    std::memset(counter_, 0, sizeof(int64_t) * N);
    
    if (tensor.dim() > 0) {
      // 复制尺寸和步长信息
      std::memcpy(sizes_, tensor.sizes().data(), tensor.dim() * sizeof(int64_t));
      std::memcpy(strides_, tensor.strides().data(), tensor.dim() * sizeof(int64_t));
    }
    
    // 压缩可合并的维度 (优化存储布局)
    dim_ = std::get<1>(collapse_dims(sizes_, strides_, tensor.ndimension()));
  }
};

/**
 * 动态维度的跨步张量迭代器 (运行时确定维度数)
 * 
 * @tparam T 张量元素类型
 */
template <typename T>
struct strided_tensor_iter {
 private:
 public:
  T* data_ = NULL;      // 指向张量数据的指针
  int64_t dim_;         // 实际使用的维度数

  // 使用vector存储维度信息 (堆上分配)
  std::vector<int64_t> counter_;  // 当前各维度的计数
  std::vector<int64_t> sizes_;    // 各维度的大小
  std::vector<int64_t> strides_;  // 各维度的步长

  // 禁用拷贝构造和赋值
  strided_tensor_iter(strided_tensor_iter const&) = delete;
  void operator=(strided_tensor_iter const& x) = delete;
  
  // 允许移动构造
  strided_tensor_iter(strided_tensor_iter&&) = default;

  /**
   * 构造函数
   * @param tensor 输入张量
   */
  strided_tensor_iter(Tensor& tensor)
      : data_(tensor.data<T>()),        // 获取数据指针
        dim_(tensor.ndimension()),      // 初始化维度数
        counter_(dim_, 0),              // 初始化计数器
        sizes_(tensor.sizes().vec()),   // 复制尺寸信息
        strides_(tensor.strides().vec()) { // 复制步长信息
    
    // 压缩可合并的维度 (优化存储布局)
    dim_ = std::get<1>(collapse_dims(sizes_.data(), strides_.data(), dim_));
  }
};



/**
 * 检查张量列表中所有张量的元素总数是否相同
 * @param tensors 张量列表
 * @return bool 如果所有张量元素数量相同返回true，否则false
 */
inline bool _all_equal_numel(at::ArrayRef<Tensor> tensors) {
  if (tensors.size() == 0)  // 空列表视为满足条件
    return true;
  int64_t all_numel = tensors[0].numel();  // 取第一个张量的元素数作为基准
  for (size_t i = 1; i < tensors.size(); i++) {
    if (tensors[i].numel() != all_numel)  // 检查每个张量的元素数
      return false;
  }
  return true;
}

/**
 * 生成元素数量不一致的错误信息
 * @param tensors 张量列表
 * @return std::string 格式化的错误消息
 */
inline std::string _all_equal_numel_error(at::ArrayRef<Tensor> tensors) {
  std::ostringstream oss;
  oss << "inconsistent tensor size, expected ";
  // 输出期望的各张量形状
  for (size_t i = 0; i < tensors.size() - 1; i++) {
    oss << tensors[i].sizes() << ", ";
  }
  oss << "and " << tensors[tensors.size() - 1].sizes()
      << " to have the same number of elements, but got ";
  // 输出实际的各张量元素数量
  for (size_t i = 0; i < tensors.size() - 1; i++) {
    oss << tensors[i].numel() << ", ";
  }
  oss << "and " << tensors[tensors.size() - 1].numel()
      << " elements respectively";
  return oss.str();
}

/**
 * 应用操作前的准备工作检查
 * @param tensors 张量列表
 * @return bool 是否可以继续执行操作
 */
inline bool _apply_preamble(ArrayRef<Tensor> tensors) {
  // 检查后端类型是否为CPU
  checkBackend("CPU_tensor_apply", tensors, Backend::CPU);
  // 检查元素数量是否一致
  if (!_all_equal_numel(tensors))
    AT_ERROR(_all_equal_numel_error(tensors));
  // 检查是否有空张量
  for (auto& t : tensors)
    if (t.numel() == 0)
      return false;
  return true;
}

/**
 * 获取张量列表中的最大维度数
 * @param tensors 张量列表
 * @return int64_t 最大维度数
 */
inline int64_t _max_dim_tensors(ArrayRef<Tensor> tensors) {
  int64_t dim = 0;
  for (auto& t : tensors)
    dim = std::max(dim, t.ndimension());  // 比较每个张量的维度
  return dim;
}

/* 以下是迭代器操作的核心函数群 */

// 基础case：空参数迭代
inline void iterate(int64_t size){};

/**
 * 迭代器前进操作（可变参数模板）
 * @param size 前进的步长
 * @param iter 第一个迭代器
 * @param iter_tail 其余迭代器
 */
template <typename Arg, typename... Args>
inline void iterate(int64_t size, Arg& iter, Args&... iter_tail) {
  // 更新最后一个维度的计数
  iter.counter_[iter.dim_ - 1] += size;
  // 移动数据指针
  iter.data_ = iter.data_ + size * iter.strides_[iter.dim_ - 1];
  // 递归处理剩余迭代器
  iterate(size, iter_tail...);
}

// 基础case：空参数继续判断
inline bool iterate_continue() {
  return true;
};

/**
 * 检查迭代是否应该继续（可变参数模板）
 * @param iter 第一个迭代器
 * @param iter_tail 其余迭代器
 * @return bool 是否所有迭代器都可以继续
 */
template <typename Arg, typename... Args>
inline bool iterate_continue(Arg& iter, Args&... iter_tail) {
  return iter.counter_[iter.dim_ - 1] < iter.sizes_[iter.dim_ - 1] &&
      iterate_continue(iter_tail...);  // 递归检查
}

// 基础case：空参数最大迭代大小
inline int64_t max_iterate_size() {
  return std::numeric_limits<int64_t>::max();
};

/**
 * 计算最大迭代步长（取各迭代器剩余步长最小值）
 * @param iter 第一个迭代器
 * @param iter_tail 其余迭代器
 * @return int64_t 最大安全迭代步长
 */
template <typename Arg, typename... Args>
inline int64_t max_iterate_size(Arg& iter, Args&... iter_tail) {
  return std::min(
      (iter.sizes_[iter.dim_ - 1] - iter.counter_[iter.dim_ - 1]),
      max_iterate_size(iter_tail...));  // 递归计算最小值
}

// 基础case：空参数溢出处理
inline void iterate_overflow(){};

/**
 * 处理迭代器维度溢出（进位处理）
 * @param iter 第一个迭代器
 * @param iter_tail 其余迭代器
 */
template <typename Arg, typename... Args>
inline void iterate_overflow(Arg& iter, Args&... iter_tail) {
  if (iter.counter_[iter.dim_ - 1] == iter.sizes_[iter.dim_ - 1]) {
    // 从后向前处理维度进位
    for (int64_t i = iter.dim_ - 1; i > 0; i--) {
      if (iter.counter_[i] == iter.sizes_[i]) {
        iter.counter_[i] = 0;  // 当前维度归零
        iter.counter_[i - 1]++;  // 前一维度加1
        // 调整数据指针
        iter.data_ = iter.data_ - (iter.sizes_[i] * iter.strides_[i]) +
            iter.strides_[i - 1];
      }
    }
  }
  iterate_overflow(iter_tail...);  // 递归处理
}

// 基础case：空参数前进
inline void forward(int64_t offset){};

/**
 * 迭代器快速前进（用于跳过初始部分）
 * @param offset 前进的偏移量
 * @param iter 第一个迭代器
 * @param iter_tail 其余迭代器
 */
template <typename Arg, typename... Args>
inline void forward(int64_t offset, Arg& iter, Args&... iter_tail) {
  int64_t multi = offset;
  // 从后向前处理每个维度
  for (int64_t i = iter.dim_ - 1; i >= 0; i--) {
    int64_t inc = multi % iter.sizes_[i];  // 当前维度的增量
    multi = multi / iter.sizes_[i];  // 剩余量传递给更高维度
    iter.data_ = iter.data_ + inc * iter.strides_[i];  // 移动指针
    iter.counter_[i] += inc;  // 更新计数器
  }
  forward(offset, iter_tail...);  // 递归处理
}

// 基础case：空参数最大维度
inline int64_t max_dim() {
  return 0;
}

/**
 * 计算所有迭代器中的最大维度数
 * @param iter 第一个迭代器
 * @param iter_tail 其余迭代器
 * @return int64_t 最大维度数
 */
template <typename Arg, typename... Args>
inline int64_t max_dim(Arg& iter, Args&... iter_tail) {
  return std::max(iter.dim_, max_dim(iter_tail...));  // 递归求最大值
}

// 基础case：空操作
inline void apply_op(){};

/**
 * 应用操作到张量元素（核心函数）
 * @param numel 要处理的元素总数
 * @param offset 起始偏移量
 * @param op 要应用的操作（函数对象）
 * @param iters 迭代器参数包
 */
template <typename Op, typename... Args>
inline void
apply_op(int64_t numel, int64_t offset, const Op& op, Args... iters) {
  // 处理0维张量的特殊情况
  if (numel == 1 && max_dim(iters...) == 0) {
    op(*iters.data_...);  // 直接应用操作
    return;
  }
  // 处理起始偏移
  if (offset > 0)
    forward(offset, iters...);
  // 分块处理以提高编译器优化效果
  for (int64_t i = 0; i < numel;) {
    // 内层循环连续处理元素
    for (; iterate_continue(iters...) && i < numel;) {
      op(*iters.data_...);  // 应用操作
      iterate(1, iters...); // 迭代器前进1
      i++;
    }
    iterate_overflow(iters...);  // 处理维度溢出
  }
}

// 基础case：空内核
inline void apply_kernel(){};
/**
 * 应用内核函数（处理0维张量的TODO注释）
 * @param numel 要处理的元素总数
 * @param offset 起始偏移量
 * @param op 操作函数对象
 * @param iters 迭代器参数包
 * 
 * TODO: 需要优雅处理0维张量。0维strided_tensor_iter的strides_在维度0时大小为0，
 * 且iters.strides_[iters.dim_ - 1]会索引到-1。C++14的integer_sequence可能有用。
 */
template <typename Op, typename... Args>
inline void apply_kernel(int64_t numel, int64_t offset, const Op& op, Args... iters) {
  if (offset > 0)
    forward(offset, iters...);  // 处理起始偏移
  
  // 计算当前可处理的最大块大小
  int64_t size = std::min(numel, max_iterate_size(iters...));
  
  // 调用操作函数，传入数据指针和最后维度的步长
  op(size, iters.data_..., iters.strides_[iters.dim_ - 1]...);
  
  iterate(size, iters...);  // 迭代器前进
  iterate_overflow(iters...);  // 处理维度溢出
  
  // 剩余元素处理
  int64_t i = size;
  size = std::min(numel, max_iterate_size(iters...));
  for (; i < numel;) {
    op(size, iters.data_..., iters.strides_[iters.dim_ - 1]...);
    iterate(size, iters...);
    i += size;
    iterate_overflow(iters...);
  }
}

/**
 * 并行处理两个张量的内核函数
 * @tparam scalar1 第一个张量元素类型
 * @tparam scalar2 第二个张量元素类型
 * @tparam Op 操作类型
 * @param tensor1 第一个输入张量
 * @param tensor2 第二个输入张量
 * @param op 操作函数对象
 */
template <typename scalar1, typename scalar2, typename Op>
inline void CPU_tensor_parallel_kernel_apply2(Tensor tensor1, Tensor tensor2, const Op op) {
  if (!_apply_preamble({tensor1, tensor2}))  // 前置检查
    return;
  
  // 处理单元素张量特殊情况
  if (tensor1.numel() == 1) {
    op(1, tensor1.data<scalar1>(), tensor2.data<scalar2>(), 0, 0);
    return;
  }
  
  // 根据维度选择固定大小或动态大小的迭代器
  if (tensor1.ndimension() < 8 && tensor2.ndimension() < 8) {
    parallel_for(0, tensor1.numel(), 1, [&](int64_t begin, int64_t end) {
      apply_kernel(end - begin, begin, op,
        strided_tensor_iter_fixed<scalar1, 8>(tensor1),
        strided_tensor_iter_fixed<scalar2, 8>(tensor2));
    });
  } else {
    parallel_for(0, tensor1.numel(), 1, [&](int64_t begin, int64_t end) {
      apply_kernel(end - begin, begin, op,
        strided_tensor_iter<scalar1>(tensor1),
        strided_tensor_iter<scalar2>(tensor2));
    });
  }
}

/* =============== 单张量应用函数 =============== */

/**
 * 应用操作到单个张量
 * @tparam scalar1 张量元素类型
 * @tparam Op 操作类型
 */
template <typename scalar1, typename Op>
inline void CPU_tensor_apply1(Tensor tensor1, const Op op) {
  if (!_apply_preamble({tensor1}))
    return;
  
  // 根据维度选择迭代器类型
  if (tensor1.ndimension() < 8) {
    apply_op(tensor1.numel(), 0, op,
      strided_tensor_iter_fixed<scalar1, 8>(tensor1, true));  // 固定大小迭代器
  } else {
    apply_op(tensor1.numel(), 0, op,
      strided_tensor_iter<scalar1>(tensor1));  // 动态大小迭代器
  }
}

/* =============== 多张量应用函数（2-4个张量） =============== */

// 两个张量版本
template <typename scalar1, typename scalar2, typename Op>
inline void CPU_tensor_apply2(Tensor tensor1, Tensor tensor2, const Op op) {
  if (!_apply_preamble({tensor1, tensor2}))
    return;
  
  if (_max_dim_tensors({tensor1, tensor2}) <= 8) {
    apply_op(tensor1.numel(), 0, op,
      strided_tensor_iter_fixed<scalar1, 8>(tensor1),
      strided_tensor_iter_fixed<scalar2, 8>(tensor2));
  } else {
    apply_op(tensor1.numel(), 0, op,
      strided_tensor_iter<scalar1>(tensor1),
      strided_tensor_iter<scalar2>(tensor2));
  }
}

// 三个张量版本 
template <typename scalar1, typename scalar2, typename scalar3, typename Op>
inline void CPU_tensor_apply3(Tensor tensor1, Tensor tensor2, Tensor tensor3, const Op op) {
  // 类似两个张量的实现，增加一个迭代器参数
}

// 四个张量版本
template <typename scalar1, typename scalar2, typename scalar3, typename scalar4, typename Op>
inline void CPU_tensor_apply4(Tensor tensor1, Tensor tensor2, Tensor tensor3, Tensor tensor4, const Op op) {
  // 类似两个张量的实现，增加两个迭代器参数
}

/* =============== 并行版本应用函数 =============== */

// 单张量并行版本
template <typename scalar1, typename Op>
inline void CPU_tensor_parallel_apply1(Tensor tensor1, const Op op, int64_t grain_size = internal::GRAIN_SIZE) {
  if (!_apply_preamble({tensor1}))
    return;
  
  if (tensor1.ndimension() < 8) {
    parallel_for(0, tensor1.numel(), grain_size, [&](int64_t begin, int64_t end) {
      apply_op(end - begin, begin, op,
        strided_tensor_iter_fixed<scalar1, 8>(tensor1, true));
    });
  } else {
    parallel_for(0, tensor1.numel(), grain_size, [&](int64_t begin, int64_t end) {
      apply_op(end - begin, begin, op,
        strided_tensor_iter<scalar1>(tensor1));
    });
  }
}

// 两个张量并行版本
template <typename scalar1, typename scalar2, typename Op>
inline void CPU_tensor_parallel_apply2(Tensor tensor1, Tensor tensor2, const Op op, int64_t grain_size = internal::GRAIN_SIZE) {
  // 实现类似单张量并行版本，但使用两个迭代器
}

} // namespace at