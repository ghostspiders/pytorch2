#include "ATen/ExpandUtils.h"

namespace at {

// 实现两个形状的广播规则推断
std::vector<int64_t> infer_size(IntList a, IntList b) {
  // 确定输出张量的维度数（取较大者）
  auto dimsA = a.size();
  auto dimsB = b.size();
  ptrdiff_t ndim = dimsA > dimsB ? dimsA : dimsB;
  std::vector<int64_t> expandedSizes(ndim);

  // 从尾部开始逐维度处理（广播规则对齐方式）
  for (long i = ndim - 1; i >= 0; --i) {
    // 计算当前维度在两个形状中的对应位置
    long offset = ndim - 1 - i;
    long dimA = dimsA - 1 - offset;
    long dimB = dimsB - 1 - offset;
    
    // 获取维度大小（越界维度视为1，实现自动补齐）
    long sizeA = (dimA >= 0) ? a[dimA] : 1;
    long sizeB = (dimB >= 0) ? b[dimB] : 1;

    // 核心广播规则检查：
    // 1. 维度相等 或 
    // 2. 其中一方为1（可广播）
    AT_CHECK(
        sizeA == sizeB || sizeA == 1 || sizeB == 1,
        "The size of tensor a (", sizeA,
        ") must match the size of tensor b (", sizeB,
        ") at non-singleton dimension ", i);

    // 广播规则应用：优先保留非1的尺寸
    expandedSizes[i] = sizeA == 1 ? sizeB : sizeA;
  }

  return expandedSizes;
}

// 推断扩展后的形状和步长（内存布局）
std::tuple<std::vector<int64_t>, std::vector<int64_t>> inferExpandGeometry(
    IntList tensor_sizes,    // 原始形状
    IntList tensor_strides,  // 原始步长
    IntList sizes) {         // 目标形状
  int64_t ndim = sizes.size();
  int64_t tensor_dim = tensor_sizes.size();

  // 处理标量张量（0维）的特殊情况
  if (tensor_dim == 0) {
    std::vector<int64_t> expandedStrides(ndim, 0);  // 标量扩展后步长全为0
    return std::make_tuple(sizes.vec(), expandedStrides);
  }

  std::vector<int64_t> expandedSizes(ndim);
  std::vector<int64_t> expandedStrides(ndim);

  // 从尾部开始构造新的内存布局
  for (int64_t i = ndim - 1; i >= 0; --i) {
    int64_t offset = ndim - 1 - i;
    int64_t dim = tensor_dim - 1 - offset;
    
    // 获取原始尺寸和步长（越界维度处理）
    int64_t size = (dim >= 0) ? tensor_sizes[dim] : 1;
    int64_t stride = (dim >= 0) ? tensor_strides[dim]
                     : expandedSizes[i + 1] * expandedStrides[i + 1]; // 虚拟步长计算
    
    // 处理-1特殊值（表示继承原始尺寸）
    int64_t targetSize = sizes[i];
    if (targetSize == -1) {
      AT_CHECK(
          dim >= 0,
          "The expanded size of the tensor (",
          targetSize,
          ") isn't allowed in a leading, non-existing dimension ",
          i);
      targetSize = size;  // 保持原始尺寸
    }

    // 尺寸不匹配时的处理逻辑
    if (size != targetSize) {
      AT_CHECK(
          size == 1,  // 只有原始尺寸为1时才允许扩展
          "The expanded size of the tensor (",
          targetSize,
          ") must match the existing size (",
          size,
          ") at non-singleton dimension ",
          i,
          ".  Target sizes: ",
          sizes,
          ".  Tensor sizes: ",
          tensor_sizes);
      size = targetSize;
      stride = 0;  // 广播维度的步长设为0（内存优化关键）
    }
    
    // 记录结果
    expandedSizes[i] = size;
    expandedStrides[i] = stride;
  }
  
  return std::make_tuple(expandedSizes, expandedStrides);
}

} // namespace at