#include "IndexUtils.cuh"  // CUDA索引工具头文件

namespace at {
namespace cuda {
namespace detail {

// 结构体：存储张量的尺寸(size)和步长(stride)
struct SizeAndStride {
  int64_t size;    // 维度大小
  int64_t stride;  // 维度步长
};

/*
 * 比较函数：用于对SizeAndStride结构体按stride升序排序
 * 参数：两个待比较的void指针，实际指向SizeAndStride结构体
 * 返回值：-1(a<b), 0(a==b), 1(a>b)
 */
int compareSizeAndStride(const void* a, const void* b) {
  const SizeAndStride* aS = (const SizeAndStride*) a;
  const SizeAndStride* bS = (const SizeAndStride*) b;

  if (aS->stride < bS->stride) return -1;
  if (aS->stride == bS->stride) return 0;
  return 1;
}

/*
 * 检查张量是否可能存在索引重叠
 * 参数：张量引用
 * 返回值：true-可能存在重叠，false-绝对不存在重叠
 * 
 * 原理：检查张量是否满足"无重叠"的充分条件(非必要条件)
 *       即是否存在一种维度排列方式，使得每个维度都被下一个维度"嵌套"
 */
bool maybeOverlappingIndices(const Tensor& t) {
  /* 提取size>1的维度信息到临时数组 */
  SizeAndStride *info = (SizeAndStride *)alloca(sizeof(SizeAndStride) * t.dim());
  int dims = t.dim();
  int nonSize1Dims = 0;  // 记录size>1的维度数量
  
  for (int i = 0; i < dims; ++i) {
    int64_t size = t.size(i);
    if (size > 1) {
      info[nonSize1Dims].size = size;
      info[nonSize1Dims].stride = t.stride(i);

      // 如果步长为负，直接判定可能存在重叠
      if (info[nonSize1Dims].stride < 1) {
        return true;
      }

      ++nonSize1Dims;
    }
  }

  // 单元素张量绝对无重叠
  if (nonSize1Dims == 0) {
    return false;
  }

  /* 按步长升序排序(最内层维度排在info[0]) */
  qsort(info, nonSize1Dims, sizeof(SizeAndStride), compareSizeAndStride);

  /* 检查相邻维度是否满足嵌套关系 */
  for (int i = 0; i < (nonSize1Dims - 1); ++i) {
    // 如果当前维度的最大偏移 >= 下一个维度的步长，可能存在重叠
    if (((info[i].size - 1) * info[i].stride) >= info[i + 1].stride) {
      return true;
    }
  }

  return false;
}

/*
 * 检查张量是否可以使用32位索引计算
 * 参数：张量引用，最大元素限制
 * 返回值：true-可以使用32位索引，false-不能使用
 * 
 * 原理：检查两个条件：
 *       1. 元素总数不超过max_elem
 *       2. 最大线性偏移不超过max_elem
 */
bool canUse32BitIndexMath(const Tensor& t, int64_t max_elem) {
  int64_t elements = t.numel();
  // 条件1：元素总数检查
  if (elements >= max_elem) {
    return false;
  }

  // 计算最大线性索引对应的偏移量
  int64_t offset = 0;
  int64_t linearId = elements - 1;  // 最大线性索引

  // 反向遍历维度计算偏移
  for (int i = t.dim() - 1; i >= 0; --i) {
    int64_t curDimIndex = linearId % t.size(i);  // 当前维度索引
    int64_t curDimOffset = curDimIndex * t.stride(i);  // 当前维度偏移
    offset += curDimOffset;
    linearId /= t.size(i);  // 移向下一个更高维度
  }

  // 条件2：最大偏移检查
  if (offset >= max_elem) {
    return false;
  }

  return true;
}

} // namespace detail
} // namespace cuda
} // namespace at