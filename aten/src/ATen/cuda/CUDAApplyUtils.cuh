#pragma once

#include "ATen/cuda/detail/IndexUtils.cuh"
#include "ATen/TensorUtils.h"
#include "THC/THCAtomics.cuh"
#include "ATen/cuda/CUDAContext.h"

#include <math.h>

//
// 本文件包含逐点操作函数和内核，
// 可处理任意维度（最高MAX_CUTORCH_DIMS）的张量参数，
// 无论内存是否连续，均无需复制或临时存储。
//

/*
  说明 [ CUDA_tensor_applyN 辅助函数 ]

  下列CUDA_tensor_applyN函数（当前N可为1、2、3或4）
  对N个张量执行逐点操作。

  调用约定：

  1. 模板参数应按顺序指定：
    - 前N个类型参数指定每个张量的标量类型
    - （可选）`int step`参数指定每次迭代处理的元素数量
      默认值为1
    - （通常可省略的）类型参数指定应用于每次迭代中
      `N*step`个值的函数/函子类型

  2. 函数参数应按顺序提供：
    - N个张量
    - op：处理`N*step`个值的函数/函子
      - 若`step == 1`，需满足签名：
        `void(*)(scalar1_t&, scalar2_t&, ..., scalarN_t&)`
        其中输入为N个张量在相同索引处的值
      - 否则需满足签名：
        `void(*)(int n, scalar1_t&, ..., scalar1_t&,  // 重复step次
                scalar2_t&, ..., scalar2_t&,          // 重复step次
                ...,
                scalarN_t&, ..., scalarN_t&)`         // 重复step次
        此时会处理来自step个连续索引的`N*step`个值。
        首参数n表示有效元素数（0 < n <= step），
        边界情况可能不足step个元素。

        例如当`step == 4`且`N == 2`时，op可以是：
          [](int n, scalar1_t &u1, ..., scalar1_t &u4,
                    scalar2_t &v1, ..., scalar2_t &v4) {
            // 仅处理前n个u和v元素
            // 当n==3时，u4/v4无需处理
          }

      两种情况下引用实际可为const，但至少需一个非const引用以写入结果
    - （可选但推荐）N个TensorArgType参数，指定每个张量是
      读写（TensorArgType::ReadWrite）还是只读（TensorArgType::ReadOnly）
      默认为：第一个张量ReadWrite，其余ReadOnly

  示例：

  计算a = b²（同数据类型）：
  CUDA_tensor_apply2<scalar, scalar>(
    a, b,
    [] __device__ (scalar &a_val, const scalar &b_val) { 
      a_val = b_val * b_val; 
    }
  );

  批量处理2个元素：
  CUDA_tensor_apply2<scalar1, scalar2, 2>(
    a, b,
    [] __device__ (int n, scalar1 &a_val1, scalar1 &a_val2,
                          const scalar2 &b_val1, const scalar2 &b_val2) {
      // 调用向量化操作，或直接逐元素处理以利用循环展开...
      // 当n==1时仅需处理a_val1和b_val1
    }
  );
*/

namespace at {
namespace cuda {

// TODO: combine with TensorArg?  So far that's been for debugging, and this is functional...
enum class TensorArgType { ReadWrite, ReadOnly };

namespace {

// 为逐点操作重新排列维度，尽可能使步长(strides)呈递减顺序，
// 从而优化内核的内存访问模式。

// 举例说明：假设有两个转置后的2维张量进行二元运算：
// 尺寸: 256 512
// aInfo->步长: 1 256
// bInfo->步长: 1 256

// 这种情况下，kernelPointwiseApply2()中的每次并发内存访问
// 都相隔256个元素，导致性能低下。

// 本函数通过交换维度使内存访问连续：
// 尺寸: 512 256
// aInfo->步长: 256 1
// bInfo->步长: 256 1

// （实际效果更好，因为此时collapseDims()能将每个输入
// 转换为连续内存数组）

// 通用情况：给定M(<=4)个N维TensorInfo结构体，
// 我们可以将每个strides[i](0 <= i < N)视为M元组。
// 对于任意i < j这对维度，当满足以下条件时交换strides[i]和[j]：
// (1) 存在某个k(0 <= k < M)使得strides[i][k] < strides[j][k]
// （交换将有利于第k个输入的内存访问）
// (2) 对所有k都有strides[i][k] <= strides[j][k]
// （交换不会使任何输入的内存访问模式恶化）
template <typename T1, typename IndexType,
          typename T2 = void, typename T3 = void, typename T4 = void>
inline void rearrangeDims(detail::TensorInfo<T1, IndexType>* aInfo,
                          detail::TensorInfo<T2, IndexType>* bInfo = nullptr,
                          detail::TensorInfo<T3, IndexType>* cInfo = nullptr,
                          detail::TensorInfo<T4, IndexType>* dInfo = nullptr) {
  int numInfos = 1;
  int dims = aInfo->dims;
  IndexType *sizes[4] = { aInfo->sizes, };
  IndexType *strides[4] = { aInfo->strides, };

  if (bInfo != nullptr) {
    ++numInfos;
    if (bInfo->dims != dims) return;
    sizes[1] = bInfo->sizes;
    strides[1] = bInfo->strides;
  }

  if (cInfo != nullptr) {
    ++numInfos;
    if (cInfo->dims != dims) return;
    sizes[2] = cInfo->sizes;
    strides[2] = cInfo->strides;
  }

  if (dInfo != nullptr) {
    ++numInfos;
    if (dInfo->dims != dims) return;
    sizes[3] = dInfo->sizes;
    strides[3] = dInfo->strides;
  }

  // Bail out if sizes do not match: we are using "deprecated pointwise
  // behavior" among tensors of different shapes but same number of elements.
  for (int i = 1; i < numInfos; ++i) {
    for (int j = 0; j < dims; ++j) {
      if (sizes[i][j] != sizes[0][j]) return;
    }
  }

  for (int i = 0; i < dims - 1; ++i) {
    // No need to consider dimensions of size 1.
    if (sizes[0][i] == 1) continue;

    for (int j = i + 1; j < dims; ++j) {
      if (sizes[0][j] == 1) continue;

      // Compare the relative sizes of strides between dim #i and dim #j.
      bool hasIncreasingStrides = false;
      bool hasDecreasingStrides = false;

      for (int k = 0; k < numInfos; k++) {
        IndexType stride_i = strides[k][i];
        IndexType stride_j = strides[k][j];
        if (stride_i < stride_j) {
          hasIncreasingStrides = true;
        } else if (stride_i > stride_j) {
          hasDecreasingStrides = true;
        }
      }

      if (hasIncreasingStrides && !hasDecreasingStrides) {
        for (int k = 0; k < numInfos; k++) {
          IndexType size = sizes[k][i];
          sizes[k][i] = sizes[k][j];
          sizes[k][j] = size;

          IndexType stride = strides[k][i];
          strides[k][i] = strides[k][j];
          strides[k][j] = stride;
        }
      }
    }
  }
}

// Threads per block for our apply kernel
// FIXME: use occupancy calculator instead
#define AT_APPLY_THREADS_PER_BLOCK 32 * 16
#define AT_APPLY_BLOCKS_PER_SM 4

// The `remaining_steps` argument is used to support Op that operates on
// multiple elements at the same time. Generally, the strategy of ApplyOpN is to
//  1. Initialize `remaining_steps = step`, where `step` is the template arg of
//     CUDA_tensor_applyN helpers. The input arg `n` to `apply()` represents the
//     number of elements in bound for this call. It will almost always equal to
//     `step` except at boundaries.
//  2. If `remaining_steps > 0` convert the current linearIndex to offset (if in
//     bound), and recursively call `ApplyOpN` with `remaining_steps - 1`.
//  3. At `remaining_steps = 0`,
//       if `step = 1`, call `op(tensor1_val, tensor2_val, ...)`;
//       if `step > 1`, call `op(n, tensor1_val1, tensor1_val2, ..., tesor1_valstep,
//                                  tensor2_val1, tensor2_val2, ..., tesor2_valstep,
//                                       ...
//                                  tensorN_val1, tensorN_val2, ..., tesorN_valstep);`
//
// See NOTE [ CUDA_tensor_applyN helpers ] above for how Op may look like.

template <typename Op,
          typename scalar,
          typename IndexType,
          int ADims,
          int remaining_steps,
          typename... Offsets>
struct ApplyOp1 {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar, IndexType> &a, const Op &op, int n,
                  IndexType linearIndex, Offsets... aOffsets) {
  // Convert `linearIndex` into an offset of `a`
  const IndexType aOffset = sizeof...(Offsets) < n ?
    detail::IndexToOffset<scalar, IndexType, ADims>::get(linearIndex, a) : 0;

  ApplyOp1<Op, scalar, IndexType, ADims, remaining_steps - 1, const IndexType, Offsets...>::apply(
    a, op, n, linearIndex + 1, aOffsets..., aOffset
  );
}
};

// Specialize `step=1` case (i.e., `remaining_steps=0` and `len(Offsets)=1`).
// We don't need to pass in how many elements need to processed in this case.
template <typename Op,
          typename scalar,
          typename IndexType,
          int ADims,
          typename Offset>
struct ApplyOp1<Op, scalar, IndexType, ADims, 0, Offset> {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar, IndexType> &a, const Op &op,
                  int n, IndexType linearIndex, Offset offset) {
  op(a.data[offset]);
}
};

template <typename Op,
          typename scalar,
          typename IndexType,
          int ADims,
          typename... Offsets>
struct ApplyOp1<Op, scalar, IndexType, ADims, 0, Offsets...> {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar, IndexType> &a, const Op &op, int n,
                 IndexType linearIndex, Offsets... offsets) {
  op(n, a.data[offsets]...);
}
};

template <typename Op,
          typename scalar,
          typename IndexType,
          int ADims,
          int step>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(AT_APPLY_THREADS_PER_BLOCK, AT_APPLY_BLOCKS_PER_SM)
#endif
__global__ void kernelPointwiseApply1(detail::TensorInfo<scalar, IndexType> a,
                                      IndexType totalElements, const Op op) {
  for (IndexType linearIndex = (blockIdx.x * blockDim.x + threadIdx.x) * step;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x * step) {
    ApplyOp1<Op, scalar, IndexType, ADims, step>::apply(
      a, op, ::min(step, static_cast<int>(totalElements - linearIndex)), linearIndex);
  }
}


template <typename Op,
          typename scalar1,
          typename scalar2,
          typename IndexType,
          int ADims,
          int BDims,
          int remaining_steps,
          typename... Offsets>
struct ApplyOp2 {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar1, IndexType> &a,
                  detail::TensorInfo<scalar2, IndexType> &b,
                  const Op &op, int n, IndexType linearIndex,
                  Offsets... aOffsets, Offsets... bOffsets) {
  // Convert `linearIndex` into an offset of `a`
  const IndexType aOffset = sizeof...(Offsets) < n ?
    detail::IndexToOffset<scalar1, IndexType, ADims>::get(linearIndex, a) : 0;

  // Convert `linearIndex` into an offset of `b`
  const IndexType bOffset = sizeof...(Offsets) < n ?
    detail::IndexToOffset<scalar2, IndexType, BDims>::get(linearIndex, b) : 0;

  ApplyOp2<Op, scalar1, scalar2, IndexType, ADims, BDims, remaining_steps - 1, const IndexType, Offsets...>::apply(
    a, b, op, n, linearIndex + 1, aOffsets..., aOffset, bOffsets..., bOffset
  );
}
};

// Specialize `step=1` case (i.e., `remaining_steps=0` and `len(Offsets)=1`).
// We don't need to pass in how many elements need to processed in this case.
template <typename Op,
          typename scalar1,
          typename scalar2,
          typename IndexType,
          int ADims,
          int BDims,
          typename Offset>
struct ApplyOp2<Op, scalar1, scalar2, IndexType, ADims, BDims, 0, Offset> {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar1, IndexType> &a,
                  detail::TensorInfo<scalar2, IndexType> &b,
                  const Op &op, int n, IndexType linearIndex,
                  Offset aOffset, Offset bOffset) {
  op(a.data[aOffset], b.data[bOffset]);
}
};

template <typename Op,
          typename scalar1,
          typename scalar2,
          typename IndexType,
          int ADims,
          int BDims,
          typename... Offsets>
struct ApplyOp2<Op, scalar1, scalar2, IndexType, ADims, BDims, 0, Offsets...> {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar1, IndexType> &a,
                  detail::TensorInfo<scalar2, IndexType> &b,
                  const Op &op, int n, IndexType linearIndex,
                  Offsets... aOffsets, Offsets... bOffsets) {
  op(n, a.data[aOffsets]..., b.data[bOffsets]...);
}
};

template <typename Op,
          typename scalar1,
          typename scalar2,
          typename IndexType,
          int ADims, int BDims,
          int step>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(AT_APPLY_THREADS_PER_BLOCK, AT_APPLY_BLOCKS_PER_SM)
#endif
__global__ void
kernelPointwiseApply2(detail::TensorInfo<scalar1, IndexType> a,
                      detail::TensorInfo<scalar2, IndexType> b,
                      IndexType totalElements,
                      const Op op) {
  for (IndexType linearIndex = (blockIdx.x * blockDim.x + threadIdx.x) * step;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x * step) {
    ApplyOp2<Op, scalar1, scalar2, IndexType, ADims, BDims, step>::apply(
      a, b, op, ::min(step, static_cast<int>(totalElements - linearIndex)),
      linearIndex);
  }
}


template <typename Op,
          typename scalar1,
          typename scalar2,
          typename scalar3,
          typename IndexType,
          int ADims,
          int BDims,
          int CDims,
          int remaining_steps,
          typename... Offsets>
struct ApplyOp3 {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar1, IndexType> &a,
                  detail::TensorInfo<scalar2, IndexType> &b,
                  detail::TensorInfo<scalar3, IndexType> &c,
                  const Op &op, int n, IndexType linearIndex,
                  Offsets... aOffsets, Offsets... bOffsets,
                  Offsets... cOffsets) {
  // Convert `linearIndex` into an offset of `a`
  const IndexType aOffset = sizeof...(Offsets) < n ?
    detail::IndexToOffset<scalar1, IndexType, ADims>::get(linearIndex, a) : 0;

  // Convert `linearIndex` into an offset of `b`
  const IndexType bOffset = sizeof...(Offsets) < n ?
    detail::IndexToOffset<scalar2, IndexType, BDims>::get(linearIndex, b) : 0;

  // Convert `linearIndex` into an offset of `c`
  const IndexType cOffset = sizeof...(Offsets) < n ?
    detail::IndexToOffset<scalar3, IndexType, CDims>::get(linearIndex, c) : 0;

  ApplyOp3<Op, scalar1, scalar2, scalar3, IndexType, ADims, BDims, CDims,
           remaining_steps - 1, const IndexType, Offsets...>::apply(
    a, b, c, op, n, linearIndex + 1, aOffsets..., aOffset, bOffsets..., bOffset,
    cOffsets..., cOffset
  );
}
};

// Specialize `step=1` case (i.e., `remaining_steps=0` and `len(Offsets)=1`).
// We don't need to pass in how many elements need to processed in this case.
template <typename Op,
          typename scalar1,
          typename scalar2,
          typename scalar3,
          typename IndexType,
          int ADims,
          int BDims,
          int CDims,
          typename Offset>
struct ApplyOp3<Op, scalar1, scalar2, scalar3, IndexType,
                ADims, BDims, CDims, 0, Offset> {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar1, IndexType> &a,
                  detail::TensorInfo<scalar2, IndexType> &b,
                  detail::TensorInfo<scalar3, IndexType> &c,
                  const Op &op, int n, IndexType linearIndex,
                  Offset aOffset, Offset bOffset, Offset cOffset) {
  op(a.data[aOffset], b.data[bOffset], c.data[cOffset]);
}
};

template <typename Op,
          typename scalar1,
          typename scalar2,
          typename scalar3,
          typename IndexType,
          int ADims,
          int BDims,
          int CDims,
          typename... Offsets>
struct ApplyOp3<Op, scalar1, scalar2, scalar3, IndexType,
                ADims, BDims, CDims, 0, Offsets...> {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar1, IndexType> &a,
                  detail::TensorInfo<scalar2, IndexType> &b,
                  detail::TensorInfo<scalar3, IndexType> &c,
                  const Op &op, int n, IndexType linearIndex,
                  Offsets... aOffsets, Offsets... bOffsets,
                  Offsets... cOffsets) {
  op(n, a.data[aOffsets]..., b.data[bOffsets]..., c.data[cOffsets]...);
}
};


template <typename Op,
          typename scalar1,
          typename scalar2,
          typename scalar3,
          typename IndexType,
          int ADims, int BDims, int CDims,
          int step>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(AT_APPLY_THREADS_PER_BLOCK, AT_APPLY_BLOCKS_PER_SM)
#endif
__global__ void
kernelPointwiseApply3(detail::TensorInfo<scalar1, IndexType> a,
                      detail::TensorInfo<scalar2, IndexType> b,
                      detail::TensorInfo<scalar3, IndexType> c,
                      IndexType totalElements,
                      const Op op) {
  for (IndexType linearIndex = (blockIdx.x * blockDim.x + threadIdx.x) * step;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x * step) {
    ApplyOp3<Op, scalar1, scalar2, scalar3, IndexType, ADims, BDims, CDims, step>::apply(
      a, b, c, op, ::min(step, static_cast<int>(totalElements - linearIndex)), linearIndex);
  }
}


template <typename Op,
          typename scalar1,
          typename scalar2,
          typename scalar3,
          typename scalar4,
          typename IndexType,
          int ADims,
          int BDims,
          int CDims,
          int DDims,
          int remaining_steps,
          typename... Offsets>
struct ApplyOp4 {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar1, IndexType> &a,
                  detail::TensorInfo<scalar2, IndexType> &b,
                  detail::TensorInfo<scalar3, IndexType> &c,
                  detail::TensorInfo<scalar4, IndexType> &d,
                  const Op &op, int n, IndexType linearIndex,
                  Offsets... aOffsets, Offsets... bOffsets,
                  Offsets... cOffsets, Offsets... dOffsets) {
  // Convert `linearIndex` into an offset of `a`
  const IndexType aOffset = sizeof...(Offsets) < n ?
    detail::IndexToOffset<scalar1, IndexType, ADims>::get(linearIndex, a) : 0;

  // Convert `linearIndex` into an offset of `b`
  const IndexType bOffset = sizeof...(Offsets) < n ?
    detail::IndexToOffset<scalar2, IndexType, BDims>::get(linearIndex, b) : 0;

  // Convert `linearIndex` into an offset of `c`
  const IndexType cOffset = sizeof...(Offsets) < n ?
    detail::IndexToOffset<scalar3, IndexType, CDims>::get(linearIndex, c) : 0;

  // Convert `linearIndex` into an offset of `d`
  const IndexType dOffset = sizeof...(Offsets) < n ?
    detail::IndexToOffset<scalar4, IndexType, DDims>::get(linearIndex, d) : 0;

  ApplyOp4<Op, scalar1, scalar2, scalar3, scalar4, IndexType,
           ADims, BDims, CDims, DDims, remaining_steps - 1, const IndexType, Offsets...>::apply(
    a, b, c, d, op, n, linearIndex + 1, aOffsets..., aOffset, bOffsets..., bOffset,
    cOffsets..., cOffset, dOffsets..., dOffset
  );
}
};

// Specialize `step=1` case (i.e., `remaining_steps=0` and `len(Offsets)=1`).
// We don't need to pass in how many elements need to processed in this case.
template <typename Op,
          typename scalar1,
          typename scalar2,
          typename scalar3,
          typename scalar4,
          typename IndexType,
          int ADims,
          int BDims,
          int CDims,
          int DDims,
          typename Offset>
struct ApplyOp4<Op, scalar1, scalar2, scalar3, scalar4, IndexType,
                ADims, BDims, CDims, DDims, 0, Offset> {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar1, IndexType> &a,
                  detail::TensorInfo<scalar2, IndexType> &b,
                  detail::TensorInfo<scalar3, IndexType> &c,
                  detail::TensorInfo<scalar4, IndexType> &d,
                  const Op &op, int n, IndexType linearIndex,
                  Offset aOffset, Offset bOffset,
                  Offset cOffset, Offset dOffset) {
  op(a.data[aOffset], b.data[bOffset], c.data[cOffset], d.data[dOffset]);
}
};

template <typename Op,
          typename scalar1,
          typename scalar2,
          typename scalar3,
          typename scalar4,
          typename IndexType,
          int ADims,
          int BDims,
          int CDims,
          int DDims,
          typename... Offsets>
struct ApplyOp4<Op, scalar1, scalar2, scalar3, scalar4, IndexType,
                ADims, BDims, CDims, DDims, 0, Offsets...> {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar1, IndexType> &a,
                  detail::TensorInfo<scalar2, IndexType> &b,
                  detail::TensorInfo<scalar3, IndexType> &c,
                  detail::TensorInfo<scalar4, IndexType> &d,
                  const Op &op, int n, IndexType linearIndex,
                  Offsets... aOffsets, Offsets... bOffsets,
                  Offsets... cOffsets, Offsets... dOffsets) {
  op(n, a.data[aOffsets]..., b.data[bOffsets]..., c.data[cOffsets]..., d.data[dOffsets]...);
}
};

template <typename Op,
          typename scalar1,
          typename scalar2,
          typename scalar3,
          typename scalar4,
          typename IndexType,
          int ADims, int BDims, int CDims, int DDims,
          int step>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(AT_APPLY_THREADS_PER_BLOCK, AT_APPLY_BLOCKS_PER_SM)
#endif
__global__ void
kernelPointwiseApply4(detail::TensorInfo<scalar1, IndexType> a,
                      detail::TensorInfo<scalar2, IndexType> b,
                      detail::TensorInfo<scalar3, IndexType> c,
                      detail::TensorInfo<scalar4, IndexType> d,
                      IndexType totalElements,
                      const Op op) {
  for (IndexType linearIndex = (blockIdx.x * blockDim.x + threadIdx.x) * step;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x * step) {
    ApplyOp4<Op, scalar1, scalar2, scalar3, scalar4, IndexType,
             ADims, BDims, CDims, DDims, step>::apply(
      a, b, c, d, op, ::min(step, static_cast<int>(totalElements - linearIndex)), linearIndex);
  }
}

} // namespace

/**
   Computes ceil(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ T ATenCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <int step = 1>
inline bool getApplyGrid(uint64_t totalElements, dim3& grid, int64_t curDevice) {
  if (curDevice == -1) return false;
  uint64_t numel_per_thread = static_cast<uint64_t>(AT_APPLY_THREADS_PER_BLOCK) * static_cast<uint64_t>(step);
  uint64_t numBlocks = ATenCeilDiv(totalElements, numel_per_thread);
  uint64_t maxGridX = at::cuda::getDeviceProperties(curDevice)->maxGridSize[0];
  if (numBlocks > maxGridX)
      numBlocks = maxGridX;
  grid = dim3(numBlocks);
  return true;
}

inline dim3 getApplyBlock() {
  return dim3(AT_APPLY_THREADS_PER_BLOCK);
}


template <typename scalar, int step, typename Op>
inline bool CUDA_tensor_apply1(at::Tensor a,
                               const Op op,
                               TensorArgType aType = TensorArgType::ReadWrite) {
  checkBackend("CUDA_tensor_apply1", {a}, Backend::CUDA);
  auto dim = a.dim();

  /*
  Since this is a unary op, we can easily first check for expanded dimensions
  (with stride 0), and remove them, to avoid calling .contiguous() in such
  case when detail::maybeOverlappingIndices(a) returns true.
  */
  std::vector<int64_t> collapsed_shape;
  std::vector<int64_t> collapsed_strides;
  collapsed_shape.reserve(dim);
  collapsed_strides.reserve(dim);
  for (int64_t i = 0; i < dim; i++) {
    if (a.stride(i) != 0) {
      collapsed_shape.push_back(a.size(i));
      collapsed_strides.push_back(a.stride(i));
    }
  }
  if (collapsed_shape.size() != dim) {
    a = a.as_strided(collapsed_shape, collapsed_strides);
  }

  int64_t totalElements = a.numel();

  if (dim > MAX_TENSORINFO_DIMS) {
    return false;
  }

  if (totalElements == 0) {
    // Empty tensor; do nothing
    return true;
  }
  const dim3 block = getApplyBlock();

  dim3 grid;
  int64_t curDevice = current_device();
  if (curDevice == -1) return false;
  if (!getApplyGrid<step>(totalElements, grid, curDevice)) {
    return false;
  }

  /*
  Expands readable/writable tensors whose indices may be "overlapped."
  This ensures that each element of the tensor is operated on once and only
  once.
  */
  Tensor oldA;

  if (aType == TensorArgType::ReadWrite && detail::maybeOverlappingIndices(a)) {
    // Must perform in contiguous space
    oldA = a;
    a = a.contiguous();
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.

#define HANDLE_CASE(TYPE, A)                                           \
  kernelPointwiseApply1<Op,                                            \
                        scalar,                                        \
                        TYPE, A, step>                                 \
   <<<grid, block, 0, at::cuda::getCurrentCUDAStream(curDevice)>>>(    \
       aInfo, static_cast<TYPE>(totalElements), op);

#define HANDLE_A_CASE(TYPE, A) {            \
  switch (A) {                              \
    case 1:                                 \
      HANDLE_CASE(TYPE, 1);                 \
      break;                                \
    case 2:                                 \
      HANDLE_CASE(TYPE, 2);                 \
      break;                                \
    default:                                \
      HANDLE_CASE(TYPE, -1);                \
      break;                                \
  }                                         \
}

  if (detail::canUse32BitIndexMath(a)) {
    detail::TensorInfo<scalar, unsigned int> aInfo =
      detail::getTensorInfo<scalar, unsigned int>(a);

    rearrangeDims(&aInfo);
    aInfo.collapseDims();
#if CUDA_VERSION < 9000
    if (!aInfo.isContiguous())
        grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * AT_APPLY_BLOCKS_PER_SM , grid.x);
#endif

    HANDLE_A_CASE(unsigned int, aInfo.dims);
  } else {
    detail::TensorInfo<scalar, uint64_t> aInfo =
      detail::getTensorInfo<scalar, uint64_t>(a);

    rearrangeDims(&aInfo);
    aInfo.collapseDims();

    /*
    Only instantiates the all 1D special case and the fallback all nD case for
    large (64-bit indexed) tensors to reduce compilation time.
    */
    if (aInfo.dims == 1) {
      HANDLE_CASE(uint64_t, 1);
    } else {
#if CUDA_VERSION < 9000
      grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * AT_APPLY_BLOCKS_PER_SM , grid.x);
#endif
      HANDLE_CASE(uint64_t, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_A_CASE

  if (oldA.defined()) {
    // Ignore overlaps when copying back; if we use copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    at::_th_copy_ignoring_overlaps_(oldA, a);
  }

  return true;
}

/* Provides default step = 1 to CUDA_tensor_apply1. */
template <typename scalar, typename Op>
inline bool CUDA_tensor_apply1(at::Tensor a,
                               const Op op,
                               TensorArgType aType = TensorArgType::ReadWrite) {
  return CUDA_tensor_apply1<scalar, 1, Op>(a, op, aType);
}


template <typename scalar1, typename scalar2, int step, typename Op>
inline bool CUDA_tensor_apply2(at::Tensor a,
                               at::Tensor b,
                               const Op op,
                               TensorArgType aType = TensorArgType::ReadWrite,
                               TensorArgType bType = TensorArgType::ReadOnly) {
  checkBackend("CUDA_tensor_apply2", {a, b}, Backend::CUDA);
  int64_t totalElements = a.numel();

  if (totalElements != b.numel()) {
    return false;
  }

  if (a.dim() > MAX_TENSORINFO_DIMS ||
      b.dim() > MAX_TENSORINFO_DIMS) {
    return false;
  }

  if (a.numel() == 0) {
    // Empty tensor; do nothing
    return true;
  }
  const dim3 block = getApplyBlock();

  dim3 grid;
  int64_t curDevice = current_device();
  if (curDevice == -1) return false;
  if (!getApplyGrid<step>(totalElements, grid, curDevice)) {
    return false;
  }

  /*
  Expands readable/writable tensors whose indices may be "overlapped."
  This ensures that each element of the tensor is operated on once and only
  once.
  */
  Tensor oldA;
  Tensor oldB;

  if (aType == TensorArgType::ReadWrite && detail::maybeOverlappingIndices(a)) {
    // Must perform in contiguous space
    oldA = a;
    a = a.contiguous();
  }
  if (bType == TensorArgType::ReadWrite && detail::maybeOverlappingIndices(b)) {
    // Must perform in contiguous space
    oldB = b;
    b = b.contiguous();
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.

#define HANDLE_CASE(TYPE, A, B)                                        \
  kernelPointwiseApply2<Op,                                            \
                        scalar1,                                       \
                        scalar2,                                       \
                        TYPE, A, B, step>                              \
   <<<grid, block, 0, at::cuda::getCurrentCUDAStream(curDevice)>>>(    \
       aInfo, bInfo, static_cast<TYPE>(totalElements), op);

#define HANDLE_B_CASE(TYPE, A, B) {         \
  switch (B) {                              \
    case 1:                                 \
      HANDLE_CASE(TYPE, A, 1);              \
      break;                                \
    case 2:                                 \
      HANDLE_CASE(TYPE, A, 2);              \
      break;                                \
    default:                                \
      HANDLE_CASE(TYPE, A, -1);             \
      break;                                \
  }                                         \
}

#define HANDLE_A_CASE(TYPE, A, B) {         \
  switch (A) {                              \
    case 1:                                 \
      HANDLE_B_CASE(TYPE, 1, B);            \
      break;                                \
    case 2:                                 \
      HANDLE_B_CASE(TYPE, 2, B);            \
      break;                                \
    default:                                \
      HANDLE_B_CASE(TYPE, -1, B);           \
      break;                                \
  }                                         \
}

  if (detail::canUse32BitIndexMath(a) &&
      detail::canUse32BitIndexMath(b)) {
    detail::TensorInfo<scalar1, unsigned int> aInfo =
      detail::getTensorInfo<scalar1, unsigned int>(a);

    detail::TensorInfo<scalar2, unsigned int> bInfo =
      detail::getTensorInfo<scalar2, unsigned int>(b);
    rearrangeDims(&aInfo, &bInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();
#if CUDA_VERSION < 9000
    if (!(aInfo.isContiguous() && bInfo.isContiguous()))
        grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * AT_APPLY_BLOCKS_PER_SM , grid.x);
#endif

    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims);
  } else {
    detail::TensorInfo<scalar1, uint64_t> aInfo =
      detail::getTensorInfo<scalar1, uint64_t>(a);

    detail::TensorInfo<scalar2, uint64_t> bInfo =
      detail::getTensorInfo<scalar2, uint64_t>(b);
    rearrangeDims(&aInfo, &bInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();

    /*
    Only instantiates the all 1D special case and the fallback all nD case for
    large (64-bit indexed) tensors to reduce compilation time.
    */
    if (aInfo.dims == 1 && bInfo.dims == 1) {
      HANDLE_CASE(uint64_t, 1, 1);
    } else {
#if CUDA_VERSION < 9000
      grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * AT_APPLY_BLOCKS_PER_SM , grid.x);
#endif
      HANDLE_CASE(uint64_t, -1, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA.defined()) {
    // Ignore overlaps when copying back; if we use copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    at::_th_copy_ignoring_overlaps_(oldA, a);
  }

  if (oldB.defined()) {
    // Ignore overlaps when copying back; if we use copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    at::_th_copy_ignoring_overlaps_(oldB, b);
  }

  return true;
}

/* Provides default step = 1 to CUDA_tensor_apply2. */
template <typename scalar1, typename scalar2, typename Op>
inline bool CUDA_tensor_apply2(at::Tensor a,
                               at::Tensor b,
                               const Op op,
                               TensorArgType aType = TensorArgType::ReadWrite,
                               TensorArgType bType = TensorArgType::ReadOnly) {
  return CUDA_tensor_apply2<scalar1, scalar2, 1, Op>(a, b, op, aType, bType);
}


template <typename scalar1, typename scalar2, typename scalar3, int step, typename Op>
inline bool CUDA_tensor_apply3(at::Tensor a,
                               at::Tensor b,
                               at::Tensor c,
                               const Op op,
                               TensorArgType aType = TensorArgType::ReadWrite,
                               TensorArgType bType = TensorArgType::ReadOnly,
                               TensorArgType cType = TensorArgType::ReadOnly) {
  checkBackend("CUDA_tensor_apply3", {a, b, c}, Backend::CUDA);
  int64_t totalElements = a.numel();

  if (totalElements != b.numel() ||
      totalElements != c.numel()) {
    return false;
  }

  if (a.dim() > MAX_TENSORINFO_DIMS ||
      b.dim() > MAX_TENSORINFO_DIMS ||
      c.dim() > MAX_TENSORINFO_DIMS) {
    return false;
  }

  if (a.numel() == 0) {
    // Empty tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  int64_t curDevice = current_device();
  if (curDevice == -1) return false;
  if (!getApplyGrid<step>(totalElements, grid, curDevice)) {
    return false;
  }

  /*
  Expands readable/writable tensors whose indices may be "overlapped."
  This ensures that each element of the tensor is operated on once and only
  once.
  */
  Tensor oldA;
  Tensor oldB;
  Tensor oldC;

  if (aType == TensorArgType::ReadWrite && detail::maybeOverlappingIndices(a)) {
    // Must perform in contiguous space
    oldA = a;
    a = a.contiguous();
  }
  if (bType == TensorArgType::ReadWrite && detail::maybeOverlappingIndices(b)) {
    // Must perform in contiguous space
    oldB = b;
    b = b.contiguous();
  }
  if (cType == TensorArgType::ReadWrite && detail::maybeOverlappingIndices(c)) {
    // Must perform in contiguous space
    oldC = c;
    c = c.contiguous();
  }

#define HANDLE_CASE(TYPE, A, B, C)                                     \
  kernelPointwiseApply3<Op,                                            \
                        scalar1,                                       \
                        scalar2,                                       \
                        scalar3,                                       \
                        TYPE, A, B, C, step>                           \
    <<<grid, block, 0, at::cuda::getCurrentCUDAStream(curDevice)>>>(   \
      aInfo, bInfo, cInfo, static_cast<TYPE>(totalElements), op);

#define HANDLE_C_CASE(TYPE, A, B, C) {      \
  switch (C) {                              \
    case 1:                                 \
      HANDLE_CASE(TYPE, A, B, 1);           \
      break;                                \
    case 2:                                 \
      HANDLE_CASE(TYPE, A, B, 2);           \
      break;                                \
    default:                                \
      HANDLE_CASE(TYPE, A, B, -1);          \
      break;                                \
  }                                         \
}

#define HANDLE_B_CASE(TYPE, A, B, C) {      \
  switch (B) {                              \
    case 1:                                 \
      HANDLE_C_CASE(TYPE, A, 1, C);         \
      break;                                \
    case 2:                                 \
      HANDLE_C_CASE(TYPE, A, 2, C);         \
      break;                                \
    default:                                \
      HANDLE_C_CASE(TYPE, A, -1, C);        \
      break;                                \
  }                                         \
}

#define HANDLE_A_CASE(TYPE, A, B, C) {      \
  switch (A) {                              \
    case 1:                                 \
      HANDLE_B_CASE(TYPE, 1, B, C);         \
      break;                                \
    case 2:                                 \
      HANDLE_B_CASE(TYPE, 2, B, C);         \
      break;                                \
    default:                                \
      HANDLE_B_CASE(TYPE, -1, B, C);        \
      break;                                \
  }                                         \
}

  if (detail::canUse32BitIndexMath(a) &&
      detail::canUse32BitIndexMath(b) &&
      detail::canUse32BitIndexMath(c)) {
    detail::TensorInfo<scalar1, unsigned int> aInfo =
      detail::getTensorInfo<scalar1, unsigned int>(a);

    detail::TensorInfo<scalar2, unsigned int> bInfo =
      detail::getTensorInfo<scalar2, unsigned int>(b);

    detail::TensorInfo<scalar3, unsigned int> cInfo =
      detail::getTensorInfo<scalar3, unsigned int>(c);

    rearrangeDims(&aInfo, &bInfo, &cInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();
    cInfo.collapseDims();

#if CUDA_VERSION < 9000
    if (!(aInfo.isContiguous() && bInfo.isContiguous() && cInfo.isContiguous()))
      grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * AT_APPLY_BLOCKS_PER_SM , grid.x);
#endif
    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims, cInfo.dims);
  } else {
    detail::TensorInfo<scalar1, uint64_t> aInfo =
      detail::getTensorInfo<scalar1, uint64_t>(a);

    detail::TensorInfo<scalar2, uint64_t> bInfo =
      detail::getTensorInfo<scalar2, uint64_t>(b);

    detail::TensorInfo<scalar3, uint64_t> cInfo =
      detail::getTensorInfo<scalar3, uint64_t>(c);

    rearrangeDims(&aInfo, &bInfo, &cInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();
    cInfo.collapseDims();

    /*
    Only instantiates the all 1D special case and the fallback all nD case for
    large (64-bit indexed) tensors to reduce compilation time.
    */
    if (aInfo.dims == 1 && bInfo.dims == 1 && cInfo.dims == 1) {
      HANDLE_CASE(uint64_t, 1, 1, 1);
    } else {
#if CUDA_VERSION < 9000
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * AT_APPLY_BLOCKS_PER_SM , grid.x);
#endif

      HANDLE_CASE(uint64_t, -1, -1, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_C_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA.defined()) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    at::_th_copy_ignoring_overlaps_(oldA, a);
    a = oldA;
  }

  if (oldB.defined()) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    at::_th_copy_ignoring_overlaps_(oldB, b);
    b = oldB;
  }

  if (oldC.defined()) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldC contiguous.
    at::_th_copy_ignoring_overlaps_(oldC, c);
    c = oldC;
  }

  return true;
}

/* Provides default step = 1 to CUDA_tensor_apply3. */
template <typename scalar1, typename scalar2, typename scalar3, typename Op>
inline bool CUDA_tensor_apply3(at::Tensor a,
                               at::Tensor b,
                               at::Tensor c,
                               const Op op,
                               TensorArgType aType = TensorArgType::ReadWrite,
                               TensorArgType bType = TensorArgType::ReadOnly,
                               TensorArgType cType = TensorArgType::ReadOnly) {
  return CUDA_tensor_apply3<scalar1, scalar2, scalar3, 1, Op>(
    a, b, c, op, aType, bType, cType);
}


template <typename scalar1, typename scalar2, typename scalar3, typename scalar4,
          int step, typename Op>
inline bool CUDA_tensor_apply4(at::Tensor a,
                               at::Tensor b,
                               at::Tensor c,
                               at::Tensor d,
                               const Op op,
                               TensorArgType aType = TensorArgType::ReadWrite,
                               TensorArgType bType = TensorArgType::ReadOnly,
                               TensorArgType cType = TensorArgType::ReadOnly,
                               TensorArgType dType = TensorArgType::ReadOnly) {
  checkBackend("CUDA_tensor_apply4", {a, b, c, d}, Backend::CUDA);
  int64_t totalElements = a.numel();

  if (totalElements != b.numel() ||
      totalElements != c.numel() ||
      totalElements != d.numel()) {
    return false;
  }

  if (a.dim() > MAX_TENSORINFO_DIMS ||
      b.dim() > MAX_TENSORINFO_DIMS ||
      c.dim() > MAX_TENSORINFO_DIMS ||
      d.dim() > MAX_TENSORINFO_DIMS) {
    return false;
  }

  if (a.numel() == 0) {
    // Empty tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  int64_t curDevice = current_device();
  if (curDevice == -1) return false;
  if (!getApplyGrid<step>(totalElements, grid, curDevice)) {
    return false;
  }

  /*
  Expands readable/writable tensors whose indices may be "overlapped."
  This ensures that each element of the tensor is operated on once and only
  once.
  */
  Tensor oldA;
  Tensor oldB;
  Tensor oldC;
  Tensor oldD;

  if (aType == TensorArgType::ReadWrite && detail::maybeOverlappingIndices(a)) {
    // Must perform in contiguous space
    oldA = a;
    a = a.contiguous();
  }
  if (bType == TensorArgType::ReadWrite && detail::maybeOverlappingIndices(b)) {
    // Must perform in contiguous space
    oldB = b;
    b = b.contiguous();
  }
  if (cType == TensorArgType::ReadWrite && detail::maybeOverlappingIndices(c)) {
    // Must perform in contiguous space
    oldC = c;
    c = c.contiguous();
  }
  if (dType == TensorArgType::ReadWrite && detail::maybeOverlappingIndices(c)) {
    // Must perform in contiguous space
    oldD = d;
    d = d.contiguous();
  }

#define HANDLE_CASE(TYPE, A, B, C, D)                                  \
  kernelPointwiseApply4<Op,                                            \
                        scalar1,                                       \
                        scalar2,                                       \
                        scalar3,                                       \
                        scalar4,                                       \
                        TYPE, A, B, C, D, step>                        \
    <<<grid, block, 0, at::cuda::getCurrentCUDAStream(curDevice)>>>(   \
    aInfo, bInfo, cInfo, dInfo, static_cast<TYPE>(totalElements), op);

#define HANDLE_D_CASE(TYPE, A, B, C, D) {       \
  switch (D) {                                  \
    case 1:                                     \
      HANDLE_CASE(TYPE, A, B, C, 1);            \
      break;                                    \
    case 2:                                     \
      HANDLE_CASE(TYPE, A, B, C, 2);            \
      break;                                    \
    default:                                    \
      HANDLE_CASE(TYPE, A, B, C, -1);           \
      break;                                    \
  }                                             \
}

#define HANDLE_C_CASE(TYPE, A, B, C, D) {       \
  switch (C) {                                  \
    case 1:                                     \
      HANDLE_D_CASE(TYPE, A, B, 1, D);          \
      break;                                    \
    case 2:                                     \
      HANDLE_D_CASE(TYPE, A, B, 2, D);          \
      break;                                    \
    default:                                    \
      HANDLE_D_CASE(TYPE, A, B, -1, D);         \
      break;                                    \
  }                                             \
}

#define HANDLE_B_CASE(TYPE, A, B, C, D) {       \
  switch (B) {                                  \
    case 1:                                     \
      HANDLE_C_CASE(TYPE, A, 1, C, D);          \
      break;                                    \
    case 2:                                     \
      HANDLE_C_CASE(TYPE, A, 2, C, D);          \
      break;                                    \
    default:                                    \
      HANDLE_C_CASE(TYPE, A, -1, C, D);         \
      break;                                    \
  }                                             \
}

#define HANDLE_A_CASE(TYPE, A, B, C, D) {       \
  switch (A) {                                  \
    case 1:                                     \
      HANDLE_B_CASE(TYPE, 1, B, C, D);          \
      break;                                    \
    case 2:                                     \
      HANDLE_B_CASE(TYPE, 2, B, C, D);          \
      break;                                    \
    default:                                    \
      HANDLE_B_CASE(TYPE, -1, B, C, D);         \
      break;                                    \
  }                                             \
}

  if (detail::canUse32BitIndexMath(a) &&
      detail::canUse32BitIndexMath(b) &&
      detail::canUse32BitIndexMath(c) &&
      detail::canUse32BitIndexMath(d)) {
    detail::TensorInfo<scalar1, unsigned int> aInfo =
      detail::getTensorInfo<scalar1, unsigned int>(a);

    detail::TensorInfo<scalar2, unsigned int> bInfo =
      detail::getTensorInfo<scalar2, unsigned int>(b);

    detail::TensorInfo<scalar3, unsigned int> cInfo =
      detail::getTensorInfo<scalar3, unsigned int>(c);

    detail::TensorInfo<scalar4, unsigned int> dInfo =
      detail::getTensorInfo<scalar4, unsigned int>(d);

    rearrangeDims(&aInfo, &bInfo, &cInfo, &dInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();
    cInfo.collapseDims();
    dInfo.collapseDims();

#if CUDA_VERSION < 9000
    if (!(aInfo.isContiguous() && bInfo.isContiguous() && cInfo.isContiguous() && dInfo.isContiguous()))
      grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * AT_APPLY_BLOCKS_PER_SM , grid.x);
#endif
    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims, cInfo.dims, dInfo.dims);
  } else {
    detail::TensorInfo<scalar1, uint64_t> aInfo =
      detail::getTensorInfo<scalar1, uint64_t>(a);

    detail::TensorInfo<scalar2, uint64_t> bInfo =
      detail::getTensorInfo<scalar2, uint64_t>(b);

    detail::TensorInfo<scalar3, uint64_t> cInfo =
      detail::getTensorInfo<scalar3, uint64_t>(c);

    detail::TensorInfo<scalar4, uint64_t> dInfo =
      detail::getTensorInfo<scalar4, uint64_t>(d);

    rearrangeDims(&aInfo, &bInfo, &cInfo, &dInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();
    cInfo.collapseDims();
    dInfo.collapseDims();

    /*
    Only instantiates the all 1D special case and the fallback all nD case for
    large (64-bit indexed) tensors to reduce compilation time.
    */
    if (aInfo.dims == 1 && bInfo.dims == 1 && cInfo.dims == 1 && dInfo.dims == 1) {
      HANDLE_CASE(uint64_t, 1, 1, 1, 1);
    } else {
#if CUDA_VERSION < 9000
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * AT_APPLY_BLOCKS_PER_SM , grid.x);
#endif
      HANDLE_CASE(uint64_t, -1, -1, -1, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_D_CASE
#undef HANDLE_C_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA.defined()) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    at::_th_copy_ignoring_overlaps_(oldA, a);
  }

  if (oldB.defined()) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    at::_th_copy_ignoring_overlaps_(oldB, b);
  }

  if (oldC.defined()) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldC contiguous.
    at::_th_copy_ignoring_overlaps_(oldC, c);
  }

  if (oldD.defined()) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldC contiguous.
    at::_th_copy_ignoring_overlaps_(oldD, c);
  }

  return true;
}

/* Provides default step = 1 to CUDA_tensor_apply4. */
template <typename scalar1, typename scalar2, typename scalar3, typename scalar4,
          typename Op>
inline bool CUDA_tensor_apply4(at::Tensor a,
                               at::Tensor b,
                               at::Tensor c,
                               at::Tensor d,
                               const Op op,
                               TensorArgType aType = TensorArgType::ReadWrite,
                               TensorArgType bType = TensorArgType::ReadOnly,
                               TensorArgType cType = TensorArgType::ReadOnly,
                               TensorArgType dType = TensorArgType::ReadOnly) {
  return CUDA_tensor_apply4<scalar1, scalar2, scalar3, scalar4, 1, Op>(
    a, b, c, d, op, aType, bType, cType);
}

} // cuda
} // at
