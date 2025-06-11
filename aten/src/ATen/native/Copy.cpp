#include "ATen/native/Copy.h"

#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/native/cpu/CopyKernel.h"

namespace {

// 模板函数：将 src 张量的值复制到 self 张量中
template <typename self_T, typename src_T>
void _copy__cpu(at::Tensor& self, const at::Tensor& src) {
  // 使用 CPU_tensor_apply2 对两个张量进行逐元素操作
  at::CPU_tensor_apply2<self_T, src_T>(
      self, src, [](self_T& self_val, const src_T& src_val) {
        // 将 src 的值转换为 self 的类型后赋值
        self_val = static_cast<self_T>(
            static_cast<at::native::inter_copy_type_t<self_T>>(src_val));
      });
}

// 模板函数：处理 self 和 src 类型不同的情况
template <typename self_T>
void _copy__cpu(at::Tensor& self, const at::Tensor& src) {
  // 检查两个张量的元素数量是否一致
  AT_CHECK(self.numel() == src.numel(), "sizes do not match");
  // 根据 src 的类型分发到不同的实现
  AT_DISPATCH_ALL_TYPES_AND_HALF(src.type(), "_copy__cpu", [&]() {
    _copy__cpu<self_T, scalar_t>(self, src);
  });
}

// 判断是否满足特殊转置复制的条件
bool copy_transpose_valid(const at::Tensor& self, const at::Tensor& src) {
  const int MIN_SZ = 60 * 60; // 最小尺寸阈值
  return self.is_contiguous() && src.numel() != 0 && src.dim() == 2 &&
      src.stride(0) == 1 && src.stride(1) == src.size(0) &&
      self.numel() >= MIN_SZ;
}

} // namespace

namespace at {
namespace native {

// 异步复制函数：将 src 的内容复制到 self 中
Tensor& _s_copy__cpu(Tensor& self, const Tensor& src, bool non_blocking) {
  // 如果 src 是 CUDA 张量，调用 _s_copy_from 进行复制
  if (src.is_cuda()) {
    _s_copy_from(src, self, non_blocking);
    return self;
  }
  // 根据 self 的类型分发到不同的实现
  AT_DISPATCH_ALL_TYPES_AND_HALF(
      self.type(), "_copy__cpu", [&]() { ::_copy__cpu<scalar_t>(self, src); });
  return self;
}

// 特殊情况：处理张量连续且 src 是转置矩阵的复制
void _copy_same_type_transpose_(Tensor& self, const Tensor& src) {
  int64_t BLOCK_SZ;
  // 根据数据类型设置块大小
  if (self.scalar_type() == kByte) {
    BLOCK_SZ = 120;
  } else {
    BLOCK_SZ = 60;
  }
  // 创建一个临时缓冲区
  Tensor buf = empty({BLOCK_SZ, BLOCK_SZ}, self.options());

  // 根据 self 的类型分发到不同的实现
  AT_DISPATCH_ALL_TYPES_AND_HALF(
      self.type(), "_copy_same_type_transpose_", [&]() {
        scalar_t* sp = src.data<scalar_t>(); // src 的数据指针
        scalar_t* rp = self.data<scalar_t>(); // self 的数据指针
        scalar_t* bp = buf.data<scalar_t>(); // 缓冲区的数据指针

        int64_t NR = src.size(0); // src 的行数
        int64_t NC = src.size(1); // src 的列数
        // 按块进行复制和转置
        for (int64_t R = 0; R < NR; R += BLOCK_SZ) {
          for (int64_t C = 0; C < NC; C += BLOCK_SZ) {
            scalar_t* spo = sp + R + C * NR; // 当前块的 src 起始位置
            scalar_t* rpo = rp + C + R * NC; // 当前块的 self 起始位置

            int nr = std::min(NR - R, BLOCK_SZ); // 当前块的行数
            int nc = std::min(NC - C, BLOCK_SZ); // 当前块的列数

            // 1. 将 src 的列复制到缓冲区
            for (int c = 0; c < nc; c++) {
              memcpy(bp + c * BLOCK_SZ, spo + c * NR, nr * sizeof(scalar_t));
            }

            // 2. 在缓冲区内进行原地转置
            int rc_max = std::max(nr, nc);
            int rc_min = std::min(nr, nc);
            for (int r = 0; r < rc_max; r++) {
              int end = std::min(r, rc_min);
              for (int c = 0; c < end; c++) {
                scalar_t tmp = bp[r + BLOCK_SZ * c];
                bp[r + BLOCK_SZ * c] = bp[r * BLOCK_SZ + c];
                bp[r * BLOCK_SZ + c] = tmp;
              }
            }

            // 3. 将缓冲区的行复制到 self
            for (int r = 0; r < nr; r++) {
              memcpy(rpo + r * NC, bp + r * BLOCK_SZ, nc * sizeof(scalar_t));
            }
          }
        }
      });
}

// 同类型张量的复制函数
void _copy_same_type__cpu(Tensor& self, const Tensor& src) {
  // 如果 self 和 src 是同一个张量，直接返回
  if (self.is_same(src)) {
    return;
  }

  bool serial_path = false; // 是否走串行路径
  // 如果 self 和 src 的元素数量一致
  if (self.numel() == src.numel()) {
    // 如果 self 和 src 都是连续的，直接调用 copy_kernel
    if (self.is_contiguous() && src.is_contiguous()) {
      copy_kernel(kCPU, self, src);
    } 
    // 如果满足特殊转置复制的条件，调用 _copy_same_type_transpose_
    else if (copy_transpose_valid(self, src)) {
      _copy_same_type_transpose_(self, src);
    } 
    // 其他情况
    else {
#ifdef _OPENMP
      // 如果不在并行区域
      if (!in_parallel_region()) {
        // 使用 OpenMP 并行化复制
        AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "_copy_same_type_", [&]() {
          at::CPU_tensor_parallel_apply2<scalar_t, scalar_t>(
              self, src, [](scalar_t& self_val, const scalar_t& src_val) {
                self_val = src_val;
              });
        });
      } else {
        // 走串行路径
        serial_path = true;
      }
#else
      // 如果没有启用 OpenMP，走串行路径
      serial_path = true;
#endif
    }
  } else {
    // 如果 self 和 src 的元素数量不一致，走串行路径
    serial_path = true;
  }

  // 串行路径：逐元素复制
  if (serial_path) {
    AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "_copy_same_type_", [&]() {
      at::CPU_tensor_apply2<scalar_t, scalar_t>(
          self, src, [](scalar_t& self_val, const scalar_t& src_val) {
            self_val = src_val;
          });
    });
  }
}

// 定义 copy_kernel 的分发函数
DEFINE_DISPATCH(copy_kernel);

} // namespace native
} // namespace at