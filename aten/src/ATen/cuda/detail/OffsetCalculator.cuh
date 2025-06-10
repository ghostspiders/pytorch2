#pragma once  // 防止头文件重复包含

#include <array>
#include <cstdint>
#include <c10/macros/Macros.h>       // 基础宏定义
#include <ATen/cuda/Array.h>         // CUDA数组工具
#include <THC/THCIntegerDivider.cuh> // 整数除法工具

/// OffsetCalculator 计算线性索引在NARGS个操作数中的字节偏移量
/// 这些操作数共享相同形状但可能有不同的步长(stride)

template <int NARGS>  // 模板参数：操作数数量
struct OffsetCalculator {
  static constexpr int MAX_DIMS = 25;  // 支持的最大维度数

  // 每个参数的偏移量(字节)，基于固定大小数组的包装类型
  using offset_type = at::cuda::Array<uint32_t, NARGS>;

  // 构造函数：初始化维度、尺寸和步长信息
  OffsetCalculator(int dims, const int64_t* sizes, const int64_t* const* strides) : dims(dims) {
    AT_CHECK(dims <= MAX_DIMS, "tensor has too many (>25) dims");  // 维度数检查
    
    // 初始化每个维度的尺寸和步长
    for (int i = 0; i < MAX_DIMS; ++i) {
      if (i < dims) {
        sizes_[i] = IntDivider<uint32_t>(sizes[i]);  // 实际维度的整数除法器
      } else {
        sizes_[i] = IntDivider<uint32_t>(1);         // 超出维度的填充值
      }
      
      // 初始化每个参数在当前维度的步长
      for (int arg = 0; arg < NARGS; arg++) {
        strides_[i][arg] = i < dims ? strides[arg][i] : 0;  // 实际维度步长或0
      }
    }
  }

  // 设备/主机函数：计算给定线性索引对应的各参数偏移量
  C10_HOST_DEVICE offset_type get(uint32_t linear_idx) const {
    offset_type offsets;  // 存储结果的偏移量数组
    
    // 初始化所有参数的偏移量为0
    #pragma unroll  // 循环展开优化提示
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = 0;
    }

    // 通过连续整数除法计算各维度索引
    #pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {  // 达到实际维度数时终止
        break;
      }
      
      // 计算当前维度的商和余数
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;  // 更新线性索引为商

      // 累加各参数在当前维度的偏移量
      #pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] += divmod.mod * strides_[dim][arg];  // 余数×步长
      }
    }
    return offsets;
  }

  // 成员变量
  int dims;                           // 实际维度数
  IntDivider<uint32_t> sizes_[MAX_DIMS]; // 各维度的整数除法器
  uint32_t strides_[MAX_DIMS][NARGS]; // 各维度各参数的步长
};