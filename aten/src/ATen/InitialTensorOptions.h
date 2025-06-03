#pragma once

#include <ATen/core/TensorOptions.h>

namespace at {

/**
 * 返回初始默认的TensorOptions配置（在默认值被修改前）
 * 
 * 设计用途：
 * - 主要用于库内部代码，当明确知道需要的设备/数据类型等配置时
 * - 提供稳定的默认初始配置，不受用户全局默认设置影响
 * 
 * 注意：
 * - 这不是稳定API，可能在PyTorch版本更新时改变
 * 
 * 默认配置说明：
 * - 设备(device): CPU
 * - 数据类型(dtype): 32位浮点(kFloat)
 * - 内存布局(layout): 连续内存(kStrided)
 * - 自动求导(requires_grad): 关闭
 * - 变量标识(is_variable): 关闭（用于向后兼容）
 */
inline TensorOptions initialTensorOptions() {
  return TensorOptions(kCPU)  // 默认CPU设备
         .dtype(kFloat)      // 默认float32类型
         .layout(kStrided)   // 默认连续内存布局
         .requires_grad(false)  // 默认不需要梯度计算
         .is_variable(false);  // 默认非变量（历史遗留参数）
}

} // namespace at
