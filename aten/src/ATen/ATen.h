#pragma once

// ATen 核心头文件集合
// 该文件包含了使用 ATen 张量库所需的主要组件

// 内存分配器接口（定义分配策略）
#include "ATen/Allocator.h"

// CPU 通用功能（特定于 CPU 设备的实现细节）
#include "ATen/CPUGeneral.h"

// 设备上下文管理（包含后端注册和设备状态）
#include "ATen/Context.h"

// 设备类型定义（CPU/CUDA 设备表示）
#include "ATen/Device.h"

// 设备保护作用域（自动设备切换和恢复）
#include "ATen/DeviceGuard.h"

// 维度向量（动态维度操作的安全容器）
#include "ATen/DimVector.h"

// 分发机制（实现基于后端的函数分发）
#include "ATen/Dispatch.h"

// 格式化工具（张量打印和字符串表示）
#include "ATen/Formatting.h"

// 预定义张量函数（标准操作的函数接口）
#include "ATen/Functions.h"

// 标量操作（标量与张量间的运算）
#include "ATen/ScalarOps.h"

// 张量核心定义（张量类型和基础API）
#include "ATen/Tensor.h"

// 张量几何操作（形状计算和广播规则）
#include "ATen/TensorGeometry.h"

// 张量运算符重载（+, -, * 等运算符实现）
#include "ATen/TensorOperators.h"

// 类型系统（已弃用，用于旧版类型分发）
#include "ATen/Type.h"

// ATen 通用配置（全局编译设置和宏定义）
#include "ATen/core/ATenGeneral.h"

// 随机数生成器（RNG 状态管理）
#include "ATen/core/Generator.h"

// 内存布局定义（Strided, Sparse 等布局类型）
#include <c10/core/Layout.h>

// 标量类型系统（统一标量表示）
#include "ATen/core/Scalar.h"

// 存储系统（张量的底层数据存储）
#include <c10/core/Storage.h"

// 张量方法（历史遗留，部分便捷方法）
#include "ATen/core/TensorMethods.h"

// 张量配置选项（数据类型/设备/布局的配置容器）
#include "ATen/core/TensorOptions.h"

// 异常处理（错误检查和异常抛出）
#include <c10/util/Exception.h>

/*****************************************************************
 * 使用说明：
 * 1. 这是传统的ATen头文件包含方式
 * 2. 现代用法推荐优先使用主头文件：
 *    #include <ATen/ATen.h> // 包含大部分核心功能
 *    #include <c10/util/Exception.h> // 异常处理
 * 
 * 3. 特殊功能按需添加：
 *    - 动态维度计算：<ATen/DimVector.h>
 *    - 设备管理：<ATen/DeviceGuard.h>
 *    - 自定义分发：<ATen/Dispatch.h>
 *    - 形状计算：<ATen/TensorGeometry.h>
 * 
 * 4. 注意冗余包含：
 *    - 多个头文件可能被<ATen/ATen.h>覆盖
 *    - 某些头文件(如Type.h)在新版本中已弃用
 * 
 * 5. 编译优化建议：
 *    - 在源文件中包含而非头文件
 *    - 使用前置声明减少依赖
 *    - 定期清理未使用的头文件
 *****************************************************************/