#include "DispatchStub.h"  // 分发桩头文件

#include <c10/util/Exception.h>  // 异常处理

#include <cpuinfo.h>  // CPU信息检测库
#include <cstdlib>
#include <cstring>

namespace at { namespace native {

// 计算当前CPU的能力等级
static CPUCapability compute_cpu_capability() {
  // 首先检查环境变量设置
  auto envar = std::getenv("ATEN_CPU_CAPABILITY");
  if (envar) {
    // 如果环境变量明确指定AVX2
    if (strcmp(envar, "avx2") == 0) {
      return CPUCapability::AVX2;
    }
    // 如果环境变量明确指定AVX
    if (strcmp(envar, "avx") == 0) {
      return CPUCapability::AVX;
    }
    // 如果环境变量明确指定默认
    if (strcmp(envar, "default") == 0) {
      return CPUCapability::DEFAULT;
    }
    // 无效值警告
    AT_WARN("ignoring invalid value for ATEN_CPU_CAPABILITY: ", envar);
  }

// 在非PowerPC架构上进行CPU能力检测
#ifndef __powerpc__
  // 初始化cpuinfo库
  if (cpuinfo_initialize()) {
    // 检测是否支持AVX2和FMA3指令集
    if (cpuinfo_has_x86_avx2() && cpuinfo_has_x86_fma3()) {
      return CPUCapability::AVX2;
    }
    // 检测是否支持AVX指令集
    if (cpuinfo_has_x86_avx()) {
      return CPUCapability::AVX;
    }
  }
#endif
  // 默认返回基础能力
  return CPUCapability::DEFAULT;
}

// 获取CPU能力等级（带缓存）
CPUCapability get_cpu_capability() {
  // 使用静态变量缓存检测结果
  static CPUCapability capability = compute_cpu_capability();
  return capability;
}

}}  // namespace at::native