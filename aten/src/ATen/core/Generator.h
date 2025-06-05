#pragma once  // 防止头文件重复包含

#include <ATen/core/ATenGeneral.h>  // 包含ATen核心通用定义
#include <stdint.h>                 // 标准整数类型定义

namespace at {

// 生成器抽象基类，用于生成随机数
// CAFFE2_API宏确保在不同平台/编译器下的符号可见性
struct CAFFE2_API Generator {
  Generator() {};  // 默认构造函数
  Generator(const Generator& other) = delete;  // 禁用拷贝构造函数
  Generator(Generator&& other) = delete;       // 禁用移动构造函数
  virtual ~Generator() {};                     // 虚析构函数

  // 纯虚函数，需要子类实现:

  // 复制生成器状态
  virtual Generator& copy(const Generator& other) = 0;
  
  // 释放生成器资源
  virtual Generator& free() = 0;

  // 获取当前种子
  virtual uint64_t seed() = 0;
  
  // 获取初始种子
  virtual uint64_t initialSeed() = 0;
  
  // 手动设置种子(单个设备)
  virtual Generator& manualSeed(uint64_t seed) = 0;
  
  // 手动设置种子(所有设备)
  virtual Generator& manualSeedAll(uint64_t seed) = 0;
  
  // 获取底层TH(TH是Torch的C库)指针(不安全操作)
  virtual void * unsafeGetTH() = 0;
};

} // namespace at
