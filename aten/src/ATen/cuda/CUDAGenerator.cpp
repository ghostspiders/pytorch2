// 引入ATen配置头文件
#include "ATen/Config.h"

// 引入相关头文件
#include "ATen/CUDAGenerator.h"  // CUDA随机数生成器定义
#include "ATen/Context.h"        // ATen上下文
#include "THCTensorRandom.h"     // THC随机数生成函数
#include <stdexcept>             // 标准异常处理

// 全局唯一CUDAGenerator实例
// seed(), manualSeed(), initialSeed(), unsafeGetTH()等操作
// 都指向当前设备上的THCGenerator

// 声明THC函数：获取当前设备的随机数生成器
THCGenerator* THCRandom_getGenerator(THCState* state);

namespace at {

// 构造函数：初始化上下文指针
CUDAGenerator::CUDAGenerator(Context * context_)
  : context(context_)  // 保存ATen上下文
{
  // 注：实际生成器状态由THC全局管理，此处不进行初始化
}

// 析构函数：无操作（生成器状态由程序全局管理）
CUDAGenerator::~CUDAGenerator() {
  // 生成器状态是程序全局共享的，不需要特殊清理
}

// 复制函数：未实现（CUDA生成器不支持复制）
CUDAGenerator& CUDAGenerator::copy(const Generator& from) {
  throw std::runtime_error("CUDAGenerator::copy() not implemented");
}

// 释放资源：关闭THC随机数系统
CUDAGenerator& CUDAGenerator::free() {
  THCRandom_shutdown(context->getTHCState());
  return *this;
}

// 获取当前随机种子（会自动生成新种子如果未设置）
uint64_t CUDAGenerator::seed() {
  return THCRandom_initialSeed(context->getTHCState());
}

// 获取初始随机种子
uint64_t CUDAGenerator::initialSeed() {
  return THCRandom_initialSeed(context->getTHCState());
}

// 手动设置当前设备随机种子
CUDAGenerator& CUDAGenerator::manualSeed(uint64_t seed) {
  THCRandom_manualSeed(context->getTHCState(), seed);
  return *this;  // 支持链式调用
}

// 手动设置所有设备的随机种子
CUDAGenerator& CUDAGenerator::manualSeedAll(uint64_t seed) {
  THCRandom_manualSeedAll(context->getTHCState(), seed);
  return *this;  // 支持链式调用
}

// 获取底层THCGenerator指针（不安全操作）
void * CUDAGenerator::unsafeGetTH() {
  return (void*)THCRandom_getGenerator(context->getTHCState());
}

} // namespace at