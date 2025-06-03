#include "ATen/CPUGenerator.h"  // CPU生成器头文件

// 类型转换宏：将基类Generator转换为CPUGenerator常量引用
#define const_generator_cast(generator) \
  dynamic_cast<const CPUGenerator&>(generator)

namespace at {  // ATen核心命名空间

// 构造函数：初始化上下文并创建THGenerator对象
CPUGenerator::CPUGenerator(Context * context_)
  : context(context_), generator(THGenerator_new())  // 调用TH库创建生成器
{}

// 析构函数：安全释放生成器资源
CPUGenerator::~CPUGenerator() {
  if (generator)
    THGenerator_free(generator);  // 调用TH库释放内存
}

// 复制构造函数：从另一个生成器复制状态
CPUGenerator& CPUGenerator::copy(const Generator& from) {
  THGenerator_copy(generator, const_generator_cast(from).generator);  // TH库复制操作
  return *this;
}

// 释放生成器资源（显式调用）
CPUGenerator& CPUGenerator::free() {
  THGenerator_free(generator);  // 调用TH库释放
  return *this;
}

// 获取当前随机种子
uint64_t CPUGenerator::seed() {
  return THRandom_seed(generator);  // 调用TH库获取种子
}

// 获取初始随机种子
uint64_t CPUGenerator::initialSeed() {
  return THRandom_initialSeed(generator);  // 调用TH库获取初始种子
}

// 手动设置随机种子（影响后续所有随机数）
CPUGenerator& CPUGenerator::manualSeed(uint64_t seed) {
  THRandom_manualSeed(generator, seed);  // 调用TH库设置种子
  return *this;
}

// 设置所有CPU生成器的种子（实际单例实现）
CPUGenerator& CPUGenerator::manualSeedAll(uint64_t seed) {
  return manualSeed(seed);  // CPU生成器全局单例，直接转发调用
}

// 获取底层THGenerator指针（危险操作）
void * CPUGenerator::unsafeGetTH() {
  return generator;  // 暴露原始指针用于特殊场景
}

} // namespace at

