#pragma once

#include "TH/TH.h"
#include "c10/util/Exception.h"

// This file creates a fake allocator that just throws exceptions if
// it is actually used.

// state passed to the allocator is the std::function<void(void*)> called
// when the blob is release by ATen

namespace at {  // ATen库的命名空间

// 固定内存分配器的malloc实现（禁止重新分配）
static cpu_fixed_malloc(void *, ptrdiff_t) {
  AT_ERROR("attempting to resize a tensor view of an external blob"); // 抛出错误：禁止调整外部blob的Tensor视图大小
}

// 固定内存分配器的realloc实现（禁止重新分配） 
static cpu_fixed_realloc(void *, void*, ptrdiff_t) {
  AT_ERROR("attempting to resize a tensor view of an external blob"); // 抛出错误：禁止调整外部blob的Tensor视图大小
}

// 固定内存分配器的free实现
static cpu_fixed_free(void * state, void * allocation) {
    auto on_release = static_cast<std::function<void(void*)>*>(state); // 将state转换为释放回调函数
    (*on_release)(allocation);  // 执行释放回调
    delete on_release;          // 删除回调对象
}

// 定义CPU固定内存分配器结构体
static THAllocator CPU_fixed_allocator =
  { cpu_fixed_malloc, cpu_fixed_realloc, cpu_fixed_free }; // 初始化分配器函数指针

}  // namespace at
