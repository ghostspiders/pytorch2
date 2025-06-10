#pragma once

#include <c10/util/ArrayRef.h>
#include <ATen/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>

namespace at { namespace cuda {

// TODO: 在c10中通用化实现这个功能。需要从GuardImpl获取GPU数量
// 多流保护器类，用于管理多个CUDA流的状态
class CUDAMultiStreamGuard final {
public:
  /// 构造函数：为列表中每个流调用`set_stream`
  /// 当需要为不同设备设置不同流时非常有用
  explicit CUDAMultiStreamGuard(ArrayRef<CUDAStream> streams) : CUDAMultiStreamGuard() {
    for (const auto& s : streams) {
      setCurrentCUDAStream(s);  // 为每个流设置当前CUDA流
    }
  }

  /// 默认构造函数：保存所有设备的当前流状态
  CUDAMultiStreamGuard() {
    const size_t device_count = getNumGPUs();  // 获取GPU数量
    original_streams_.reserve(device_count);  // 预分配空间
    for (size_t device = 0; device < device_count; ++device) {
      original_streams_.push_back(getCurrentCUDAStream(device));  // 保存每个设备的当前流
    }
  }

  // 禁止拷贝构造和拷贝赋值
  CUDAMultiStreamGuard(const CUDAGuard&) = delete;
  CUDAMultiStreamGuard& operator=(const CUDAGuard&) = delete;

  // 注意：[RAII保护类的移动构造很复杂]
  CUDAMultiStreamGuard(CUDAGuard&& other) = delete;

  // 注意：[RAII保护类的移动赋值很复杂]
  CUDAMultiStreamGuard& operator=(CUDAGuard&& other) = delete;

  /// 获取原始流数组的引用
  ArrayRef<CUDAStream> original_streams() const {
    return original_streams_;
  }

  /// 析构函数：将所有设备的CUDA流重置为构造时的状态
  ~CUDAMultiStreamGuard() {
    for (const auto& s : original_streams_) {
      setCurrentCUDAStream(s);  // 恢复每个设备的原始流
    }
  }

private:
  /// 保存所有设备原始流的向量
  std::vector<CUDAStream> original_streams_;
};

}} // namespace at::cuda