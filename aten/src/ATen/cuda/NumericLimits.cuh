#pragma once

#include "ATen/cuda/ATenCUDAGeneral.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/CUDAStream.h"
#include "ATen/cuda/CUDAGuard.h"
#include "ATen/cuda/Exceptions.h"
#include "c10/util/Exception.h"

#include "cuda_runtime_api.h"

#include <cstdint>
#include <utility>

namespace at { namespace cuda {

/*
* CUDAEvents 是不可复制但可移动的 CUDA 事件包装器
*
* CUDAEvents 在被流记录时延迟构造。事件具有设备属性，
* 该属性从第一个记录流获取。后续记录到此事件的流必须共享该设备，
* 但任何设备上的流都可以等待该事件。
*/
struct AT_CUDA_API CUDAEvent {
  // 常量定义
  static constexpr unsigned int DEFAULT_FLAGS = cudaEventDisableTiming; // 默认禁用计时功能

  // 构造函数
  CUDAEvent(unsigned int flags = DEFAULT_FLAGS)
  : flags_{flags} { }

  // 注意：事件销毁在创建设备上执行，避免在其他设备上创建CUDA上下文
  ~CUDAEvent() {
    try {
      if (is_created_) {
        at::cuda::CUDAGuard device_guard(static_cast<int16_t>(device_index_));
        cudaEventDestroy(event_); // 销毁CUDA事件
      }
    } catch (...) { /* 不抛出异常 */ }
  }

  // 禁止拷贝构造和拷贝赋值
  CUDAEvent(const CUDAEvent&) = delete;
  CUDAEvent& operator=(const CUDAEvent&) = delete;

  // 移动构造和移动赋值
  CUDAEvent(CUDAEvent&& other) { moveHelper(std::move(other)); }
  CUDAEvent& operator=(CUDAEvent&& other) {
    moveHelper(std::move(other));
    return *this;
  }

  // 类型转换运算符，返回底层CUDA事件句柄
  operator cudaEvent_t() const { return event(); }

  // 小于运算符（用于集合排序）
  friend bool operator<(const CUDAEvent& left, const CUDAEvent& right) {
    return left.event_ < right.event_;
  }

  // 查询事件是否已创建
  bool isCreated() const { return is_created_; }
  // 获取事件关联的设备索引
  int64_t device() const { return device_index_; }
  // 获取底层CUDA事件句柄
  cudaEvent_t event() const { return event_; }

  // 注意：cudaEventQuery可以从任何设备安全调用
  // 检查事件是否已完成
  bool happened() const {
    return (was_recorded_ && cudaEventQuery(event_) == cudaSuccess);
  }

  // 在当前CUDA流上记录事件
  void record() { record(getCurrentCUDAStream()); }

  // 仅当事件未被记录过时，在指定流上记录事件
  void recordOnce(const CUDAStream& stream) {
    if (!was_recorded_) record(stream);
  }

  // 注意：cudaEventRecord必须在与流相同的设备上调用
  // 在指定流上记录事件
  void record(const CUDAStream& stream) {
    at::cuda::CUDAGuard guard(static_cast<int16_t>(stream.device_index()));

    if (is_created_) {
      AT_ASSERT(device_index_ == stream.device_index());
    } else {
      AT_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags_)); // 创建CUDA事件
      is_created_ = true;
      device_index_ = stream.device_index(); // 设置关联设备
    }

    AT_CUDA_CHECK(cudaEventRecord(event_, stream)); // 记录事件到流
    was_recorded_ = true;
  }

  // 注意：cudaStreamWaitEvent必须在与流相同的设备上调用
  // 使指定流等待此事件
  void block(const CUDAStream& stream) {
    if (is_created_) {
      at::cuda::CUDAGuard guard(static_cast<int16_t>(stream.device_index()));
      AT_CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0)); // 流等待事件
    }
  }

private:
  unsigned int flags_ = DEFAULT_FLAGS;       // 事件标志
  bool is_created_ = false;                  // 是否已创建
  bool was_recorded_ = false;                // 是否已被记录过
  int64_t device_index_ = -1;                // 关联设备索引
  cudaEvent_t event_;                        // CUDA事件句柄

  // 移动辅助函数
  void moveHelper(CUDAEvent&& other) {
    std::swap(flags_, other.flags_);
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }
};

} // namespace cuda
} // namespace at