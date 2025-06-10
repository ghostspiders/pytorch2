// 防止头文件重复包含
#pragma once

// 引入必要的头文件
#include <ATen/cuda/detail/CUDAGuardImpl.h>  // CUDA设备保护的实现细节
#include <c10/DeviceType.h>                  // 设备类型定义
#include <c10/impl/InlineDeviceGuard.h>      // 内联设备保护模板
#include <c10/impl/InlineStreamGuard.h>      // 内联流保护模板

#include <cstddef>  // 标准库大小定义

namespace at { namespace cuda {

// ==============================================
// CUDAGuard - 专用于CUDA设备的RAII保护类
// ==============================================
struct CUDAGuard {
  // 禁止默认构造（RAII类通常需要显式初始化）
  explicit CUDAGuard() = delete;

  // 通过设备索引构造（立即切换设备）
  explicit CUDAGuard(DeviceIndex device_index) : guard_(device_index) {}

  // 通过Device对象构造（验证是否为CUDA设备）
  explicit CUDAGuard(Device device) : guard_(device) {}

  // 禁止拷贝构造/赋值
  CUDAGuard(const CUDAGuard&) = delete;
  CUDAGuard& operator=(const CUDAGuard&) = delete;

  // 禁止移动构造/赋值（因为没有未初始化状态）
  CUDAGuard(CUDAGuard&& other) = delete;
  CUDAGuard& operator=(CUDAGuard&& other) = delete;

  // 设备管理方法
  void set_device(Device device) { guard_.set_device(device); }       // 设置设备（验证CUDA类型）
  void reset_device(Device device) { guard_.reset_device(device); }   // 别名set_device
  void set_index(DeviceIndex device_index) { guard_.set_index(device_index); } // 通过索引设置

  // 状态查询
  Device original_device() const { return guard_.original_device(); }  // 获取构造时的原始设备
  Device current_device() const { return guard_.current_device(); }    // 获取当前设备

private:
  // 使用内联设备保护模板（编译时多态）
  c10::impl::InlineDeviceGuard<at::cuda::impl::CUDAGuardImpl> guard_;
};

// ==============================================
// OptionalCUDAGuard - 可选的CUDA设备保护
// ==============================================
struct OptionalCUDAGuard {
  // 默认构造（未初始化状态）
  explicit OptionalCUDAGuard() : guard_() {}

  // 通过可选设备构造（可能不切换设备）
  explicit OptionalCUDAGuard(optional<Device> device_opt) : guard_(device_opt) {}
  explicit OptionalCUDAGuard(optional<DeviceIndex> device_index_opt) : guard_(device_index_opt) {}

  // 禁止拷贝/移动
  OptionalCUDAGuard(const OptionalCUDAGuard&) = delete;
  OptionalCUDAGuard& operator=(const OptionalCUDAGuard&) = delete;
  OptionalCUDAGuard(OptionalCUDAGuard&& other) = delete;
  OptionalCUDAGuard& operator=(OptionalCUDAGuard&& other) = delete;

  // 设备管理方法
  void set_device(Device device) { guard_.set_device(device); }       // 初始化并设置设备
  void reset_device(Device device) { guard_.reset_device(device); }   // 别名set_device
  void set_index(DeviceIndex device_index) { guard_.set_index(device_index); }

  // 状态查询
  optional<Device> original_device() const { return guard_.original_device(); }  // 原始设备（可能空值）
  optional<Device> current_device() const { return guard_.current_device(); }    // 当前设备（可能空值）
  void reset() { guard_.reset(); }  // 恢复到原始设备并转为未初始化状态

private:
  // 使用内联可选设备保护模板
  c10::impl::InlineOptionalDeviceGuard<at::cuda::impl::CUDAGuardImpl> guard_;
};

// ==============================================
// CUDAStreamGuard - CUDA流和设备双重RAII保护
// ==============================================
struct CUDAStreamGuard {
  // 禁止默认构造
  explicit CUDAStreamGuard() = delete;

  // 通过流对象构造（自动切换设备和流）
  explicit CUDAStreamGuard(Stream stream) : guard_(stream) {}

  // 禁止拷贝/移动
  CUDAStreamGuard(const CUDAStreamGuard&) = delete;
  CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;
  CUDAStreamGuard(CUDAStreamGuard&& other) = delete;
  CUDAStreamGuard& operator=(CUDAStreamGuard&& other) = delete;

  // 流管理
  void reset_stream(Stream stream) { guard_.reset_stream(stream); }  // 重置流（可能跳过冗余设置）

  // 状态查询
  CUDAStream original_stream() const {  // 构造时的原始流
    return CUDAStream(CUDAStream::UNCHECKED, guard_.original_stream());
  }
  CUDAStream current_stream() const {   // 当前设置的流
    return CUDAStream(CUDAStream::UNCHECKED, guard_.current_stream());
  }
  Device current_device() const { return guard_.current_device(); }  // 当前设备
  Device original_device() const { return guard_.original_device(); } // 原始设备

private:
  // 使用内联流保护模板
  c10::impl::InlineStreamGuard<at::cuda::impl::CUDAGuardImpl> guard_;
};

// ==============================================
// OptionalCUDAStreamGuard - 可选的CUDA流保护
// ==============================================
struct OptionalCUDAStreamGuard {
  // 默认构造（未初始化）
  explicit OptionalCUDAStreamGuard() : guard_() {}

  // 通过流对象构造（显式初始化）
  explicit OptionalCUDAStreamGuard(Stream stream) : guard_(stream) {}
  explicit OptionalCUDAStreamGuard(optional<Stream> stream_opt) : guard_(stream_opt) {}

  // 禁止拷贝/移动
  OptionalCUDAStreamGuard(const OptionalCUDAStreamGuard&) = delete;
  OptionalCUDAStreamGuard& operator=(const OptionalCUDAStreamGuard&) = delete;
  OptionalCUDAStreamGuard(OptionalCUDAStreamGuard&& other) = delete;
  OptionalCUDAStreamGuard& operator=(OptionalCUDAStreamGuard&& other) = delete;

  // 流管理
  void reset_stream(Stream stream) { guard_.reset_stream(stream); }  // 初始化并设置流

  // 状态查询
  optional<CUDAStream> original_stream() const {  // 原始流（可能空值）
    auto r = guard_.original_stream();
    return r.has_value() ? make_optional(CUDAStream(CUDAStream::UNCHECKED, r.value())) : nullopt;
  }
  optional<CUDAStream> current_stream() const {   // 当前流（可能空值）
    auto r = guard_.current_stream();
    return r.has_value() ? make_optional(CUDAStream(CUDAStream::UNCHECKED, r.value())) : nullopt;
  }
  void reset() { guard_.reset(); }  // 恢复原始状态并取消初始化

private:
  // 使用内联可选流保护模板
  c10::impl::InlineOptionalStreamGuard<at::cuda::impl::CUDAGuardImpl> guard_;
};

}} // namespace at::cuda