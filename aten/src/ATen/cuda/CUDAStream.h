#pragma once

#include <cstdint>
#include <utility>

#include "cuda_runtime_api.h"

#include <ATen/cuda/ATenCUDAGeneral.h>
#include <c10/util/Exception.h>
#include <c10/Stream.h>

/*
* Stream pool note.
*
* A CUDAStream is an abstraction of an actual cuStream on the GPU. CUDAStreams
* are backed by cuStreams, but they use several pools to minimize the costs
* associated with creating, retaining, and destroying cuStreams.
*
* There are three pools per device, and a device's pools are lazily created.
*
* The first pool contains only the default stream. When the default stream
* is requested it's returned.
*
* The second pool is the "low priority" or "default priority" streams. In
* HIP builds there is no distinction between streams in this pool and streams
* in the third pool (below). There are 32 of these streams per device, and
* when a stream is requested one of these streams is returned round-robin.
* That is, the first stream requested is at index 0, the second at index 1...
* to index 31, then index 0 again.
*
* This means that if 33 low priority streams are requested, the first and
* last streams requested are actually the same stream (under the covers)
* and kernels enqueued on them cannot run concurrently.
*
* The third pool is the "high priority" streams. The third pool acts like
* the second pool except the streams are created with a higher priority.
*
* These pools suggest that stream users should prefer many short-lived streams,
* as the cost of acquiring and releasing streams is effectively zero. If
* many longer-lived streams are required in performance critical scenarios
* then the functionality here may need to be extended to allow, for example,
* "reserving" a subset of the pool so that other streams do not accidentally
* overlap the performance critical streams.
*
* Note: although the notion of "current stream for device" is thread local
* (every OS thread has a separate current stream, as one might expect),
* the stream pool is global across all threads; stream 0 is always stream 0
* no matter which thread you use it on.  Multiple threads can synchronize
* on the same stream.  Although the CUDA documentation is not very clear
* on the matter, streams are thread safe; e.g., it is safe to enqueue
* a kernel on the same stream from two different threads.
*/

namespace at {
namespace cuda {

// 表示CUDA流的对象。这是对c10::Stream的包装，
// 但提供了额外的CUDA特定功能（如转换为cudaStream_t），
// 并保证包装的c10::Stream确实是CUDA流。
class AT_CUDA_API CUDAStream {
public:
  // 用于无检查构造的标记
  enum Unchecked { UNCHECKED };

  /// 从Stream构造CUDAStream。此构造会进行检查，
  /// 如果Stream实际上不是CUDA流，将引发错误。
  explicit CUDAStream(Stream stream) : stream_(stream) {
    AT_CHECK(stream_.device_type() == DeviceType::CUDA);
  }

  /// 从Stream无检查地构造CUDAStream。
  /// 使用"命名"构造函数习惯用法，可以这样调用：
  /// CUDAStream(CUDAStream::UNCHECKED, stream)
  explicit CUDAStream(Unchecked, Stream stream) : stream_(stream) {}

  // 比较运算符
  bool operator==(const CUDAStream& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const CUDAStream& other) const noexcept {
    return unwrap() != other.unwrap();
  }

  /// 隐式转换为cudaStream_t
  operator cudaStream_t() const { return stream(); }

  /// 隐式转换为Stream（即忘记这个流是CUDA流）
  operator Stream() const { return unwrap(); }

  /// 获取此流关联的CUDA设备索引
  DeviceIndex device_index() const { return stream_.device_index(); }

  /// 获取此流关联的完整Device对象。保证是CUDA设备。
  Device device() const { return Device(DeviceType::CUDA, device_index()); }

  /// 返回此特定流对应的流ID
  StreamId id() const { return stream_.id(); }

  /// 显式转换为cudaStream_t
  cudaStream_t stream() const;

  /// 显式转换为Stream
  Stream unwrap() const { return stream_; }

  /// 将CUDAStream可逆地打包为uint64_t表示形式。
  /// 这在将CUDAStream存储在C结构体中时可能有用，
  /// 因为不能方便地放置CUDAStream对象本身。
  ///
  /// 可以使用unpack()解包CUDAStream。
  /// uint64_t的格式未指定且可能更改。
  uint64_t pack() const noexcept {
    return stream_.pack();
  }

  /// 从pack()生成的uint64_t表示形式解包CUDAStream
  static CUDAStream unpack(uint64_t bits) {
    return CUDAStream(Stream::unpack(bits));
  }

private:
  Stream stream_;  // 底层Stream对象
};

/**
 * 从CUDA流池获取新流。你可以认为这是在"创建"新流，
 * 但实际上并没有这样的创建操作；而是从池中预分配流并以轮询方式返回。
 *
 * 可以通过设置isHighPriority为true从高优先级池请求流，
 * 或通过设置device参数为特定设备请求流（默认为当前CUDA流设备）。
 */
CAFFE2_API CUDAStream
getStreamFromPool(const bool isHighPriority = false, DeviceIndex device = -1);

/**
 * 获取指定CUDA设备的默认CUDA流，如果没有传递设备索引，
 * 则获取当前设备的默认流。默认流是在不显式使用流时大多数计算发生的地方。
 */
CAFFE2_API CUDAStream getDefaultCUDAStream(DeviceIndex device_index = -1);

/**
 * 获取指定CUDA设备的当前CUDA流，如果没有传递设备索引，
 * 则获取当前设备的当前流。当前CUDA流通常是设备的默认CUDA流，
 * 但如果有人调用了'setCurrentCUDAStream'或使用了'StreamGuard'/'CUDAStreamGuard'，
 * 则可能不同。
 */
CAFFE2_API CUDAStream getCurrentCUDAStream(DeviceIndex device_index = -1);

/**
 * 将传入流所在设备的当前流设置为传入流。
 * 注意：此函数与当前设备无关：它切换的是传入流所在设备的当前流。
 *
 * 为避免混淆，建议使用'CUDAStreamGuard'代替此函数
 * （它会以预期的方式切换当前设备和当前流，并在之后重置回原始状态）。
 */
CAFFE2_API void setCurrentCUDAStream(CUDAStream stream);

// CUDAStream输出运算符
C10_API std::ostream& operator<<(std::ostream& stream, const CUDAStream& s);

} // namespace cuda
} // namespace at

// std命名空间特化，为CUDAStream提供哈希支持
namespace std {
  template <>
  struct hash<at::cuda::CUDAStream> {
    size_t operator()(at::cuda::CUDAStream s) const noexcept {
      return std::hash<c10::Stream>{}(s.unwrap());
    }
  };
} // namespace std