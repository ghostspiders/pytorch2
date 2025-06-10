#include <ATen/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Exception.h>

#include <mutex>
#include <atomic>
#include <cstdint>
#include <deque>
#include <vector>
#include <array>

namespace at {
namespace cuda {

namespace {

// 内部实现完全隐藏
struct CUDAStreamInternals {
  CUDAStreamInternals() = default;

  ~CUDAStreamInternals() {
    if (stream) cudaStreamDestroy(stream);  // 析构时销毁CUDA流
  }

  DeviceIndex device_index = -1;  // 设备索引
  int32_t stream_id = -1;         // 流ID
  cudaStream_t stream = nullptr;  // CUDA流句柄
};

// 全局流状态和常量
static DeviceIndex num_gpus = -1;  // GPU数量
static constexpr int kStreamsPerPoolBits = 5;  // 每个池的流数量位数
static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;  // 每个池32个流
static constexpr unsigned int kDefaultFlags = cudaStreamNonBlocking;  // 默认非阻塞流

// 注意：HIP平台不支持流优先级
// 注意：数字越小优先级越高，0是默认优先级
#ifndef __HIP_PLATFORM_HCC__
  static int kHighPriority = -1;  // 高优先级
  static int kLowPriority = 0;    // 低优先级
#endif // __HIP_PLATFORM_HCC__

// 默认流（每个设备一个）
static std::once_flag init_flag;  // 一次性初始化标志
static std::vector<CUDAStreamInternals> default_streams;  // 默认流数组

// 非默认流
// 注意：CUDA设备数量在运行时确定，
// 低优先级和高优先级池在首次请求流时延迟初始化。
// device_flags跟踪每个设备的初始化状态，
// low/high_priority_counters以轮询方式返回池中的下一个流
static std::deque<std::once_flag> device_flags;  // 每个设备的初始化标志
static std::deque<std::atomic<uint32_t>> low_priority_counters;  // 低优先级流计数器
static std::deque<std::atomic<uint32_t>> high_priority_counters; // 高优先级流计数器
static std::vector<std::array<CUDAStreamInternals, kStreamsPerPool>> low_priority_streams;  // 低优先级流池
static std::vector<std::array<CUDAStreamInternals, kStreamsPerPool>> high_priority_streams; // 高优先级流池

// 注意[StreamId分配]
// ~~~~~~~~~~~~~~~~~~
// 如何分配流ID？
//
// -- 25位 -- -- 2位 --  -- 5位 -----
// 零         流类型      流索引
//
// 流类型:
//  00 = 默认流
//  01 = 低优先级流
//  10 = 高优先级流
//
// 这不是为了效率，只是为了更容易用位掩码提取索引
//
// 这是内部实现细节，我们保留重新编号流的权利
//
// 注意MSB必须为零，因为StreamId是有符号整数

// 流类型枚举
enum class StreamIdType : uint8_t {
  DEFAULT = 0x0,  // 默认流
  LOW     = 0x1,  // 低优先级流
  HIGH    = 0x2,  // 高优先级流
};

// 流类型输出运算符
std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  switch (s) {
    case StreamIdType::DEFAULT: stream << "DEFAULT"; break;
    case StreamIdType::LOW:     stream << "LOW";     break;
    case StreamIdType::HIGH:    stream << "HIGH";    break;
    default: stream << static_cast<uint8_t>(s); break;
  }
  return stream;
}

// 从StreamId提取流类型
static inline StreamIdType streamIdType(StreamId s) {
  return static_cast<StreamIdType>(s >> kStreamsPerPoolBits);
}

// 从StreamId提取流索引
static inline size_t streamIdIndex(StreamId s) {
  return static_cast<size_t>(s & ((1 << kStreamsPerPoolBits) - 1));
}

// 根据类型和索引创建StreamId
StreamId makeStreamId(StreamIdType st, size_t si) {
  return (static_cast<StreamId>(st) << kStreamsPerPoolBits) | static_cast<StreamId>(si);
}

// 检查指针是否在数组范围内
template <typename T, typename A>
static bool pointer_within(const T* ptr, const A& arr) {
  return std::greater_equal<const T*>()(ptr, arr.data()) && 
         std::less<const T*>()(ptr, arr.data() + arr.size());
}

// 根据CUDAStreamInternals指针获取StreamId
static StreamId CUDAStream_getStreamId(const CUDAStreamInternals* ptr) {
  DeviceIndex device_index = ptr->device_index;

  // 检查是否是默认流
  if (ptr == &default_streams[device_index]) {
    return makeStreamId(StreamIdType::DEFAULT, 0);
  }

  // 检查是否是低优先级流
  if (pointer_within<CUDAStreamInternals>(ptr, low_priority_streams[device_index])) {
    return makeStreamId(StreamIdType::LOW, ptr - low_priority_streams[device_index].data());
  }

  // 检查是否是高优先级流
  if (pointer_within<CUDAStreamInternals>(ptr, high_priority_streams[device_index])) {
    return makeStreamId(StreamIdType::HIGH, ptr - high_priority_streams[device_index].data());
  }

  AT_ASSERTM(0, "无法计算设备", device_index, "上流", ptr, "的ID(可能出现了严重错误)");
}

// 线程本地当前流数组
static thread_local CUDAStreamInternals** current_streams = nullptr;

// 初始化全局流状态
static void initGlobalStreamState() {
  num_gpus = device_count();  // 获取GPU数量

  // 调整容器大小
  default_streams.resize(num_gpus);
  device_flags.resize(num_gpus);
  low_priority_counters.resize(num_gpus);
  high_priority_counters.resize(num_gpus);
  low_priority_streams.resize(num_gpus);
  high_priority_streams.resize(num_gpus);

  // 初始化默认流
  for (auto i = decltype(num_gpus){0}; i < num_gpus; ++i) {
    default_streams[i].device_index = i;
    low_priority_counters[i] = 0;
    high_priority_counters[i] = 0;
  }
}

// 初始化指定设备的流池
static void initDeviceStreamState(DeviceIndex device_index) {
  // 切换到指定设备以确保流正确关联
  at::cuda::CUDAGuard device_guard{device_index};

  // 初始化低优先级和高优先级流池
  for (auto i = decltype(kStreamsPerPool){0}; i < kStreamsPerPool; ++i) {
    auto& lowpri_stream = low_priority_streams[device_index][i];
    auto& hipri_stream = high_priority_streams[device_index][i];

    lowpri_stream.device_index = device_index;
    hipri_stream.device_index = device_index;

    #ifndef __HIP_PLATFORM_HCC__
      // 创建带优先级的CUDA流
      C10_CUDA_CHECK(cudaStreamCreateWithPriority(
        &lowpri_stream.stream, kDefaultFlags, kLowPriority));
      C10_CUDA_CHECK(cudaStreamCreateWithPriority(
        &hipri_stream.stream, kDefaultFlags, kHighPriority));
    #else
      // HIP平台创建普通流
      C10_CUDA_CHECK(cudaStreamCreateWithFlags(&lowpri_stream.stream, kDefaultFlags));
      C10_CUDA_CHECK(cudaStreamCreateWithFlags(&hipri_stream.stream, kDefaultFlags));
    #endif
  }
}

// 初始化前端，确保只初始化一次
static void initCUDAStreamsOnce() {
  // 初始化默认流(全局一次)
  std::call_once(init_flag, initGlobalStreamState);

  if (current_streams) return;

  // 初始化线程本地当前流(指向默认流)
  current_streams = (CUDAStreamInternals**) malloc(num_gpus * sizeof(CUDAStreamInternals*));
  for (auto i = decltype(num_gpus){0}; i < num_gpus; ++i) {
    current_streams[i] = &default_streams[i];
  }
}

// 检查GPU索引是否有效
static inline void check_gpu(DeviceIndex device_index) {
  AT_ASSERT(device_index >= 0 && device_index < num_gpus);
}

// 获取轮询索引
static uint32_t get_idx(std::atomic<uint32_t> &counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;  // 取模保证在池范围内
}

// 根据CUDAStream获取内部结构指针
CUDAStreamInternals* CUDAStream_internals(CUDAStream s) {
  c10::DeviceIndex device_index = s.device_index();
  StreamIdType st = streamIdType(s.unwrap().id());
  size_t si = streamIdIndex(s.unwrap().id());
  switch (st) {
    case StreamIdType::DEFAULT:
      AT_ASSERTM(si == 0, "无法识别的流", s.unwrap(), "(应该是默认流但索引非零", si, ")");
      return &default_streams[device_index];
    case StreamIdType::LOW:
      return &low_priority_streams[device_index][si];
    case StreamIdType::HIGH:
      return &high_priority_streams[device_index][si];
    default:
      AT_ASSERTM(0, "无法识别的流", s.unwrap(), "(未知流类型", st, ")");
  }
}

// 从内部结构创建CUDAStream
CUDAStream CUDAStream_fromInternals(const CUDAStreamInternals* ptr) {
  return CUDAStream(CUDAStream::UNCHECKED,
                    Stream(c10::Device(DeviceType::CUDA, ptr->device_index),
                           CUDAStream_getStreamId(ptr)));
}

} // 匿名命名空间

// 获取CUDA流原生句柄
cudaStream_t CUDAStream::stream() const {
  auto ptr = CUDAStream_internals(*this);
  AT_ASSERT(ptr);
  return ptr->stream;
}

// 从池中获取流(首次调用会初始化设备流池)
CUDAStream getStreamFromPool(bool isHighPriority, DeviceIndex device_index) {
  initCUDAStreamsOnce();
  if (device_index == -1) device_index = current_device();
  check_gpu(device_index);

  // 初始化流池(每个设备一次)
  std::call_once(device_flags[device_index], initDeviceStreamState, device_index);

  if (isHighPriority) {
    const auto idx = get_idx(high_priority_counters[device_index]);
    return CUDAStream_fromInternals(&high_priority_streams[device_index][idx]);
  }

  const auto idx = get_idx(low_priority_counters[device_index]);
  return CUDAStream_fromInternals(&low_priority_streams[device_index][idx]);
}

// 获取默认CUDA流
CUDAStream getDefaultCUDAStream(DeviceIndex device_index) {
  initCUDAStreamsOnce();
  if (device_index == -1) device_index = current_device();
  check_gpu(device_index);
  return CUDAStream_fromInternals(&default_streams[device_index]);
}

// 获取当前CUDA流
CUDAStream getCurrentCUDAStream(DeviceIndex device_index) {
  initCUDAStreamsOnce();
  if (device_index == -1) device_index = current_device();
  check_gpu(device_index);
  return CUDAStream_fromInternals(current_streams[device_index]);
}

// 设置当前CUDA流
void setCurrentCUDAStream(CUDAStream stream) {
  initCUDAStreamsOnce();
  auto ptr = CUDAStream_internals(stream);
  AT_ASSERT(ptr);
  current_streams[ptr->device_index] = ptr;
}

// CUDAStream输出运算符
std::ostream& operator<<(std::ostream& stream, const CUDAStream& s) {
  return stream << s.unwrap();
}

} // namespace cuda
} // namespace at