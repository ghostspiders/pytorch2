#include "ATen/ATen.h"  // ATen核心库，提供张量操作
#include "THC.h"        // Torch CUDA库，用于CUDA相关操作

// 如果启用了MAGMA支持（通过USE_MAGMA宏控制）
#ifdef USE_MAGMA
#include <magma.h>       // MAGMA库头文件 - 用于GPU加速的线性代数计算
#include <magma_types.h> // MAGMA类型定义
#endif

namespace at {    // ATen命名空间
namespace native { // 原生函数命名空间

#ifdef USE_MAGMA

// MAGMA队列的RAII（资源获取即初始化）封装类
// 用于管理MAGMA队列的生命周期
struct MAGMAQueue {

  // 禁止默认构造（未指定设备ID会引发问题）
  MAGMAQueue() = delete;

  // 显式构造函数，接收设备ID作为参数
  explicit MAGMAQueue(int64_t device_id) {
    auto& context = at::globalContext();
    // 从当前CUDA环境创建MAGMA队列
    magma_queue_create_from_cuda(
      device_id,  // CUDA设备ID
      at::cuda::getCurrentCUDAStream(),         // 获取当前CUDA流
      at::cuda::getCurrentCUDABlasHandle(),     // 获取当前cuBLAS句柄
      at::cuda::getCurrentCUDASparseHandle(),   // 获取当前cuSPARSE句柄
      &magma_queue_);  // 输出参数：创建的MAGMA队列
  }

  // 获取底层MAGMA队列
  magma_queue_t get_queue() const { return magma_queue_; }

  // 析构函数：自动销毁MAGMA队列
  ~MAGMAQueue() {
    magma_queue_destroy(magma_queue_);
  }

 private:
  magma_queue_t magma_queue_;  // MAGMA队列句柄
};

// 将int64_t安全转换为magma_int_t的辅助函数
// 避免因数值范围溢出导致的问题
static inline magma_int_t magma_int_cast(int64_t value, const char* varname) {
  auto result = static_cast<magma_int_t>(value);
  // 检查转换后的值是否与原始值一致（防止截断）
  if (static_cast<int64_t>(result) != value) {
    // 抛出错误信息，包含变量名和数值信息
    AT_ERROR("magma: The value of ", varname, "(", (long long)value,
             ") is too large to fit into a magma_int_t (", sizeof(magma_int_t), " bytes)");
  }
  return result;
}
#endif

// 创建固定内存(pinned memory)存储的模板函数
// 固定内存可加速主机-设备间的数据传输
template<class T>
static inline Storage pin_memory(int64_t size, Tensor dummy) {
  int64_t adjusted_size = size * sizeof(T);  // 计算实际需要的字节数
  auto* allocator = cuda::getPinnedMemoryAllocator();  // 获取固定内存分配器
  // 创建CPU后端的字节存储，使用固定内存分配器
  auto& backend = dummy.type().toBackend(Backend::CPU).toScalarType(kByte);
  return backend.storageWithAllocator(adjusted_size, allocator);
}
  
} // namespace native
} // namespace at