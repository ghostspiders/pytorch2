
#include "Handle.h"  
#include "ATen/cuda/Exceptions.h"  // CUDA异常处理头文件
#include <unordered_map>  // 哈希表容器
#include <mutex>  // 互斥锁

// TODO: 建议后续移除mutex，改为在at::Context中初始化这些句柄
// 与延迟CUDA初始化一起完成

namespace at { namespace native {  // ATen库的native命名空间

namespace {  // 匿名命名空间

struct Handle {
  cudnnHandle_t handle;  // cuDNN库句柄
  Handle() : handle(NULL) {
    AT_CUDNN_CHECK(cudnnCreate(&handle));  // 创建cuDNN句柄并检查错误
  }
  ~Handle() {
    if (handle) {
// 由于销毁顺序问题，在fbcode环境下可能出现CUDA上下文
// 已销毁时才调用此析构函数。@soumith和@colesbury决定
// 通过不销毁句柄作为临时解决方案
#ifdef NO_CUDNN_DESTROY_HANDLE  // 条件编译控制是否销毁句柄
#else
      cudnnDestroy(handle);  // 正常销毁cuDNN句柄
#endif
    }
  }
};

std::mutex mutex;  // 全局互斥锁
std::unordered_map<int, Handle> handles;  // 设备ID到句柄的映射

}  // 匿名命名空间结束

// 获取当前设备的cuDNN句柄
cudnnHandle_t getCudnnHandle()
{
  int device;
  AT_CUDA_CHECK(cudaGetDevice(&device));  // 获取当前CUDA设备ID

  std::lock_guard<std::mutex> guard(mutex);  // 加锁保证线程安全
  return handles[device].handle;  // 返回对应设备的cuDNN句柄
}

}} // namespace at::native结束
