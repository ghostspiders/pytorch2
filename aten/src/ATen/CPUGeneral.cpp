#include <ATen/CPUGeneral.h>
#include <atomic>
#include <memory>
#include <thread>


namespace at {  // ATen库命名空间

// 原子整型变量，用于线程数控制，初始值-1表示未设置
std::atomic<int> num_threads(-1);

// 设置线程数（线程安全）
void set_num_threads(int num_threads_) {
  if (num_threads_ >= 0)  // 仅接受非负值
    num_threads.store(num_threads_);  // 原子存储新值
}

// 获取当前线程数（线程安全）
int get_num_threads() { 
  return num_threads.load();  // 原子读取当前值
}

}  // namespace at

