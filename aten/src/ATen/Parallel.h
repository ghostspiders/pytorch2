#pragma once
#include <ATen/ATen.h>
#include <atomic>
#include <cstddef>
#include <exception>

#ifdef _OPENMP
#include <omp.h>  // 如果定义了_OPENMP，则包含OpenMP头文件
#endif

namespace at {
namespace internal {
// 这个参数是启发式选择的，用于确定需要并行处理的最小工作量。
// 例如，在求和数组时，认为对长度小于32768的数组进行并行化效率不高。
// 此外，任何并行算法（如parallel_reduce）都不应将工作拆分成小于GRAIN_SIZE的块。
constexpr int64_t GRAIN_SIZE = 32768;
} // namespace internal

// 向上取整除法函数
inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

// 获取最大线程数
inline int get_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();  // 如果支持OpenMP，返回OpenMP的最大线程数
#else
  return 1;  // 否则返回1（单线程）
#endif
}

// 获取当前线程编号
inline int get_thread_num() {
#ifdef _OPENMP
  return omp_get_thread_num();  // 如果支持OpenMP，返回当前线程编号
#else
  return 0;  // 否则返回0（主线程）
#endif
}

// 检查是否在并行区域内
inline bool in_parallel_region() {
#ifdef _OPENMP
  return omp_in_parallel();  // 如果支持OpenMP，检查是否在并行区域内
#else
  return false;  // 否则返回false
#endif
}

// 并行for循环模板函数
template <class F>
inline void parallel_for(
    const int64_t begin,     // 循环起始索引
    const int64_t end,       // 循环结束索引
    const int64_t grain_size, // 每个线程处理的最小数据量（粒度）
    const F& f) {            // 要并行执行的函数
#ifdef _OPENMP
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;  // 错误标志，用于捕获异常
  std::exception_ptr eptr;  // 异常指针，用于存储第一个发生的异常
  
  // 如果不在并行区域内且数据量足够大，则开启并行区域
#pragma omp parallel if (!omp_in_parallel() && ((end - begin) >= grain_size))
  {
    int64_t num_threads = omp_get_num_threads();  // 获取当前线程总数
    int64_t tid = omp_get_thread_num();          // 获取当前线程ID
    int64_t chunk_size = divup((end - begin), num_threads);  // 计算每个线程处理的数据块大小
    int64_t begin_tid = begin + tid * chunk_size;  // 计算当前线程的起始位置
    
    if (begin_tid < end) {  // 确保起始位置有效
      try {
        // 执行用户函数，处理从begin_tid到min(end, chunk_size + begin_tid)的数据
        f(begin_tid, std::min(end, chunk_size + begin_tid));
      } catch (...) {
        // 如果发生异常，设置错误标志并保存第一个异常
        if (!err_flag.test_and_set()) {
          eptr = std::current_exception();
        }
      }
    }
  }
  
  // 如果有异常发生，重新抛出
  if (eptr) {
    std::rethrow_exception(eptr);
  }
#else
  // 如果不支持OpenMP，直接串行执行
  if (begin < end) {
    f(begin, end);
  }
#endif
}

// 并行归约模板函数
template <class scalar_t, class F, class SF>
inline scalar_t parallel_reduce(
    const int64_t begin,      // 归约起始索引
    const int64_t end,        // 归约结束索引
    const int64_t grain_size, // 每个线程处理的最小数据量（粒度）
    const scalar_t ident,     // 归约操作的初始值
    const F f,                // 归约操作函数
    const SF sf) {            // 合并部分结果的函数
  if (get_num_threads() == 1) {
    // 如果只有一个线程，直接串行执行
    return f(begin, end, ident);
  } else {
    // 计算需要的结果数量（根据粒度划分）
    const int64_t num_results = divup((end - begin), grain_size);
    std::vector<scalar_t> results(num_results);  // 存储部分结果的向量
    scalar_t* results_data = results.data();
    
    // 并行执行归约操作
#pragma omp parallel for if ((end - begin) >= grain_size)
    for (int64_t id = 0; id < num_results; id++) {
      int64_t i = begin + id * grain_size;
      // 计算每个数据块的归约结果
      results_data[id] = f(i, i + std::min(end - i, grain_size), ident);
    }
    
    // 合并所有部分结果
    return std::accumulate(
        results_data, results_data + results.size(), ident, sf);
  }
}

} // namespace at