#include <ATen/core/thread_pool.h>
#include <ATen/core/ivalue.h>

namespace c10 {

/**
 * 线程池构造函数
 * @param pool_size 线程池中线程的数量
 * @param numa_node_id NUMA节点ID，用于优化多核CPU架构下的内存访问
 * 
 * 初始化线程池中的所有线程，每个线程将执行main_loop函数
 */
ThreadPool::ThreadPool(std::size_t pool_size, int numa_node_id)
    : threads_(pool_size),    // 存储线程对象的vector
      running_(true),         // 控制线程池运行的标志位
      complete_(true),        // 表示所有任务是否完成的标志位
      available_(pool_size),  // 当前可用线程数
      total_(pool_size),      // 线程总数
      numa_node_id_(numa_node_id) {  // NUMA节点ID
  // 创建并启动所有工作线程
  for (std::size_t i = 0; i < pool_size; ++i) {
    threads_[i] = std::thread(std::bind(&ThreadPool::main_loop, this, i));
  }
}

/**
 * 线程池析构函数
 * 安全地停止所有线程并等待它们退出
 */
ThreadPool::~ThreadPool() {
  // 第一步：设置运行标志为false并通知所有线程
  {
    std::unique_lock<std::mutex> lock(mutex_);
    running_ = false;          // 设置停止标志
    condition_.notify_all();   // 唤醒所有等待中的线程
  }

  // 第二步：等待所有线程结束
  for (auto& t : threads_) {
    try {
      t.join();  // 等待线程结束
    } catch (const std::exception& e) {
      // 记录join异常，但不会影响析构过程
      std::cerr << "Thread join failed: " << e.what() << std::endl;
    }
  }
}

/**
 * 获取线程池大小
 * @return 线程池中的线程总数
 */
size_t ThreadPool::size() const {
  return threads_.size();
}

/**
 * 获取当前可用线程数
 * @return 当前可立即执行任务的线程数
 */
size_t ThreadPool::numAvailable() const {
  return available_;
}

/**
 * 检查当前线程是否属于线程池
 * @return 如果当前线程是线程池中的线程则返回true，否则返回false
 */
bool ThreadPool::inThreadPool() const {
  auto this_id = std::this_thread::get_id();
  for (auto& thread : threads_) {
    if (thread.get_id() == this_id) {
      return true;
    }
  }
  return false;
}

/**
 * 提交任务到线程池
 * @param func 要执行的任务函数
 * 
 * 将任务加入队列并通知一个工作线程开始处理
 */
void ThreadPool::run(const std::function<void()>& func) {
  std::unique_lock<std::mutex> lock(mutex_);

  // 将任务加入队列并通知一个工作线程
  tasks_.push(task_element_t(func));  // 封装任务
  complete_ = false;                  // 标记有未完成任务
  condition_.notify_one();            // 唤醒一个工作线程
}

/**
 * 等待所有工作完成
 * 阻塞当前线程直到线程池中所有任务都执行完毕
 */
void ThreadPool::waitWorkComplete() {
  std::unique_lock<std::mutex> lock(mutex_);
  // 等待完成信号
  while (!complete_) {
    completed_.wait(lock);
  }
}

/**
 * 处理任务直到Future完成
 * @param future 要等待完成的Future对象
 * 
 * 阻塞当前线程直到指定的Future完成
 */
void ThreadPool::workOnTasksUntilCompleted(
    c10::intrusive_ptr<ivalue::Future> future) {
  if (future->completed()) {
    return;  // 如果已完成则直接返回
  }
  
  std::condition_variable finished;
  // 添加完成回调
  future->addCallback([&] { 
    finished.notify_all(); 
  });

  // 等待Future完成
  std::unique_lock<std::mutex> future_lock(future->get_mutex());
  while (!future->completed()) {
    finished.wait(future_lock);
  }
}

/**
 * 工作线程主循环
 * @param index 线程索引号
 * 
 * 每个工作线程执行的核心循环，不断从任务队列中获取并执行任务
 */
void ThreadPool::main_loop(std::size_t index) {
  init_thread();  // 初始化线程（如设置NUMA亲和性等）

  while (running_) {
    // 等待任务或停止信号
    std::unique_lock<std::mutex> lock(mutex_);
    while (tasks_.empty() && running_) {
      condition_.wait(lock);
    }
    
    // 检查是否停止运行
    if (!running_) {
      break;
    }

    // 获取并执行任务
    {
      auto tasks = tasks_.front();  // 获取队列中的第一个任务
      tasks_.pop();                // 从队列中移除该任务
      --available_;                // 减少可用线程计数

      lock.unlock();  // 解锁，允许其他线程操作队列

      // 执行任务
      try {
        if (tasks.run_with_id) {
          tasks.with_id(index);  // 执行带线程ID的任务
        } else {
          tasks.no_id();        // 执行普通任务
        }
      } catch (const std::exception& e) {
        // 记录任务执行异常
        std::cerr << "Task execution failed: " << e.what() << std::endl;
      }

      // 更新状态
      lock.lock();  // 重新加锁
      ++available_; // 增加可用线程计数

      // 检查是否所有任务都已完成
      if (tasks_.empty() && available_ == total_) {
        complete_ = true;          // 设置完成标志
        completed_.notify_one();   // 通知等待线程
      }
    }
  } // while running_
}

/**
 * 获取全局工作队列
 * @return 全局单例线程池引用
 * 
 * 提供一个全局的单线程池，用于执行轻量级任务
 */
ThreadPool& global_work_queue() {
  static ThreadPool thread_pool(1);  // 全局单线程池
  return thread_pool;
}

} // namespace c10