#pragma once  // 头文件保护，防止重复包含

#include <condition_variable>  // 条件变量，用于线程同步
#include <functional>         // 函数对象支持
#include <mutex>              // 互斥锁
#include <queue>              // 任务队列
#include <thread>             // 线程支持
#include <utility>            // 通用工具组件

#include <c10/util/intrusive_ptr.h>  // 侵入式智能指针

namespace c10 {

namespace ivalue {
struct Future;  // 前向声明Future类型
} // namespace ivalue

// 线程池抽象基类
class CAFFE2_API TaskThreadPoolBase {
 public:
  // 提交无参任务到线程池
  virtual void run(const std::function<void()>& func) = 0;

  // 获取线程池大小
  virtual size_t size() const = 0;

  /**
   * 获取线程池中空闲线程数量
   */
  virtual size_t numAvailable() const = 0;

  /**
   * 检查当前线程是否属于线程池
   */
  virtual bool inThreadPool() const = 0;

  // 虚析构函数确保正确销毁派生类对象
  virtual ~TaskThreadPoolBase() noexcept {}
};

// 具体线程池实现类
class CAFFE2_API ThreadPool : public c10::TaskThreadPoolBase {
 protected:
  // 任务元素结构体，封装可执行任务
  struct task_element_t {
    bool run_with_id;  // 标识是否带线程ID执行
    const std::function<void()> no_id;  // 无ID任务函数
    const std::function<void(std::size_t)> with_id;  // 带ID任务函数

    // 构造无ID任务
    explicit task_element_t(const std::function<void()>& f)
        : run_with_id(false), no_id(f), with_id(nullptr) {}
    // 构造带ID任务  
    explicit task_element_t(const std::function<void(std::size_t)>& f)
        : run_with_id(true), no_id(nullptr), with_id(f) {}
  };

  // 成员变量
  std::queue<task_element_t> tasks_;  // 任务队列
  std::vector<std::thread> threads_;  // 工作线程集合
  std::mutex mutex_;                 // 保护共享数据的互斥锁
  std::condition_variable condition_; // 任务可用通知条件变量
  std::condition_variable completed_; // 任务完成通知条件变量
  bool running_;                     // 线程池运行状态标志
  bool complete_;                    // 任务完成状态标志
  std::size_t available_;            // 当前可用线程数
  std::size_t total_;                // 线程总数
  int numa_node_id_;                 // NUMA节点ID

 public:
  ThreadPool() = delete;  // 禁用默认构造函数

  // 构造函数
  explicit ThreadPool(std::size_t pool_size, int numa_node_id = -1);

  // 析构函数
  ~ThreadPool();

  // 实现基类接口 - 获取线程池大小
  size_t size() const override;

  // 实现基类接口 - 获取空闲线程数
  size_t numAvailable() const override;

  // 实现基类接口 - 检查当前线程是否属于线程池
  bool inThreadPool() const override;

  // 实现基类接口 - 提交无参任务
  void run(const std::function<void()>& func) override;

  // 提交带线程ID的任务
  template <typename Task>
  void runTaskWithID(Task task) {
    std::unique_lock<std::mutex> lock(mutex_);

    // 将任务加入队列并通知一个工作线程
    tasks_.push(
        task_element_t(static_cast<std::function<void(std::size_t)>>(task)));
    complete_ = false;
    condition_.notify_one();
  }

  /// @brief 等待所有任务完成
  void waitWorkComplete();

  // @brief 等待特定Future完成
  void workOnTasksUntilCompleted(c10::intrusive_ptr<ivalue::Future> future);

 protected:
  // 线程初始化钩子函数，子类可重写
  virtual void init_thread() {}

 private:
  // 工作线程主循环
  void main_loop(std::size_t index);
};

// 获取全局工作队列单例
CAFFE2_API ThreadPool& global_work_queue();

} // namespace c10