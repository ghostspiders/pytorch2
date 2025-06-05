#pragma once

#include <c10/util/C++17.h>
#include <c10/util/Exception.h>
#include <atomic>
#include <stdexcept>

namespace c10 {

/**
 * intrusive_ptr<T> 是 shared_ptr<T> 的替代方案，由于它将引用计数内嵌在对象本身中，
 * 因此具有更好的性能。
 * 你的类 T 需要继承自 intrusive_ptr_target 才能用于 intrusive_ptr<T>。
 */

// 注意 [栈分配的 intrusive_ptr_target 安全性]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// std::enable_shared_from_this 的一个众所周知的问题是它允许你从栈分配的对象创建 std::shared_ptr，
// 这是完全错误的，因为对象会在栈返回时被销毁。在 intrusive_ptr 中，我们可以检测到这种情况，
// 因为我们设置继承自 intrusive_ptr_target 的对象的 refcount/weakcount 为零，
// 除非我们能证明对象是动态分配的（例如通过 make_intrusive）。

class C10_API intrusive_ptr_target {
  // 注意 [侵入式引用计数的弱引用]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // 引用计数方案：
  //
  // - refcount == 对象的强引用数量
  // - weakcount == 对象的弱引用数量，如果 refcount > 0 则再加一
  //   不变式：refcount > 0 => weakcount > 0
  //
  mutable std::atomic<size_t> refcount_;  // 强引用计数
  mutable std::atomic<size_t> weakcount_; // 弱引用计数

  template <typename T, typename NullType>
  friend class intrusive_ptr;
  template <typename T, typename NullType>
  friend class weak_intrusive_ptr;

 protected:
  // 受保护的析构函数。我们永远不想直接析构 intrusive_ptr_target*
  virtual ~intrusive_ptr_target() {
// 禁用 -Wterminate 和 -Wexceptions，以便我们可以在析构函数中使用断言
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wterminate"
#pragma GCC diagnostic ignored "-Wexceptions"
    // 确保没有 intrusive_ptr 指向该对象时才析构
    AT_ASSERTM(
        refcount_.load() == 0,
        "尝试析构一个仍有 intrusive_ptr 指向的 intrusive_ptr_target");
    // 确保没有 weak_intrusive_ptr 指向该对象时才析构
    AT_ASSERTM(
        weakcount_.load() == 0,
        "尝试析构一个仍有 weak_intrusive_ptr 指向的 intrusive_ptr_target");
#pragma GCC diagnostic pop
  }

  // 构造函数，初始化引用计数为0
  constexpr intrusive_ptr_target() noexcept : refcount_(0), weakcount_(0) {}

  // 支持拷贝和移动，但引用计数不参与（因为它们是内存位置的固有属性）
  intrusive_ptr_target(intrusive_ptr_target&& other) noexcept : intrusive_ptr_target() {}
  intrusive_ptr_target& operator=(intrusive_ptr_target&& other) noexcept { return *this; }
  intrusive_ptr_target(const intrusive_ptr_target& other) noexcept : intrusive_ptr_target() {}
  intrusive_ptr_target& operator=(const intrusive_ptr_target& other) noexcept { return *this; }

 private:
  /**
   * 当 refcount 降为0时调用。
   * 你可以重写此方法来释放昂贵的资源。
   * 可能仍有弱引用，因此你的对象可能尚未被析构，
   * 但你可以假设该对象不再被使用。
   */
  virtual void release_resources() {}
};

// 命名空间 detail 包含一些实现细节
namespace detail {
template <class TTarget>
struct intrusive_target_default_null_type final {
  static constexpr TTarget* singleton() noexcept {
    return nullptr;  // 默认使用 nullptr 作为空指针
  }
};

// 指针赋值辅助函数，处理不同类型的空指针
template<class TTarget, class ToNullType, class FromNullType>
TTarget* assign_ptr_(TTarget* rhs) {
  if (FromNullType::singleton() == rhs) {
    return ToNullType::singleton();  // 如果源是空指针，返回目标类型的空指针
  } else {
    return rhs;  // 否则直接返回指针
  }
}
} // namespace detail

// 弱引用指针的前向声明
template <class TTarget, class NullType>
class weak_intrusive_ptr;

// 侵入式智能指针主模板
template <
    class TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>>
class intrusive_ptr final {
 private:
  TTarget* target_;  // 管理的裸指针

  // 友元声明
  template <class TTarget2, class NullType2>
  friend class intrusive_ptr;
  friend class weak_intrusive_ptr<TTarget, NullType>;

  // 增加引用计数
  void retain_() {
    if (target_ != NullType::singleton()) {
      size_t new_refcount = ++target_->refcount_;
      AT_ASSERTM(
          new_refcount != 1,
          "intrusive_ptr: 引用计数归零后不能增加。");
    }
  }

  // 重置指针（减少引用计数，必要时删除对象）
  void reset_() noexcept {
    if (target_ != NullType::singleton() && --target_->refcount_ == 0) {
      // 如果强引用归零，减少弱引用计数并释放资源
      auto weak_count = --target_->weakcount_;
      const_cast<c10::guts::remove_const_t<TTarget>*>(target_)->release_resources();
      if (weak_count == 0) {
        delete target_;  // 如果没有弱引用了，删除对象
      }
    }
    target_ = NullType::singleton();  // 重置为空指针
  }

  // 私有构造函数，不增加引用计数
  explicit intrusive_ptr(TTarget* target) noexcept : target_(target) {}

 public:
  using element_type = TTarget;

  // 默认构造函数，初始化为空指针
  intrusive_ptr() noexcept : intrusive_ptr(NullType::singleton()) {}

  // 移动构造函数
  intrusive_ptr(intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
    rhs.target_ = NullType::singleton();  // 转移所有权
  }

  // 从不同类型的 intrusive_ptr 移动构造
  template <class From, class FromNullType>
  /* implicit */ intrusive_ptr(intrusive_ptr<From, FromNullType>&& rhs) noexcept
      : target_(detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "类型不匹配。intrusive_ptr 移动构造函数获取了错误类型的指针。");
    rhs.target_ = FromNullType::singleton();
  }

  // 拷贝构造函数
  intrusive_ptr(const intrusive_ptr& rhs) : target_(rhs.target_) {
    retain_();  // 增加引用计数
  }

  // 从不同类型的 intrusive_ptr 拷贝构造
  template <class From, class FromNullType>
  /* implicit */ intrusive_ptr(
      const intrusive_ptr<From, FromNullType>& rhs)
      : target_(detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "类型不匹配。intrusive_ptr 拷贝构造函数获取了错误类型的指针。");
    retain_();
  }

  // 析构函数
  ~intrusive_ptr() noexcept {
    reset_();  // 减少引用计数，必要时释放对象
  }

  // 移动赋值运算符
  intrusive_ptr& operator=(intrusive_ptr&& rhs) & noexcept {
    return operator=<TTarget, NullType>(std::move(rhs));
  }

  // 从不同类型的 intrusive_ptr 移动赋值
  template <class From, class FromNullType>
      intrusive_ptr& operator=(intrusive_ptr<From, FromNullType>&& rhs) &
      noexcept {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "类型不匹配。intrusive_ptr 移动赋值获取了错误类型的指针。");
    intrusive_ptr tmp = std::move(rhs);
    swap(tmp);
    return *this;
  }

  // 拷贝赋值运算符
  intrusive_ptr& operator=(const intrusive_ptr& rhs) & noexcept {
    return operator=<TTarget, NullType>(rhs);
  }

  // 从不同类型的 intrusive_ptr 拷贝赋值
  template <class From, class FromNullType>
      intrusive_ptr& operator=(const intrusive_ptr<From, NullType>& rhs) & {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "类型不匹配。intrusive_ptr 拷贝赋值获取了错误类型的指针。");
    intrusive_ptr tmp = rhs;
    swap(tmp);
    return *this;
  }

  // 获取裸指针
  TTarget* get() const noexcept {
    return target_;
  }

  // 解引用运算符
  const TTarget& operator*() const noexcept {
    return *target_;
  }
  TTarget& operator*() noexcept {
    return *target_;
  }

  // 成员访问运算符
  const TTarget* operator->() const noexcept {
    return target_;
  }
  TTarget* operator->() noexcept {
    return target_;
  }

  // 布尔转换，检查是否非空
  operator bool() const noexcept {
    return target_ != NullType::singleton();
  }

  // 重置指针
  void reset() noexcept {
    reset_();
  }

  // 交换两个指针
  void swap(intrusive_ptr& rhs) noexcept {
    TTarget* tmp = target_;
    target_ = rhs.target_;
    rhs.target_ = tmp;
  }

  // 检查是否非空（性能优化版本）
  bool defined() const noexcept {
    return target_ != NullType::singleton();
  }

  // 获取当前强引用计数
  size_t use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    return target_->refcount_.load();
  }

  // 获取当前弱引用计数
  size_t weak_use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    return target_->weakcount_.load();
  }

  // 检查是否是唯一所有者
  bool unique() const noexcept {
    return use_count() == 1;
  }

  /**
   * 释放所有权并返回裸指针。
   * 引用计数不会减少。
   * 你必须使用 intrusive_ptr::reclaim(ptr) 将返回的指针重新包装。
   */
  TTarget* release() noexcept {
    TTarget* result = target_;
    target_ = NullType::singleton();
    return result;
  }

  /**
   * 接管一个裸指针的所有权。
   * 引用计数不会增加。
   * 该指针必须是通过 intrusive_ptr::release() 获得的。
   */
  static intrusive_ptr reclaim(TTarget* owning_ptr) {
    // 确保指针是通过 release() 获得的
    AT_ASSERTM(
        owning_ptr == NullType::singleton() || owning_ptr->refcount_.load() > 0,
        "intrusive_ptr: 只能 reclaim() 通过 intrusive_ptr::release() 获得的所有权指针。");
    return intrusive_ptr(owning_ptr);
  }

  // 工厂函数，创建新对象并包装为 intrusive_ptr
  template <class... Args>
  static intrusive_ptr make(Args&&... args) {
    auto result = intrusive_ptr(new TTarget(std::forward<Args>(args)...));
    // 手动增加引用计数和弱引用计数
    ++result.target_->refcount_;
    ++result.target_->weakcount_;
    return result;
  }
};

// 创建 intrusive_ptr 的便捷函数
template <
    class TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>,
    class... Args>
inline intrusive_ptr<TTarget, NullType> make_intrusive(Args&&... args) {
  return intrusive_ptr<TTarget, NullType>::make(std::forward<Args>(args)...);
}

// 交换两个 intrusive_ptr
template <class TTarget, class NullType>
inline void swap(
    intrusive_ptr<TTarget, NullType>& lhs,
    intrusive_ptr<TTarget, NullType>& rhs) noexcept {
  lhs.swap(rhs);
}

// 比较运算符等后续代码...
} // namespace c10