#pragma once  // 防止头文件重复包含

#include <cstddef>
#include <sstream>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include <c10/util/intrusive_ptr.h>  // 引用智能指针工具
#include <c10/util/typeid.h>         // 类型ID支持
#include <c10/macros/Macros.h>       // 宏定义

namespace caffe2 {  // 命名空间：Caffe2（PyTorch的前身）

class Tensor;  // 前向声明Tensor类

/**
 * @brief Blob是一个通用容器，用于托管类型化的指针
 *
 * Blob托管一个指针及其类型信息，并负责在Blob被释放或重新分配时正确删除指针。
 * 虽然Blob可以包含任何类型，但最常见的是包含Tensor。
 */
class CAFFE2_API Blob final : public c10::intrusive_ptr_target {  // 继承自引用计数基类
 public:
  /**
   * 初始化一个空的Blob
   */
  Blob() noexcept : meta_(), pointer_(nullptr), has_ownership_(false) {}
  
  ~Blob() {
    Reset();  // 析构时自动释放资源
  }

  // 移动构造函数
  Blob(Blob&& other) noexcept : Blob() {
    swap(other);  // 通过swap实现移动语义
  }

  // 移动赋值运算符
  Blob& operator=(Blob&& other) noexcept {
    Blob(std::move(other)).swap(*this);
    return *this;
  }

  /**
   * 检查Blob中存储的内容是否为类型T
   */
  template <class T>
  bool IsType() const noexcept {
    return meta_.Match<T>();  // 使用TypeMeta进行类型匹配
  }

  /**
   * 返回Blob的类型元信息
   */
  inline const TypeMeta& meta() const noexcept {
    return meta_;
  }

  /**
   * 返回Blob类型的可打印名称
   */
  inline const char* TypeName() const noexcept {
    return meta_.name();  // 获取类型名称
  }

  /**
   * @brief 获取存储对象的常量引用。代码会检查存储对象是否为所需类型。
   */
  template <class T>
  const T& Get() const {
    AT_ASSERTM(  // 类型检查断言
        IsType<T>(),
        "wrong type for the Blob instance. Blob contains ",
        meta_.name(),
        " while caller expects ",
        TypeMeta::TypeName<T>());
    return *static_cast<const T*>(pointer_);  // 类型转换后返回
  }

  // 获取原始指针（无类型检查）
  const void* GetRaw() const noexcept {
    return pointer_;
  }
  void* GetRaw() noexcept {
    return pointer_;
  }

  /**
   * @brief 获取存储对象的可变指针
   *
   * 如果当前对象不是正确类型，会创建新对象并释放旧对象。
   * 注意：类型T必须有默认构造函数。
   */
  template <class T>
  T* GetMutable() {
    static_assert(  // 编译期检查是否可默认构造
        std::is_default_constructible<T>::value,
        "GetMutable can't be called with non-default-constructible types");
    if (IsType<T>()) {
      return static_cast<T*>(pointer_);
    } else {
      return Reset<T>(new T());  // 类型不匹配时重建对象
    }
  }

  // 安全版GetMutable，失败返回nullptr
  template <class T>
  T* GetMutableOrNull() {
    if (IsType<T>()) {
      return static_cast<T*>(pointer_);
    } else {
      return nullptr;
    }
  }

  /**
   * 重置Blob内容为指定指针（接管所有权）
   */
  template <class T>
  T* Reset(T* allocated) {
    free_();  // 先释放旧资源
    meta_ = TypeMeta::Make<T>();  // 更新类型信息
    pointer_ = static_cast<void*>(allocated);
    has_ownership_ = true;  // 标记所有权
    return allocated;
  }

  /**
   * 共享外部指针（不接管所有权）
   */
  template <class T>
  typename std::remove_const<T>::type* ShareExternal(
      typename std::remove_const<T>::type* allocated) {
    return static_cast<T*>(ShareExternal(
        static_cast<void*>(allocated),
        TypeMeta::Make<typename std::remove_const<T>::type>()));
  }

  // 共享外部指针的底层实现
  void* ShareExternal(void* allocated, const TypeMeta& meta) {
    free_();
    meta_ = meta;
    pointer_ = static_cast<void*>(allocated);
    has_ownership_ = false;  // 明确不拥有所有权
    return allocated;
  }

  /**
   * 重置Blob为空状态
   */
  inline void Reset() {
    free_();
    pointer_ = nullptr;
    meta_ = TypeMeta();
    has_ownership_ = false;
  }

  /**
   * @brief 交换两个Blob的底层存储
   */
  void swap(Blob& rhs) {
    using std::swap;
    swap(meta_, rhs.meta_);
    swap(pointer_, rhs.pointer_);
    swap(has_ownership_, rhs.has_ownership_);
  }

 private:
  // 释放资源（仅当拥有所有权时）
  void free_() {
    if (has_ownership_) {
      AT_ASSERTM(pointer_ != nullptr, "Can't have ownership of nullptr");
      (*meta_.deleteFn())(pointer_);  // 调用类型特定的删除器
    }
  }

  TypeMeta meta_;         // 类型元信息
  void* pointer_ = nullptr;  // 存储的指针
  bool has_ownership_ = false;  // 所有权标志

  C10_DISABLE_COPY_AND_ASSIGN(Blob);  // 禁用拷贝构造和赋值
};

// 全局swap函数重载
inline void swap(Blob& lhs, Blob& rhs) {
  lhs.swap(rhs);
}

// 输出运算符重载
inline std::ostream& operator<<(std::ostream& out, const Blob& v) {
  return out << "Blob[" << v.TypeName() << "]";  // 输出类型名称
}

} // namespace caffe2