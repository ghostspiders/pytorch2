#pragma once

namespace at {

// 前向声明Tensor类
class Tensor;

/**
 * @struct SparseTensorRef
 * @brief 稀疏张量引用包装器
 * 
 * 该结构体用于持有对稀疏Tensor的常量引用，
 * 主要用于函数参数传递，避免不必要的拷贝。
 */
struct SparseTensorRef {
  /**
   * @brief 显式构造函数
   * @param t 要引用的Tensor对象
   * @note 使用explicit避免隐式转换
   */
  explicit SparseTensorRef(const Tensor& t): tref(t) {}
  
  const Tensor& tref;  ///< 持有的Tensor常量引用
};

} // namespace at