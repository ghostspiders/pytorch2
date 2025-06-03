#pragma once  // 防止头文件重复包含

#include <ATen/Utils.h>  // ATen工具函数
#include <c10/util/ArrayRef.h>  // 基础数组引用类
#include <vector>  // 标准向量容器

namespace at {
  /// MatrixRef - 类似ArrayRef，但额外记录步长(stride)以便作为多维数组视图
  ///
  /// 与ArrayRef类似，此类不拥有底层数据，适用于数据存在于其他缓冲区的情况
  ///
  /// 设计为可简单复制的类型，应通过值传递
  ///
  /// 当前仅支持2D连续布局(可返回非跨步的ArrayRef)
  ///
  /// 注意：维度0索引行，维度1索引列
  template<typename T>
  class MatrixRef {
  public:
    typedef size_t size_type;  // 尺寸类型定义

  private:
    /// 底层数据引用
    ArrayRef<T> arr;

    /// 外层维度(维度0)的步长
    size_type stride0;

    // 内层维度(维度1)的步长固定为1

  public:
    /// 构造空MatrixRef
    /*implicit*/ MatrixRef() : arr(nullptr), stride0(0) {}

    /// 从ArrayRef和外部步长构造MatrixRef
    /*implicit*/ MatrixRef(ArrayRef<T> arr, size_type stride0)
      : arr(arr), stride0(stride0) {
        // 验证数组大小能被步长整除
        AT_CHECK(arr.size() % stride0 == 0, 
                "MatrixRef: ArrayRef size ", arr.size(), 
                " not divisible by stride ", stride0)
      }

    /// @name 简单操作
    /// @{

    /// 检查矩阵是否为空
    bool empty() const { return arr.empty(); }

    /// 获取底层数据指针
    const T *data() const { return arr.data(); }

    /// 获取指定维度大小
    size_t size(size_t dim) const {
      if (dim == 0) {
        return arr.size() / stride0;  // 行数 = 总元素数 / 列数
      } else if (dim == 1) {
        return stride0;  // 列数 = 步长
      } else {
        AT_CHECK(0, "MatrixRef: out of bounds dimension ", 
                dim, "; expected 0 or 1");
      }
    }

    /// 获取元素总数
    size_t numel() const {
      return arr.size();
    }

    /// 检查元素级相等性
    bool equals(MatrixRef RHS) const {
      return stride0 == RHS.stride0 && arr.equals(RHS.arr);
    }

    /// @}
    /// @name 运算符重载
    /// @{
    
    /// 下标运算符，返回指定行的ArrayRef
    ArrayRef<T> operator[](size_t Index) const {
      return arr.slice(Index*stride0, stride0);
    }

    /// 禁止从临时对象赋值(通过SFINAE实现)
    template <typename U>
    typename std::enable_if<std::is_same<U, T>::value, MatrixRef<T>>::type &
    operator=(U &&Temporary) = delete;

    /// 禁止从初始化列表赋值(通过SFINAE实现)
    template <typename U>
    typename std::enable_if<std::is_same<U, T>::value, MatrixRef<T>>::type &
    operator=(std::initializer_list<U>) = delete;

  };

} // end namespace at