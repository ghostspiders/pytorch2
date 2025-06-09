#pragma once  // 防止头文件重复包含

#include "ATen/core/TensorImpl.h"  // 基础张量实现类

namespace at {

// 未定义张量实现类（final表示不可被继承）
// CAFFE2_API 宏用于处理库的导入导出符号
struct CAFFE2_API UndefinedTensorImpl final : public TensorImpl {
 public:
  // 单例访问方法
  // 注：Windows平台特殊处理（因MSVC编译器限制）
  // 避免设备代码中出现"at::UndefinedTensorImpl::_singleton未定义"错误
  // 在非Windows平台使用constexpr优化
#ifdef _WIN32
  static inline TensorImpl * singleton() {
#else
  static constexpr inline TensorImpl * singleton() {
#endif
    return &_singleton;  // 返回单例实例指针
  }

  // 以下为必须重写的TensorImpl虚函数
  IntList sizes() const override;        // 获取所有维度大小
  IntList strides() const override;      // 获取所有维度步长
  int64_t size(int64_t d) const override;      // 获取指定维度大小
  int64_t stride(int64_t d) const override;    // 获取指定维度步长
  int64_t dim() const override;          // 获取维度数量
  const Storage& storage() const override;     // 获取存储对象
  int64_t storage_offset() const override;    // 获取存储偏移量

 private:
  UndefinedTensorImpl();  // 私有构造函数（单例模式）
  static UndefinedTensorImpl _singleton;  // 全局唯一实例

 public:
  friend struct UndefinedType;  // 友元声明，允许UndefinedType访问私有成员
};

} // namespace at