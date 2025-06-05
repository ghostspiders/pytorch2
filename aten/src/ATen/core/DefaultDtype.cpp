#include <ATen/core/typeid.h>      // 类型ID相关功能头文件
#include <ATen/core/DefaultDtype.h> // 默认数据类型相关头文件

namespace at {  // ATen核心命名空间

// 全局默认数据类型变量，初始化为float类型
// 使用TypeMeta::Make<float>()创建float类型的元信息对象
static auto default_dtype = caffe2::TypeMeta::Make<float>();

/**
 * 设置全局默认数据类型
 * @param dtype 要设置的TypeMeta对象，包含类型信息
 * @note 使用std::move进行移动语义优化
 */
void set_default_dtype(caffe2::TypeMeta dtype) {
  default_dtype = std::move(dtype);  // 移动赋值，避免不必要的拷贝
}

/**
 * 获取当前全局默认数据类型
 * @return 返回当前默认数据类型的常量引用
 */
const caffe2::TypeMeta& get_default_dtype() {
  return default_dtype;  // 返回常量引用，避免拷贝
}

} // namespace at