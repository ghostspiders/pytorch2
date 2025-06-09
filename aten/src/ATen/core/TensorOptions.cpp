#include <ATen/core/TensorOptions.h>  // 包含TensorOptions头文件

#include <c10/Device.h>               // 包含设备类型定义
#include <c10/core/Layout.h>          // 包含布局类型定义
#include <c10/core/ScalarType.h>      // 包含标量类型定义
#include <c10/util/Optional.h>        // 包含可选值工具

#include <iostream>                   // 标准输入输出流

namespace at {  // ATen库的命名空间

// 重载<<运算符用于输出TensorOptions对象
std::ostream& operator<<(
    std::ostream& stream,             // 输出流
    const TensorOptions& options) {   // 要输出的TensorOptions对象
  return stream << "TensorOptions(dtype=" << options.dtype()       // 输出数据类型
                << ", device=" << options.device()                 // 输出设备信息
                << ", layout=" << options.layout()                 // 输出布局信息
                << ", requires_grad=" << std::boolalpha             // 输出是否需要梯度
                << options.requires_grad() << ")";                 // (使用boolalpha使bool值输出为true/false)
}

} // namespace at