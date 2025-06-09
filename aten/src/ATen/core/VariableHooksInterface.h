#pragma once  // 防止头文件重复包含

#include <c10/core/Backend.h>      // 后端类型定义(CPU/CUDA等)
#include <c10/core/ScalarType.h>   // 标量类型定义(float/int等)
#include <c10/util/Registry.h>     // 注册表工具类

// 前置声明
namespace at {
  class LegacyTypeDispatch;  // 传统类型分发器
  struct Type;              // 类型系统基类
}

// 注意：由于Registry.h的限制，注册表类实际不在detail命名空间
namespace at {

// VariableHooks接口说明：
// 1. 为自动微分(autograd)功能提供接口，当前未包含在libATen.so中
// 2. 主要用于变量类型注册系统，支持延迟初始化CUDA类型时添加额外变量类型
// 3. 如果未来将autograd整合到ATen中，此接口可能废弃
struct CAFFE2_API VariableHooksInterface {
  // 虚析构函数（仅为消除-Werror=non-virtual-dtor警告）
  virtual ~VariableHooksInterface() {} 

  // 从基础类型获取对应的变量类型（未加载libtorch时抛出错误）
  virtual Type& getVariableTypeFromBaseType(const at::Type& baseType) const {
    AT_ERROR("cannot getVariableTypeFromBaseType without libtorch");
  }

  // 为指定后端/标量类型注册变量类型（默认无操作）
  virtual void registerVariableTypeFor(
      LegacyTypeDispatch*, 
      Backend backend, 
      ScalarType scalar_type) const {
    // 当Variable不可用时无操作，后续可能由libtorch.so处理
  }
};

// 注：虚拟参数，用于满足变参宏至少需要一个参数的要求
struct CAFFE2_API VariableHooksArgs {}; 

// 声明变量钩子注册表（使用C10注册表系统）
// 参数说明：
// 1. VariableHooksRegistry - 注册表名称
// 2. VariableHooksInterface - 接口类型
// 3. VariableHooksArgs - 构造参数类型
C10_DECLARE_REGISTRY(
    VariableHooksRegistry,
    VariableHooksInterface,
    VariableHooksArgs);

// 注册变量钩子的快捷宏
#define REGISTER_VARIABLE_HOOKS(clsname) \
  C10_REGISTER_CLASS(VariableHooksRegistry, clsname, clsname)

namespace detail {
// 获取全局变量钩子接口的引用
CAFFE2_API const VariableHooksInterface& getVariableHooks();
}

} // namespace at