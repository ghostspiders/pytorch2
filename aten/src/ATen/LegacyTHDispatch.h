#pragma once

// LegacyTHDispatcher 是用于直接分派到 ATen 中 TH/THNN/THC/THCUNN 函数的遗留机制，
// 本质上是一个巨大的虚拟分派表，支持我们动态分派所有 TH 函数。

// 注意：这里实际分派的不是*运算符*，通常模式是 ATen 运算符调用此机制实现功能，
// 但运算符本身是单独声明的（例如作为原生函数"包装器"）。

// 问：为什么不直接使用 LegacyTypeDispatch？
// 答：主要是关注点分离：
//   1) Type 用于运算符实现，需要生成 Variables、JIT 等代码。这由原生函数"包装器"处理；
//      单纯调用 TH 不需要这些
//   2) Type 不需要特定标量分派，而调用 TH 需要。这种分离允许我们独立演进运算符分派
//      （如使用 C10 分派器）与 TH 功能调用细节

// 此实现与 LegacyTypeDispatch 设计非常相似，但有如下简化：
// 1) 移动构建不需要此机制，因此不必放在/core目录
// 2) 因仅包含函数实现，无需处理 Variable/Tensor 拆分（由原生函数"包装器"处理）
// 3) 因运算符必须通过 Type 机制预先分派，我们需要处理设备初始化。这意味着
//     不经过 Type 分派直接调用这些函数是错误行为（标准调用链：运算符 -> Type -> LegacyTHDispatch）
// 4) 因运算符已通过 Type 机制分派，我们无需处理未定义的 Tensors

// 注意：不使用 Registry，因为不想为每次操作付出哈希表查找开销
// 注意：当我们不再调用任何 TH 实现时可以删除此机制


#include <c10/core/Backend.h>  // 后端类型定义(CPU/CUDA等)
#include <c10/core/ScalarType.h>  // 标量类型定义(Float/Int等)
#include <ATen/LegacyTHDispatcher.h>  // 传统TH调度器接口

namespace at {

struct Type;  // 前向声明ATen类型系统

// 传统TH调度器删除器定义
struct CAFFE2_API LegacyTHDispatcherDeleter {
  using LegacyTHDispatcherDeleterFun = void(LegacyTHDispatcher*);  // 删除函数类型定义
  LegacyTHDispatcherDeleterFun *fn_ = nullptr;  // 实际删除函数指针
  
  LegacyTHDispatcherDeleter() {}  // 默认构造函数
  /* implicit */ 
  LegacyTHDispatcherDeleter(LegacyTHDispatcherDeleterFun *fn) : fn_(fn) {}  // 带删除函数的构造函数
  
  // 重载函数调用运算符，用于执行删除操作
  void operator()(LegacyTHDispatcher * ptr) {
    if (fn_) {
      (*fn_)(ptr);  // 调用注册的删除函数
    }
  }
};

// 传统TH调度系统主类
class CAFFE2_API LegacyTHDispatch {
 public:
  // 定义带有自定义删除器的unique_ptr类型
  using LegacyTHDispatcherUniquePtr = std::unique_ptr<LegacyTHDispatcher, LegacyTHDispatcherDeleter>;
  
  // 注册调度器到指定后端和标量类型
  // 注意: 调用前需确保对应类型已初始化
  void registerDispatcher(Backend b, ScalarType s, LegacyTHDispatcherUniquePtr&& t) {
    dispatcher_registry[static_cast<int>(b)][static_cast<int>(s)] = std::move(t);
  }

  // 获取原始调度器指针(不检查nullptr)
  LegacyTHDispatcher* getLegacyTHDispatcherRaw(Backend p, ScalarType s) {
    return dispatcher_registry[static_cast<int>(p)][static_cast<int>(s)].get();
  }

  // 获取调度器引用(带nullptr检查)
  LegacyTHDispatcher & getLegacyTHDispatcher(Backend p, ScalarType s) {
    auto* type = getLegacyTHDispatcherRaw(p, s);
    if (!type) AT_ERROR(toString(p), toString(s), "THDispatcher is not enabled.");
    return *type;
  }
  
private:
  // 调度器注册表(二维数组)
  // 注意: CUDA后端在CUDA初始化前都为nullptr
  LegacyTHDispatcherUniquePtr dispatcher_registry
    [static_cast<int>(Backend::NumOptions)]  // 后端数量维度
    [static_cast<int>(ScalarType::NumOptions)];  // 标量类型数量维度
};

// 获取全局传统TH调度系统单例
CAFFE2_API LegacyTHDispatch& globalLegacyTHDispatch();

}  // namespace at