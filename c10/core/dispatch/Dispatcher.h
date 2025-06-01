#pragma once

#include <c10/core/dispatch/DispatchTable.h>

namespace c10 {

/**
 * 动态派发器（Dispatcher）的顶层接口。
 * 用于通过动态派发机制调用算子。
 *
 * @tparam OpSchemaDef 算子模式定义类型，包含算子的签名、名称等信息。
 */
template<class OpSchemaDef>
class Dispatcher final {
public:
  // 实现说明：这个类抽象了"每个算子有独立派发表"的事实，
  // 可以轻松调整为使用单一全局哈希表。

  /**
   * 向某个算子模式的派发表中注册一个内核（实现）。
   *
   * @tparam Args 完美转发参数类型（自动推断）
   * @param args 转发给底层 DispatchTable::registerKernel 的参数
   */
  template<class... Args>
  static void registerKernel(Args&&... args) {
    // 获取该算子对应的派发表
    auto& dispatch_table_for_this_op = c10_dispatch_table<OpSchemaDef>();
    // 转发参数到底层注册方法
    return dispatch_table_for_this_op.registerKernel(std::forward<Args>(args)...);
  }

  /**
   * 从某个算子模式的派发表中注销一个内核。
   *
   * @tparam Args 完美转发参数类型（自动推断）
   * @param args 转发给底层 DispatchTable::deregisterKernel 的参数
   */
  template<class... Args>
  static void deregisterKernel(Args&&... args) {
    // 获取该算子对应的派发表
    auto& dispatch_table_for_this_op = c10_dispatch_table<OpSchemaDef>();
    // 转发参数到底层注销方法
    return dispatch_table_for_this_op.deregisterKernel(std::forward<Args>(args)...);
  }

  /**
   * 执行动态派发调用某个算子。
   *
   * @tparam Args 完美转发参数类型（自动推断）
   * @param args 转发给底层 DispatchTable::call 的参数
   * @return 该算子的返回值类型（通过 OpSchema 的签名信息获取）
   */
  template<class... Args>
  static typename OpSchema<OpSchemaDef>::signature::return_type call(Args&&... args) {
    // 获取该算子对应的派发表
    auto& dispatch_table_for_this_op = c10_dispatch_table<OpSchemaDef>();
    // 转发参数并执行调用
    return dispatch_table_for_this_op.call(std::forward<Args>(args)...);
  }
};

} // namespace c10