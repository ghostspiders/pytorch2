#include <ATen/core/VariableHooksInterface.h>  // 变量钩子接口定义

namespace at {
namespace detail {

// 获取变量钩子接口的全局实例
// 注意：
// 1. dlopen()后返回的VariableHooks可能会发生变化
// 2. 此函数会加锁，不要在对性能敏感的路径中调用
const VariableHooksInterface& getVariableHooks() {
    // 静态互斥锁，保证线程安全
    static std::mutex var_hooks_mutex;
    // 实际的变量钩子实现（默认为nullptr）
    static std::unique_ptr<VariableHooksInterface> var_hooks = nullptr;
    // 默认的变量钩子实现（空实现）
    static std::unique_ptr<VariableHooksInterface> default_var_hooks =
        std::unique_ptr<VariableHooksInterface>(new VariableHooksInterface());
    
    // 加锁保护共享数据
    std::lock_guard<std::mutex> lock(var_hooks_mutex);

    // 延迟初始化：第一次调用时创建实际钩子
    if (!var_hooks) {
        // 从注册表中创建具体的VariableHooks实现
        var_hooks = VariableHooksRegistry()->Create("VariableHooks", VariableHooksArgs{});
    }
    
    // 返回有效的钩子实现（优先返回自定义实现）
    if (var_hooks) {
        return *var_hooks;
    }
    // 回退到默认实现
    return *default_var_hooks;
}

} // namespace detail

// 定义变量钩子注册表（使用C10的注册表宏）
// 参数说明：
// 1. VariableHooksRegistry - 注册表名称
// 2. VariableHooksInterface - 接口类型
// 3. VariableHooksArgs - 构造参数类型
C10_DEFINE_REGISTRY(
    VariableHooksRegistry,
    VariableHooksInterface,
    VariableHooksArgs)

} // namespace at