#pragma once  // 防止头文件重复包含

#include <vector>            // 标准向量容器
#include <c10/util/ArrayRef.h> // PyTorch的数组引用类

namespace c10 {

// 注意事项：传入的函数必须以T的值(T)或常量引用(const T&)形式接收参数
// 如果以非常量引用(T&)形式接收参数会导致编译错误，如：
//    error: no type named 'type' in 'class std::result_of<foobar::__lambda(T)>'
// 不需要显式指定模板参数

// 版本1: 对显式函数和ArrayRef的fmap重载
// 功能：将一个函数应用于输入集合的每个元素，返回结果向量
template<typename F, typename T>
inline auto fmap(const T& inputs, const F& fn) -> std::vector<decltype(fn(*inputs.begin()))> {
  // 创建结果向量，类型由fn的返回类型决定
  std::vector<decltype(fn(*inputs.begin()))> r;
  // 预分配空间以提高效率
  r.reserve(inputs.size());
  // 对每个输入元素应用函数fn
  for(const auto & input : inputs)
    r.push_back(fn(input));
  return r;
}

// 版本2: 对非常量输入的fmap重载
template<typename F, typename T>
inline auto fmap(T& inputs, const F& fn) -> std::vector<decltype(fn(*inputs.begin()))> {
  std::vector<decltype(fn(*inputs.begin()))> r;
  r.reserve(inputs.size());
  for(auto & input : inputs)
    r.push_back(fn(input));
  return r;
}

// 版本3: 针对构造函数的特殊fmap重载(因为C++禁止获取构造函数的地址)
// 功能：使用输入集合的每个元素构造类型R的新对象，返回结果向量
template<typename R, typename T>
inline std::vector<R> fmap(const T& inputs) {
  std::vector<R> r;
  r.reserve(inputs.size());
  // 对每个输入元素调用R的构造函数
  for(auto & input : inputs)
    r.push_back(R(input));
  return r;
}

// 版本1: 对ArrayRef的filter函数
// 功能：过滤输入集合，保留满足条件的元素
template<typename F, typename T>
inline std::vector<T> filter(at::ArrayRef<T> inputs, const F& fn) {
  std::vector<T> r;
  r.reserve(inputs.size()); // 预分配空间(可能不完全使用)
  for(auto & input : inputs) {
    if (fn(input)) {  // 如果元素满足条件
      r.push_back(input); // 添加到结果中
    }
  }
  return r;
}

// 版本2: 对vector的filter函数，转换为ArrayRef后调用版本1
template<typename F, typename T>
inline std::vector<T> filter(const std::vector<T>& inputs, const F& fn) {
  return filter<F, T>(static_cast<at::ArrayRef<T>>(inputs), fn);
}

} // namespace c10