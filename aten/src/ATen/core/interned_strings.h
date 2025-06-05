#pragma once  // 防止头文件重复包含

#include <vector>  // 向量容器
#include <cstdint> // 标准整数类型
#include <string>  // 字符串类
#include <unordered_map>  // 哈希表容器
#include <algorithm>  // 算法库

#include <ATen/core/aten_interned_strings.h>  // ATen内部字符串定义
#include <c10/macros/Macros.h>  // 宏定义

namespace c10 {

// 移动端和非移动端使用不同的符号定义集合
#if !C10_MOBILE
// 定义所有命名空间和符号的宏
#define FORALL_NS_SYMBOLS(_)       \
  _(namespaces, prim)              \  // 基本操作命名空间
  _(namespaces, aten)              \  // ATen核心命名空间
  _(namespaces, onnx)              \  // ONNX相关命名空间
  _(namespaces, attr)              \  // 属性相关命名空间
  _(namespaces, scope)             \  // 作用域命名空间
  _(namespaces, namespaces)        \  // 命名空间本身
  // 各种prim(基本)操作符号
  _(prim, Assign)                  \
  _(prim, BroadcastingChunk)       \
  _(prim, BroadcastSizes)          \
  // ...(其他prim符号省略)
  // ATen操作符号
  _(aten, warn)                    \
  _(aten, floordiv)                \
  // ...(其他aten符号省略)
  // ONNX操作符号
  _(onnx, Add)                     \
  _(onnx, Concat)                  \
  // ...(其他onnx符号省略)
  // 属性符号
  _(attr, Subgraph)                \
  _(attr, ReverseSubgraph)         \
  // ...(其他attr符号省略)
  FORALL_ATEN_BASE_SYMBOLS(_)      \  // 包含ATen基础符号
  FORALL_ATTR_BASE_SYMBOLS(_)       // 包含属性基础符号
#else
// 移动端只包含最基础的命名空间定义
#define FORALL_NS_SYMBOLS(_) \
  _(namespaces, prim)        \
  _(namespaces, aten)        \
  _(namespaces, onnx)        \
  _(namespaces, attr)        \
  _(namespaces, scope)       \
  _(namespaces, namespaces)
#endif

using unique_t = uint32_t;  // 符号唯一标识类型

const std::string& domain_prefix();  // 获取域名前缀

// Symbol类定义
struct CAFFE2_API Symbol {
  explicit constexpr Symbol() : value(0) {};  // 默认构造函数
  explicit constexpr Symbol(unique_t uniq) : value(uniq) {}  // 从唯一值构造

  // 静态工厂方法
  static Symbol fromQualString(const std::string & s);  // 从限定字符串创建
  static Symbol fromDomainAndUnqualString(const std::string & d, const std::string & s);  // 从域名和非限定名创建

  // 命名空间便捷构造方法
  static Symbol attr(const std::string & s);  // 创建属性符号
  static Symbol aten(const std::string & s);  // 创建ATen符号
  static Symbol onnx(const std::string & s);  // 创建ONNX符号
  static Symbol prim(const std::string & s);  // 创建基本操作符号
  static Symbol scope(const std::string & s);  // 创建作用域符号(待废弃)

  // 命名空间检查方法
  bool is_attr() const;  // 是否属性符号
  bool is_aten() const;  // 是否ATen符号
  bool is_prim() const;  // 是否基本操作符号
  bool is_onnx() const;  // 是否ONNX符号

  // 转换为唯一值(可用于switch)
  constexpr operator unique_t() const {
    return value;
  }

  // 获取命名空间
  Symbol ns() const;

  // 字符串表示方法
  const char * toUnqualString() const;  // 获取非限定名
  const char * toQualString() const;  // 获取限定名(含命名空间)
  const char * toDisplayString() const;  // 获取显示用字符串
  std::string domainString() const;  // 获取域名字符串

private:
  explicit Symbol(Symbol ns, const std::string & s);  // 私有构造方法
  unique_t value;  // 符号唯一值
};

// Symbol相等比较运算符
static inline bool operator==(Symbol lhs, Symbol rhs) {
  return static_cast<unique_t>(lhs) == static_cast<unique_t>(rhs);
}

// 符号枚举定义
enum class _keys : unique_t {
    #define DEFINE_KEY(ns, s) ns##_##s,  // 定义枚举值
    FORALL_NS_SYMBOLS(DEFINE_KEY)  // 应用宏
    #undef DEFINE_KEY
    num_symbols  // 符号总数
};

// 定义符号常量(第一轮)
#define DEFINE_SYMBOL(s) \
  constexpr Symbol s(static_cast<unique_t>(_keys::s));
// (这里实际没有定义任何符号，可能被注释掉了)
#undef DEFINE_SYMBOL

// 定义符号常量(第二轮，按命名空间组织)
#define DEFINE_SYMBOL(ns, s) \
  namespace ns { constexpr Symbol s(static_cast<unique_t>(_keys::ns##_##s)); }
FORALL_NS_SYMBOLS(DEFINE_SYMBOL)  // 为所有符号生成定义
#undef DEFINE_SYMBOL

// Symbol类内联方法实现
inline Symbol Symbol::attr(const std::string & s) { return Symbol::fromQualString("attr::" + s); }
inline Symbol Symbol::aten(const std::string & s)  { return Symbol::fromQualString("aten::" + s); }
inline Symbol Symbol::onnx(const std::string & s)  { return Symbol::fromQualString("onnx::" + s); }
inline Symbol Symbol::prim(const std::string & s)  { return Symbol::fromQualString("prim::" + s); }
inline Symbol Symbol::scope(const std::string & s) { return Symbol::fromQualString("scope::" + s); }
inline bool Symbol::is_attr() const { return ns() == namespaces::attr; }
inline bool Symbol::is_aten() const { return ns() == namespaces::aten; }
inline bool Symbol::is_prim() const { return ns() == namespaces::prim; }
inline bool Symbol::is_onnx() const { return ns() == namespaces::onnx; }

} // namespace c10

// 使Symbol可被用作哈希表键
namespace std {
template <>
struct hash<c10::Symbol> {
  size_t operator()(c10::Symbol s) const {
    return std::hash<uint32_t>()(static_cast<uint32_t>(s));  // 直接哈希其整数值
  }
};
}