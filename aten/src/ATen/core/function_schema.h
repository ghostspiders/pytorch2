#pragma once  // 防止头文件重复包含

#include <ATen/core/jit_type.h>        // 包含JIT类型系统相关定义
#include <ATen/core/interned_strings.h> // 包含内部字符串定义
#include <ATen/core/ivalue.h>          // 包含IValue(通用值容器)定义
#include <ATen/core/alias_info.h>      // 包含别名信息定义

namespace c10 {

// 函数参数结构体，用于编译器解析函数调用和报错
// 这些对象应该从C10 schema构造(当它们可用时)
struct Argument {
  // 构造函数，参数都有默认值
  Argument(
      std::string name = "",                     // 参数名
      TypePtr type = nullptr,                   // 参数类型
      c10::optional<int32_t> N = c10::nullopt,  // 列表类型的静态长度(如int[3]中的3)
      c10::optional<IValue> default_value = c10::nullopt, // 默认值
      bool kwarg_only = false,                  // 是否只能作为关键字参数
      c10::optional<AliasInfo> alias_info = c10::nullopt) // 别名信息
      : name_(std::move(name)),
        type_(type ? type : DynamicType::get()),  // 如果没提供类型则使用动态类型
        N_(std::move(N)),
        default_value_(std::move(default_value)),
        kwarg_only_(kwarg_only),
        alias_info_(std::move(alias_info)) {
          // 检查如果默认值是Tensor，则必须是未定义或者是变量
          if (default_value_ && default_value_->isTensor()) {
            auto t = default_value_->toTensor();
            AT_ASSERT(!t.defined() || t.is_variable());
          }
        }

  // 各种访问方法
  const std::string& name() const { return name_; }
  TypePtr type() const { return type_; }
  c10::optional<int32_t> N() const { return N_; }
  c10::optional<IValue> default_value() const { return default_value_; }
  bool kwarg_only() const { return kwarg_only_; }
  const c10::optional<AliasInfo>& alias_info() const { return alias_info_; }

private:
  std::string name_;                // 参数名称
  TypePtr type_;                    // 参数类型
  c10::optional<int32_t> N_;        // 列表类型的静态长度(可选)
  c10::optional<IValue> default_value_; // 默认值(可选)
  bool kwarg_only_;                 // 是否仅限关键字参数
  c10::optional<AliasInfo> alias_info_; // 别名信息(可选)
};

// 函数模式(schema)结构体，描述函数签名
struct FunctionSchema {
  // 构造函数1: 使用字符串名称
  FunctionSchema(
      std::string name,                  // 函数名
      std::vector<Argument> arguments,   // 参数列表
      std::vector<Argument> returns,     // 返回值列表
      bool is_vararg = false,            // 是否接受可变参数
      bool is_varret = false)            // 是否有可变返回值
      : name_(std::move(name)),
        arguments_(std::move(arguments)),
        returns_(std::move(returns)),
        is_vararg_(is_vararg),
        is_varret_(is_varret) {}

  // 构造函数2: 使用Symbol名称
  FunctionSchema(
      Symbol name,                       // 函数名(Symbol类型)
      std::vector<Argument> arguments,   // 参数列表
      std::vector<Argument> returns,     // 返回值列表
      bool is_vararg = false,            // 是否接受可变参数
      bool is_varret = false,            // 是否有可变返回值
      std::vector<std::string> writes = {}) // 写入列表(未使用)
      : FunctionSchema(                   // 委托给第一个构造函数
            name.toQualString(),          // 将Symbol转换为限定字符串
            std::move(arguments),
            std::move(returns),
            is_vararg,
            is_varret) {}

private:
  const std::string name_;               // 函数名称
  const std::vector<Argument> arguments_; // 参数列表
  const std::vector<Argument> returns_;  // 返回值列表
  const bool is_vararg_;                 // 是否接受可变数量参数
  const bool is_varret_;                 // 是否有可变数量返回值

public:
  // 各种访问方法
  const std::string& name() const { return name_; }
  const std::vector<Argument>& arguments() const { return arguments_; }
  const std::vector<Argument>& returns() const { return returns_; }
  bool is_vararg() const { return is_vararg_; }
  bool is_varret() const { return is_varret_; }
  
  // 检查函数是否有可变参数(通过检查参数别名信息)
  bool is_mutable() const {
    return std::any_of(
        arguments_.cbegin(), arguments_.cend(), [](const Argument& arg) {
          const auto& aliasInfo = arg.alias_info();
          return aliasInfo && aliasInfo.value().isWrite();
        });
  }
  
  // 根据参数名查找参数索引
  c10::optional<int> argumentIndexWithName(const std::string& name) const {
    for(size_t i = 0; i < arguments().size(); ++i) {
      if(name == arguments()[i].name())
        return i;
    }
    return c10::nullopt;  // 未找到返回空
  }
};

// Argument的输出流操作符重载，用于调试
inline std::ostream& operator<<(std::ostream& out, const Argument& arg) {
  return out << arg.type()->str() << " " << arg.name() 
             << (arg.default_value() ? "=<default>" : "");
}

// FunctionSchema的输出流操作符重载，用于调试
inline std::ostream& operator<<(std::ostream& out, const FunctionSchema& schema) {
  // 最终这应该看起来几乎和python参数解析器相同
  // 但现在直接在这个schema上工作更简单

  out << schema.name();  // 输出函数名
  out << "(";           // 开始参数列表

  bool seen_kwarg_only = false;
  for(size_t i = 0; i < schema.arguments().size(); ++i) {
    if (i > 0) out << ", ";  // 参数间用逗号分隔
    // 处理关键字参数标记(*)
    if (schema.arguments()[i].kwarg_only() && !seen_kwarg_only) {
      out << "*, ";
      seen_kwarg_only = true;
    }
    out << schema.arguments()[i];  // 输出参数
  }

  // 处理可变参数
  if(schema.is_vararg()) {
    if(schema.arguments().size() > 0)
      out << ", ";
    out << "...";
  }

  out << ") -> ";  // 开始返回值部分

  // 处理返回值
  if (schema.returns().size() == 1) {
    out << schema.returns().at(0).type()->str();  // 单个返回值
  } else if (schema.returns().size() > 1) {
    out << "(";  // 多个返回值用括号括起来
    for (size_t i = 0; i < schema.returns().size(); ++i) {
      if (i > 0) out << ", ";
      out << schema.returns()[i].type()->str();
    }
    out << ")";
  }
  return out;
}

} // namespace c10