#include "ATen/core/interned_strings.h"  // 内部字符串头文件
#include <cstdint>  // 标准整数类型
#include <cstring>  // C字符串操作
#include <iostream>  // 输入输出流
#include <mutex>  // 互斥锁(线程安全)
#include <sstream>  // 字符串流
#include <string>  // 字符串类
#include <unordered_map>  // 哈希表容器
#include <vector>  // 动态数组容器
#include "ATen/core/interned_strings_class.h"  // 内部字符串类定义
#include "c10/util/Exception.h"  // 异常处理
#include "c10/util/Optional.h"  // 可选值容器

namespace c10 {

// 返回域名前缀常量(用于符号的域名表示)
const std::string& domain_prefix() {
  static const std::string _domain_prefix = "org.pytorch.";  // PyTorch组织前缀
  return _domain_prefix;
}

// 将字符串转换为Symbol(线程安全)
Symbol InternedStrings::symbol(const std::string& s) {
  std::lock_guard<std::mutex> guard(mutex_);  // 加锁保证线程安全
  return _symbol(s);  // 调用内部实现
}

// 获取Symbol对应的字符串表示(返回完整限定名和非限定名)
std::pair<const char*, const char*> InternedStrings::string(Symbol sym) {
  // 内置Symbol可以直接返回已知字符串值，无需查表
  switch (sym) {
    // 使用宏处理所有预定义的命名空间符号
#define DEFINE_CASE(ns, s) \
  case static_cast<unique_t>(ns::s): \
    return {#ns "::" #s, #s};  // 返回"命名空间::符号名"和"符号名"
    FORALL_NS_SYMBOLS(DEFINE_CASE)  // 应用宏到所有预定义符号
#undef DEFINE_CASE
    default:
      return customString(sym);  // 自定义符号需要查表
  }
}

// 获取Symbol所属的命名空间
Symbol InternedStrings::ns(Symbol sym) {
  switch (sym) {
    // 处理预定义符号的命名空间
#define DEFINE_CASE(ns, s) \
  case static_cast<unique_t>(ns::s): \
    return namespaces::ns;  // 直接返回预定义的命名空间
    FORALL_NS_SYMBOLS(DEFINE_CASE)
#undef DEFINE_CASE
    default: {
      std::lock_guard<std::mutex> guard(mutex_);  // 加锁
      return sym_to_info_.at(sym).ns;  // 从表中查询自定义符号的命名空间
    }
  }
}

// 内部实现的字符串到Symbol转换
Symbol InternedStrings::_symbol(const std::string& s) {
  // 先在哈希表中查找是否已存在
  auto it = string_to_sym_.find(s);
  if (it != string_to_sym_.end())
    return it->second;

  // 检查字符串格式必须包含"::"
  auto pos = s.find("::");
  if (pos == std::string::npos) {
    std::stringstream ss;
    ss << "所有符号必须有命名空间，格式为<命名空间>::<字符串>，但找到: " << s;
    throw std::runtime_error(ss.str());
  }

  // 注册命名空间Symbol(递归调用)
  Symbol ns = _symbol("namespaces::" + s.substr(0, pos));

  // 创建新Symbol并存入表中
  Symbol sym(sym_to_info_.size());  // 新Symbol的值为当前表大小
  string_to_sym_[s] = sym;  // 字符串到Symbol映射
  sym_to_info_.push_back({ns, s, s.substr(pos + strlen("::"))});  // Symbol详细信息
  return sym;
}

// 处理自定义Symbol的字符串查询
std::pair<const char*, const char*> InternedStrings::customString(Symbol sym) {
  std::lock_guard<std::mutex> guard(mutex_);  // 加锁
  SymbolInfo& s = sym_to_info_.at(sym);  // 查表
  return {s.qual_name.c_str(), s.unqual_name.c_str()};  // 返回C风格字符串
}

// 全局InternedStrings单例
static InternedStrings & globalStrings() {
  static InternedStrings s;  // 静态变量保证唯一实例
  return s;
}

// Symbol类的静态方法: 从限定字符串创建Symbol
Symbol Symbol::fromQualString(const std::string & s) {
  return globalStrings().symbol(s);  // 使用全局实例
}

// 获取非限定名称字符串
const char * Symbol::toUnqualString() const {
  return globalStrings().string(*this).second;  // 返回pair的第二个元素
}

// 获取完整限定名称字符串
const char * Symbol::toQualString() const {
  return globalStrings().string(*this).first;  // 返回pair的第一个元素
}

// 获取显示用的字符串(目前与完整限定名相同)
const char * Symbol::toDisplayString() const {
  // TODO: 未来应返回更用户友好的字符串
  // 由于需要返回生命周期全局的const char*，不能动态组装字符串
  return toQualString();
}

// 获取命名空间Symbol
Symbol Symbol::ns() const {
  return globalStrings().ns(*this);
}

// 获取域名字符串(org.pytorch.<命名空间>)
std::string Symbol::domainString() const {
  return domain_prefix() + ns().toUnqualString();
}

// 从域名和非限定名创建Symbol
Symbol Symbol::fromDomainAndUnqualString(const std::string & d, const std::string & s) {
  // 检查域名前缀是否正确
  if (d.compare(0, domain_prefix().size(), domain_prefix()) != 0) {
    std::ostringstream ss;
    ss << "Symbol: 域名字符串应以前缀'" << domain_prefix() 
       << "'开头，例如'org.pytorch.aten'";
    throw std::runtime_error(ss.str());
  }
  // 组合成限定名字符串并创建Symbol
  std::string qualString = d.substr(domain_prefix().size()) + "::" + s;
  return fromQualString(qualString);
}

} // namespace c10