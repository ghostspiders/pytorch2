#pragma once
#include <unordered_set>
#include <vector>
#include <ATen/core/interned_strings.h>
#include "c10/util/Exception.h"

namespace c10 {

// 表示别名信息的类，用于跟踪哪些内存位置可能相互重叠
class AliasInfo {
 public:
  // 获取通配符集合的符号（表示可能指向任何内存位置）
  static Symbol wildcardSet() {
    static const Symbol wc = Symbol::fromQualString("alias::*");
    return wc;
  }

  // 创建一个通配符别名信息
  static AliasInfo createWildcard() {
    AliasInfo ret;
    ret.addSet(wildcardSet());
    return ret;
  }

  // 设置该别名信息是否表示写入操作
  void setIsWrite(bool isWrite) {
    isWrite_ = isWrite;
  }

  // 检查是否表示写入操作
  bool isWrite() const {
    return isWrite_;
  }

  // 添加一个新的别名集合符号
  void addSet(Symbol aliasSet) {
    sets_.insert(aliasSet);
  }

  // 获取所有别名集合的引用
  const std::unordered_set<Symbol>& sets() const {
    return sets_;
  }

  // 获取单个别名集合（仅在集合大小为1时有效）
  Symbol set() const {
    AT_ASSERT(sets_.size() == 1);
    return *sets_.begin();
  }

  // 检查是否是通配符（包含通配符集合）
  bool isWildcard() const {
    return sets_.count(wildcardSet()) != 0;
  }

  // 与另一个别名信息合并（集合并集）
  void unionWith(const AliasInfo& other) {
    for (const auto& alias : other.sets()) {
      sets_.insert(alias);
    }
  }

  // 检查当前别名信息是否是另一个的子集（非严格检查）
  bool isSubsetOf(const AliasInfo& other) const {
    for (const auto& alias : this->sets()) {
      if (other.sets().count(alias) == 0) {
        return false;
      }
    }
    return true;
  }

  // 添加包含的类型别名信息（用于容器元素类型）
  void addContainedType(AliasInfo aliasInfo) {
    containedTypes_.push_back(std::move(aliasInfo));
  }

  // 获取所有包含的类型别名信息
  const std::vector<AliasInfo>& containedTypes() const {
    return containedTypes_;
  }

 private:
  std::unordered_set<Symbol> sets_;       // 存储别名集合的符号
  std::vector<AliasInfo> containedTypes_; // 容器内元素的别名信息
  bool isWrite_ = false;                  // 标记是否涉及写入操作
};

// 调试专用：输出别名信息的可读格式（不匹配模式中的实际表示）
inline std::ostream& operator<<(std::ostream& out, const AliasInfo& aliasInfo) {
  out << "(";
  bool first = true;
  for (const auto& set : aliasInfo.sets()) {
    if (first) {
      first = false;
    } else {
      out << "|";
    }
    out << set.toUnqualString();  // 输出非限定字符串表示
  }
  out << ")";

  // 如果包含子类型信息，输出首个子类型
  if (!aliasInfo.containedTypes().empty()) {
    out << " CONTAINS " << aliasInfo.containedTypes()[0];
  }
  return out;
}
} // namespace c10