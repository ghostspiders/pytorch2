#include <cstdint>  // 标准整数类型
#include <cstring>  // 字符串操作函数
#include <iostream>  // 输入输出流
#include <mutex>  // 互斥锁(用于线程安全)
#include <sstream>  // 字符串流
#include <string>  // 字符串类
#include <unordered_map>  // 哈希表容器
#include <vector>  // 动态数组容器
#include "ATen/core/interned_strings.h"  // 内部字符串定义
#include "c10/util/Exception.h"  // 异常处理

namespace c10 {

// 内部字符串管理类(线程安全)
// CAFFE2_API宏确保符号的跨平台可见性
struct CAFFE2_API InternedStrings {
  InternedStrings();  // 构造函数
  
  // 将字符串转换为符号(Symbol)
  Symbol symbol(const std::string& s);
  
  // 获取符号对应的字符串(返回命名空间和名称的指针对)
  std::pair<const char*, const char*> string(Symbol sym);
  
  // 获取符号的命名空间
  Symbol ns(Symbol sym);

 private:
  // 内部实现方法(调用前必须持有mutex_锁)
  
  // 实际执行字符串到符号转换的内部方法
  Symbol _symbol(const std::string& s);
  
  // 处理自定义字符串格式的内部方法
  std::pair<const char*, const char*> customString(Symbol sym);
  
  // 字符串到符号的映射表(哈希表)
  std::unordered_map<std::string, Symbol> string_to_sym_;

  // 符号信息结构体
  struct SymbolInfo {
    Symbol ns;  // 命名空间符号
    std::string qual_name;  // 限定名称(包含命名空间)
    std::string unqual_name;  // 非限定名称(不包含命名空间)
  };
  
  // 符号到信息的映射表(动态数组)
  std::vector<SymbolInfo> sym_to_info_;

  // 互斥锁(保证线程安全)
  std::mutex mutex_;
};

} // namespace c10