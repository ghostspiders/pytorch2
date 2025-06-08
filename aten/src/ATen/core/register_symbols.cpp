#include "ATen/core/interned_strings_class.h"

// 本文件使用-O0优化级别编译，因为宏完全展开后的函数体积很大，
// 且仅在启动时调用一次，不需要优化编译

namespace c10 {

/**
 * @class InternedStrings
 * @brief 内部字符串管理类，用于高效处理符号字符串
 * 
 * 该构造函数初始化符号表，建立符号名与符号ID的双向映射关系
 */
InternedStrings::InternedStrings()
    : sym_to_info_(static_cast<size_t>(_keys::num_symbols)) {  // 预分配符号信息存储空间
  /**
   * @macro REGISTER_SYMBOL
   * @brief 注册符号的宏定义
   * @param n 命名空间名称
   * @param s 符号名称
   * 
   * 该宏执行以下操作：
   * 1. 将"命名空间::符号名"映射到符号ID
   * 2. 将符号ID映射到符号信息结构体(包含命名空间、全名和简称)
   */
  #define REGISTER_SYMBOL(n, s)        \
    string_to_sym_[#n "::" #s] = n::s; \  // 字符串到符号ID的映射
    sym_to_info_[n::s] = {namespaces::n, #n "::" #s, #s};  // 符号ID到信息的映射

  // 遍历所有命名空间和符号进行注册
  FORALL_NS_SYMBOLS(REGISTER_SYMBOL)
  
  #undef REGISTER_SYMBOL  // 注册完成后取消宏定义
}

} // namespace c10
