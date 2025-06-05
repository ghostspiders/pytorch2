#include <ATen/core/ivalue.h>  // IValue基础定义
#include <ATen/core/Formatting.h>  // 格式化工具
#include <cmath>  // 数学函数

namespace c10 {
namespace ivalue {

// 创建常量字符串的智能指针
CAFFE2_API c10::intrusive_ptr<ConstantString> ConstantString::create(
    std::string str_) {
  return c10::make_intrusive<ConstantString>(std::move(str_));
}

} // namespace ivalue

namespace {  // 匿名命名空间

// 通用列表打印模板函数
template<typename List>
std::ostream& printList(std::ostream & out, const List &v,
  const std::string start, const std::string finish) {
  out << start;  // 输出起始符号（如"[")
  for(size_t i = 0; i < v->elements().size(); ++i) {
    if(i > 0)
      out << ", ";  // 元素间分隔符
    // 强制使用IValue的打印方式
    out << IValue(v->elements()[i]);
  }
  out << finish;  // 输出结束符号（如"]"）
  return out;
}

} // anonymous namespace

// IValue流输出运算符重载
std::ostream& operator<<(std::ostream & out, const IValue & v) {
  switch(v.tag) {  // 根据类型标签处理不同数据类型
    case IValue::Tag::None:  // 空值
      return out << v.toNone();
    case IValue::Tag::Tensor:  // 张量
      return out << v.toTensor();
    case IValue::Tag::Double: {  // 双精度浮点数
      double d = v.toDouble();
      int c = std::fpclassify(d);  // 检查浮点类型
      if (c == FP_NORMAL || c == FP_ZERO) {  // 常规数或零
        int64_t i = int64_t(d);
        if (double(i) == d) {  // 整数形式输出
          return out << i << ".";
        }
      }
      // 高精度输出浮点数
      auto orig_prec = out.precision();
      return out
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << v.toDouble()
        << std::setprecision(orig_prec);
    } 
    case IValue::Tag::Int:  // 整型
      return out << v.toInt();
    case IValue::Tag::Bool:  // 布尔型
      return out << (v.toBool() ? "True" : "False");
    case IValue::Tag::Tuple:  // 元组
      return printList(out, v.toTuple(), "(", ")");
    case IValue::Tag::IntList:  // 整型列表
      return printList(out, v.toIntList(), "[", "]");
    // ...其他类型处理（省略类似代码）
    default:
      AT_ERROR("Tag not found\n");  // 未知类型错误
  }
}

#undef TORCH_FORALL_TAGS  // 清理宏定义

} // namespace c10
