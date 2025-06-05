#include "ATen/core/Formatting.h"  // 格式化输出相关头文件
#include <cmath>                  // 数学函数
#include <cstdint>               // 标准整数类型
#include <iomanip>               // IO流格式控制
#include <iostream>              // 标准输入输出
#include <sstream>               // 字符串流
#include <tuple>                 // 多元组

namespace c10 {
// Backend类型的输出运算符重载
std::ostream& operator<<(std::ostream & out, Backend b) {
  return out << toString(b);  // 调用toString转换为字符串输出
}
} // namespace c10

namespace at {

// 自定义defaultfloat实现(兼容非标准C++编译器)
inline std::ios_base& defaultfloat(std::ios_base& __base) {
  __base.unsetf(std::ios_base::floatfield);  // 清除浮点格式标志
  return __base;
}

// RAII格式保护器(进入作用域保存格式，离开恢复格式)
struct FormatGuard {
  FormatGuard(std::ostream & out) : out(out), saved(nullptr) {
    saved.copyfmt(out);  // 保存当前格式
  }
  ~FormatGuard() {
    out.copyfmt(saved);  // 恢复保存的格式
  }
private:
  std::ostream & out;  // 输出流引用
  std::ios saved;     // 保存的格式状态
};

// Type类型的输出运算符重载
std::ostream& operator<<(std::ostream & out, const Type& t) {
  return out << t.toString();  // 调用类型的toString方法
}

// 内部函数: 确定张量的最佳打印格式
static std::tuple<double, int64_t> __printFormat(std::ostream& stream, const Tensor& self) {
  auto size = self.numel();
  if(size == 0) {
    return std::make_tuple(1., 0);  // 空张量默认格式
  }

  // 检查是否为整数模式
  bool intMode = true;
  auto self_p = self.data<double>();
  for(int64_t i = 0; i < size; i++) {
    auto z = self_p[i];
    if(std::isfinite(z) && z != std::ceil(z)) {
      intMode = false;
      break;
    }
  }

  // 找到第一个有限值作为基准
  int64_t offset = 0;
  while(offset < size && !std::isfinite(self_p[offset])) {
    offset++;
  }

  // 计算数值范围
  double expMin = 1, expMax = 1;
  if(offset < size) {
    expMin = expMax = fabs(self_p[offset]);
    for(int64_t i = offset; i < size; i++) {
      double z = fabs(self_p[i]);
      if(std::isfinite(z)) {
        expMin = std::min(expMin, z);
        expMax = std::max(expMax, z);
      }
    }
    // 计算10的幂次范围
    expMin = (expMin != 0) ? std::floor(std::log10(expMin)) + 1 : 1;
    expMax = (expMax != 0) ? std::floor(std::log10(expMax)) + 1 : 1;
  }

  // 确定输出格式和缩放因子
  double scale = 1;
  int64_t sz;
  if(intMode) {
    sz = (expMax > 9) ? 11 : expMax + 1;
    stream << ((expMax > 9) ? std::scientific : defaultfloat);
  } else {
    if(expMax-expMin > 4) {  // 大动态范围使用科学计数法
      sz = 11 + (std::fabs(expMax) > 99 || std::fabs(expMin) > 99);
      stream << std::scientific;
    } else {  // 小范围使用固定小数
      sz = (expMax > 5 || expMax < 0) ? 7 : (expMax == 0) ? 7 : expMax+6;
      scale = (expMax > 5 || expMax < 0) ? std::pow(10, expMax-1) : 1;
      stream << std::fixed;
    }
    stream << std::setprecision(4);  // 统一4位精度
  }
  return std::make_tuple(scale, sz);
}

// 打印缩进
static void __printIndent(std::ostream &stream, int64_t indent) {
  stream << std::string(indent, ' '); 
}

// 打印缩放因子
static void printScale(std::ostream & stream, double scale) {
  FormatGuard guard(stream);  // 保护格式状态
  stream << defaultfloat << scale << " *" << std::endl;
}

// 打印矩阵(2D张量)
static void __printMatrix(std::ostream& stream, const Tensor& self, int64_t linesize, int64_t indent) {
  double scale; int64_t sz;
  std::tie(scale, sz) = __printFormat(stream, self);

  __printIndent(stream, indent);
  int64_t nColumnPerLine = (linesize-indent)/(sz+1);  // 计算每行列数
  
  // 分块打印列
  for(int64_t firstCol = 0; firstCol < self.size(1); ) {
    int64_t lastCol = std::min(firstCol + nColumnPerLine - 1, self.size(1)-1);
    
    if(nColumnPerLine < self.size(1)) {  // 多列需要分块
      if(firstCol != 0) stream << std::endl;
      stream << "Columns " << firstCol+1 << " to " << lastCol+1 << std::endl;
      __printIndent(stream, indent);
    }
    
    if(scale != 1) {  // 打印缩放因子
      printScale(stream, scale);
      __printIndent(stream, indent);
    }
    
    // 逐行打印
    for(int64_t row = 0; row < self.size(0); row++) {
      double *row_ptr = self.select(0,row).data<double>();
      for(int64_t col = firstCol; col <= lastCol; col++) {
        stream << std::setw(sz) << row_ptr[col]/scale 
               << ((col == lastCol) ? "\n" : " ");
      }
      if(row != self.size(0)-1) {
        __printIndent(stream, (scale != 1) ? indent+1 : indent);
      }
    }
    firstCol = lastCol + 1;
  }
}

// 打印高维张量(递归处理)
static void __printTensor(std::ostream& stream, Tensor& self, int64_t linesize) {
  std::vector<int64_t> counter(self.ndimension()-2, 0);
  counter[0] = -1;  // 初始化计数器
  
  while(true) {
    // 更新多维计数器
    for(int64_t i = 0; i < self.ndimension()-2; i++) {
      if(++counter[i] >= self.size(i)) {
        if(i == self.ndimension()-3) return;  // 遍历完成
        counter[i] = 0;
      } else break;
    }
    
    // 打印子张量标题
    stream << "\n(";
    Tensor subtensor = self;
    for(auto dim : counter) {
      subtensor = subtensor.select(0, dim);
      stream << dim+1 << ",";
    }
    stream << ".,.) =\n";
    
    // 打印矩阵块
    __printMatrix(stream, subtensor, linesize, 1);
  }
}

// 主打印函数(根据张量维度分派)
std::ostream& print(std::ostream& stream, const Tensor & tensor_, int64_t linesize) {
  FormatGuard guard(stream);  // 自动格式保护
  
  if(!tensor_.defined()) {
    stream << "[ Tensor (undefined) ]";
  } else if (tensor_.is_sparse()) {  // 稀疏张量特殊处理
    stream << "[ " << tensor_.toString() << "{}\n"
           << "indices:\n" << tensor_._indices() << "\n"
           << "values:\n" << tensor_._values() << "\n"
           << "size:\n" << tensor_.sizes() << "\n]";
  } else {
    // 转换为CPU双精度连续张量
    Tensor tensor = tensor_.toType(
      tensor_.type().toBackend(Backend::CPU).toScalarType(kDouble)
    ).contiguous();
    
    // 按维度处理
    switch(tensor.ndimension()) {
      case 0:  // 标量
        stream << defaultfloat << tensor.data<double>()[0] << "\n"
               << "[ " << tensor_.toString() << "{} ]";
        break;
      case 1:  // 向量
        if(tensor.numel() > 0) {
          auto [scale, sz] = __printFormat(stream, tensor);
          if(scale != 1) printScale(stream, scale);
          double* data = tensor.data<double>();
          for(int64_t i = 0; i < tensor.size(0); i++)
            stream << std::setw(sz) << data[i]/scale << "\n";
        }
        stream << "[ " << tensor_.toString() << "{" << tensor.size(0) << "} ]";
        break;
      case 2:  // 矩阵
        if(tensor.numel() > 0) 
          __printMatrix(stream, tensor, linesize, 0);
        stream << "[ " << tensor_.toString() << "{"
               << tensor.size(0) << "," << tensor.size(1) << "} ]";
        break;
      default:  // 高维张量
        if(tensor.numel() > 0)
          __printTensor(stream, tensor, linesize);
        stream << "[ " << tensor_.toString() << "{" << tensor.size(0);
        for(int64_t i = 1; i < tensor.ndimension(); i++)
          stream << "," << tensor.size(i);
        stream << "} ]";
    }
  }
  return stream;
}

} // namespace at