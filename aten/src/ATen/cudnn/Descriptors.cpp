#include "Descriptors.h"  
#include <ATen/ATen.h>  // PyTorch核心头文件
#include <ostream>
#include <sstream>
#include <string>

namespace at { namespace native {  // PyTorch原生命名空间

namespace {  // 匿名命名空间，存放内部辅助函数

// 根据Tensor类型获取对应的cuDNN数据类型
inline cudnnDataType_t getDataType(const at::Type& t) {
  auto scalar_type = t.scalarType();
  if (scalar_type == at::kFloat) {
    return CUDNN_DATA_FLOAT;
  } else if (scalar_type == at::kHalf) {
    return CUDNN_DATA_HALF;
  } else if (scalar_type == at::kDouble) {
    return CUDNN_DATA_DOUBLE;
  }
  throw std::runtime_error("TensorDescriptor only supports double, float and half tensors");
}

// Tensor对象重载版本
inline cudnnDataType_t getDataType(const at::Tensor& t) {
  return getDataType(t.type());
}

} // anonymous namespace

// 设置Tensor描述符（根据PyTorch Tensor）
void TensorDescriptor::set(const at::Tensor &t, size_t pad) {
  set(getDataType(t), t.sizes(), t.strides(), pad);
}

// 设置Tensor描述符（直接参数版本）
void TensorDescriptor::set(cudnnDataType_t datatype, IntList t_sizes, IntList t_strides, size_t pad) {
  size_t dim = t_sizes.size();
  if (dim > CUDNN_DIM_MAX || pad > CUDNN_DIM_MAX)  // 检查维度限制
#define _STR(X) #X
#define STR(X) _STR(X)
    throw std::runtime_error("cuDNN supports only up to " STR(CUDNN_DIM_MAX) " dimensions");
#undef _STR
#undef STR
  
  // 转换尺寸和步长数组
  int size[CUDNN_DIM_MAX];
  int stride[CUDNN_DIM_MAX];
  for (size_t i = 0; i < dim; ++i) {
    size[i] = static_cast<int>(t_sizes[i]);
    stride[i] = static_cast<int>(t_strides[i]);
  }
  // 填充不足的维度
  for (size_t i = dim; i < pad; ++i) {
    size[i] = 1;
    stride[i] = 1;
  }
  set(datatype, static_cast<int>(std::max(dim, pad)), size, stride);
}

// cuDNN数据类型转字符串
std::string cudnnTypeToString(cudnnDataType_t dtype) {
  switch (dtype) {
    case CUDNN_DATA_FLOAT: return "CUDNN_DATA_FLOAT";
    case CUDNN_DATA_DOUBLE: return "CUDNN_DATA_DOUBLE";
    case CUDNN_DATA_HALF: return "CUDNN_DATA_HALF";
    case CUDNN_DATA_INT8: return "CUDNN_DATA_INT8";
    case CUDNN_DATA_INT32: return "CUDNN_DATA_INT32";
    case CUDNN_DATA_INT8x4: return "CUDNN_DATA_INT8x4";
#if CUDNN_VERSION >= 7100
    case CUDNN_DATA_UINT8: return "CUDNN_DATA_UINT8";
    case CUDNN_DATA_UINT8x4: return "CUDNN_DATA_UINT8x4";
#endif
    default:  // 未知类型处理
      std::ostringstream oss;
      oss << "(unknown data-type " << static_cast<int>(dtype) << ")";
      return oss.str();
  }
}

// 重载<<运算符输出描述符信息
std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d) {
  out << "TensorDescriptor " << static_cast<void*>(d.desc()) << "\n";
  int nbDims;
  int dimA[CUDNN_DIM_MAX];
  int strideA[CUDNN_DIM_MAX];
  cudnnDataType_t dtype;
  // 获取描述符详细信息
  cudnnGetTensorNdDescriptor(d.desc(), CUDNN_DIM_MAX, &dtype, &nbDims, dimA, strideA);
  out << "    type = " << cudnnTypeToString(dtype) << "\n";
  out << "    nbDims = " << nbDims << "\n";
  // 输出维度信息
  out << "    dimA = ";
  for (auto i : ArrayRef<int>{dimA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  // 输出步长信息
  out << "    strideA = ";
  for (auto i : ArrayRef<int>{strideA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  return out;
}

// 打印描述符信息
void TensorDescriptor::print() { std::cout << *this; }

// 设置过滤器描述符
void FilterDescriptor::set(const at::Tensor &t, int64_t pad) {
  auto dim = t.ndimension();
  if (dim > CUDNN_DIM_MAX || pad > CUDNN_DIM_MAX)  // 检查维度限制
#define _STR(X) #X
#define STR(X) _STR(X)
    throw std::runtime_error("cuDNN supports only up to " STR(CUDNN_DIM_MAX) " dimensions");
#undef _STR
#undef STR
  // 检查内存连续性（过滤器必须连续存储）
  if (!t.is_contiguous()) {
    throw std::runtime_error("cuDNN filters (a.k.a. weights) must be contiguous");
  }
  // 转换尺寸数组
  int size[CUDNN_DIM_MAX];
  for (int i = 0; i < dim; ++i) {
    size[i] = (int) t.size(i);
  }
  // 填充不足的维度
  for (int i = dim; i < pad; ++i) {
    size[i] = (int) 1;
  }
  dim = std::max(dim, pad);
  set(getDataType(t), (int) dim, size);
}

}}  // namespace at::native
