#include "ATen/Config.h"        // ATen配置头文件
#include "ATen/TensorUtils.h"   // 张量工具函数
#include "ATen/ATen.h"          // ATen核心功能

#include <ostream>
#include <sstream>

namespace at {

// 张量几何参数输出运算符重载
std::ostream& operator<<(std::ostream & out, TensorGeometryArg t) {
  if (t.pos == 0) {
    // 位置0有特殊含义，通常表示'self'或返回张量
    out << "'" << t.name << "'";
  } else {
    out << "argument #" << t.pos << " '" << t.name << "'";
  }
  return out;
}

/******************** 张量维度检查函数 ********************/

// 检查张量维度是否等于指定值
void checkDim(CheckedFrom c, const TensorGeometryArg& t, int64_t dim) {
  AT_CHECK(t->dim() == dim,
    "Expected ", dim, "-dimensional tensor, but got ", t->dim(),
    "-dimensional tensor for ", t," (while checking arguments for ", c, ")");
}

// 检查张量维度是否在指定范围内
void checkDimRange(CheckedFrom c, const TensorGeometryArg& t, int64_t dim_start, int64_t dim_end) {
  AT_CHECK(
    t->dim() >= dim_start && t->dim() < dim_end,
    "Expected ", dim_start, " to ", (dim_end - 1), " dimensions, but got ",
    t->dim(), "-dimensional tensor for ", t, " (while checking arguments for ",
    c, ")");
}

/******************** 张量连续性检查函数 ********************/

// 检查单个张量是否连续存储
void checkContiguous(CheckedFrom c, const TensorGeometryArg& t) {
  AT_CHECK(
    t->is_contiguous(),
    "Expected contiguous tensor, but got non-contiguous tensor for ", t,
     " (while checking arguments for ", c, ")");
}

// 检查多个张量是否都连续存储
void checkAllContiguous(CheckedFrom c, at::ArrayRef<TensorArg> ts) {
  for (auto& t : ts) {
    if (!t->defined()) continue;  // 跳过未定义的张量
    checkContiguous(c, t);
  }
}

/******************** 张量大小检查函数 ********************/

// 检查张量大小是否匹配指定值
void checkSize(CheckedFrom c, const TensorGeometryArg& t, IntList sizes) {
  checkDim(c, t, sizes.size());  // 先检查维度是否匹配
  AT_CHECK(
    t->sizes().equals(sizes),
    "Expected tensor of size ", sizes, ", but got tensor of size ", t->sizes(),
    " for ", t, " (while checking arguments for ", c, ")");
}

// 检查指定维度的大小是否匹配
void checkSize(CheckedFrom c, const TensorGeometryArg& t, int64_t dim, int64_t size) {
  AT_CHECK(
    t->size(dim) == size,
    "Expected tensor to have size ", size, " at dimension ", dim,
    ", but got size ", t->size(dim), " for ", t,
    " (while checking arguments for ", c, ")");
}

/******************** 张量一致性检查函数 ********************/

// 通用检查函数，验证一组张量是否满足特定条件
void checkAllSame(CheckedFrom c, ArrayRef<TensorArg> tensors, 
                 void(*fn)(CheckedFrom, const TensorArg&, const TensorArg&)) {
  const TensorArg* t0 = nullptr;  // 第一个有效张量作为基准
  for (auto& t : tensors) {
    if (!t->defined()) continue;  // 跳过未定义的张量
    if (t0 != nullptr) {
      fn(c, *t0, t);  // 与基准张量比较
    } else {
      t0 = &t;  // 设置基准张量
    }
  }
}

// 检查两个张量大小是否相同
void checkSameSize(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  AT_CHECK(
    t1->sizes().equals(t2->sizes()),
    "Expected tensor for ", t1, " to have same size as tensor for ", t2,
    "; but ", t1->sizes(), " does not equal ", t2->sizes(),
    " (while checking arguments for ", c, ")");
}

// 检查多个张量大小是否相同
void checkAllSameSize(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameSize);
}

/******************** 张量元素数量检查函数 ********************/

// 检查张量元素数量是否匹配
void checkNumel(CheckedFrom c, const TensorGeometryArg& t, int64_t numel) {
  AT_CHECK(
    t->numel() == numel,
    "Expected tensor for ", t, " to have ", numel,
    " elements; but it actually has ", t->numel(), " elements",
    " (while checking arguments for ", c, ")");
}

// 检查两个张量元素数量是否相同
void checkSameNumel(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  AT_CHECK(
    t1->numel() == t2->numel(),
    "Expected tensor for ", t1,
    " to have same number of elements as tensor for ", t2, "; but ",
    t1->numel(), " does not equal ", t2->numel(),
    " (while checking arguments for ", c, ")");
}

// 检查多个张量元素数量是否相同
void checkAllSameNumel(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameNumel);
}

/******************** GPU相关检查函数 ********************/

// 检查两个张量是否在相同GPU设备上
void checkSameGPU(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  if (! (t1->is_cuda()) || ! (t2->is_cuda())) {
    std::ostringstream oss;
    if (! t1->is_cuda()) {
      oss << "Tensor for " << t1 << " is on CPU, ";
    }
    if (! t2->is_cuda()) {
      oss << "Tensor for " << t2 << " is on CPU, ";
    }
    oss << "but expected " << ((!(t1->is_cuda() || t2->is_cuda())) ? "them" : "it")
        << " to be on GPU (while checking arguments for " << c << ")";
    AT_ERROR(oss.str());
  }
  AT_CHECK(
    t1->get_device() == t2->get_device(),
    "Expected tensor for ", t1, " to have the same device as tensor for ", t2,
    "; but device ", t1->get_device(), " does not equal ", t2->get_device(),
    " (while checking arguments for ", c, ")");
}

// 检查多个张量是否在相同GPU设备上
void checkAllSameGPU(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameGPU);
}

/******************** 类型检查函数 ********************/

// 检查两个张量类型是否相同
void checkSameType(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  AT_CHECK(
    t1->type() == t2->type(),
    "Expected tensor for ", t1, " to have the same type as tensor for ", t2,
    "; but type ", t1->toString(), " does not equal ", t2->toString(),
    " (while checking arguments for ", c, ")");
}

// 检查张量标量类型是否匹配
void checkScalarType(CheckedFrom c, const TensorArg& t, ScalarType ty) {
  AT_CHECK(
    t->type().scalarType() == ty,
    "Expected tensor for ", t, " to have scalar type ", toString(ty),
    "; but got ", t->toString(), " instead (while checking arguments for ", c,
    ")");
}

// 检查张量标量类型是否在允许列表中
void checkScalarTypes(CheckedFrom c, const TensorArg& t,
                     at::ArrayRef<ScalarType> l) {
    if (std::find(l.begin(), l.end(), t->type().scalarType()) == l.end()) {
      std::ostringstream oss;
      oss << "Expected tensor for " << t << " to have one of the following "
          << "scalar types: ";
      size_t i = 0;
      for (auto ty : l) {
        if (i != 0) {
          oss << ", ";
        }
        oss << toString(ty);
        i++;
      }
      oss << "; but got " << t->toString()
          << " instead (while checking arguments for " << c << ")";
      AT_ERROR(oss.str());
    }
}

// 检查多个张量类型是否相同
void checkAllSameType(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameType);
}

/******************** 其他检查函数 ********************/

// 检查两个张量维度是否相同
void checkSameDim(CheckedFrom c, const TensorGeometryArg& t1, const TensorGeometryArg& t2) {
  AT_CHECK(
    t1->dim() == t2->dim(),
    "Expected tensor for ", t1, " to have the same dimension as tensor for ",
    t2, "; but ", t1->dim(), " does not equal ", t2->dim(),
    " (while checking arguments for ", c, ")");
}

// 检查张量是否已定义
void checkDefined(CheckedFrom c, const TensorArg& t) {
  AT_CHECK(
    t->defined(),
    "Expected tensor for ", t, " to be non-null, but it was undefined ",
    " (while checking arguments for ", c, ")");
}

// 检查多个张量是否都已定义
void checkAllDefined(CheckedFrom c, ArrayRef<TensorArg> ts) {
  for (auto t : ts) {
    checkDefined(c, t);
  }
}

// 检查张量后端类型是否匹配
void checkBackend(CheckedFrom c, const Tensor& t, Backend backend) {
  AT_CHECK(
    t.type().backend() == backend,
    "Expected tensor to have ", toString(backend),
    " Backend, but got tensor with ", toString(t.type().backend()), " Backend ",
    "(while checking arguments for ", c, ")");
}

// 检查多个张量后端类型是否匹配
void checkBackend(CheckedFrom c, ArrayRef<Tensor> tensors, at::Backend backend) {
  for (auto &t : tensors) {
    checkBackend(c, t, backend);
  }
}

/******************** 工具函数 ********************/

// 安全获取张量数据指针，处理未定义情况
void * maybe_data_ptr(const Tensor& tensor) {
  return tensor.defined() ? (void *)tensor.data_ptr() : nullptr;
}

// 安全获取张量参数数据指针，处理未定义情况
void * maybe_data_ptr(const TensorArg& tensor) {
  return tensor->defined() ? (void *)tensor->data_ptr() : nullptr;
}

// 根据大小和步长判断张量是否连续存储
bool geometry_is_contiguous(IntList sizes, IntList strides) {
  int64_t dim = sizes.size();
  int64_t expected_stride = 1;
  bool contig_if_nonempty = true;
  for (int64_t i = dim - 1; i >= 0; i--) {
    if (sizes[i] == 0) {
      return true;  // 空张量视为连续
    }
    if (contig_if_nonempty) {
      if (sizes[i] != 1 && strides[i] != expected_stride) {
        contig_if_nonempty = false;
      }
      expected_stride *= sizes[i];
    }
  }
  return contig_if_nonempty;
}

} // namespace at