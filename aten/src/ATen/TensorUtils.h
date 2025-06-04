#pragma once

#include "ATen/Tensor.h"
#include "ATen/TensorGeometry.h"
#include "ATen/Utils.h"

// 这些函数不在Utils.h中，因为本文件依赖于Tensor.h

namespace at {

// 以下是一组用于检查参数合法性的工具函数。
// 这些函数对于原生函数特别有用，因为原生函数默认不进行参数检查。

// TensorArg结构体：封装Tensor并提供名称和位置信息（用于错误报告）
struct CAFFE2_API TensorArg {
  Tensor tensor;       // 封装的张量
  const char* name;    // 参数名称（用于错误信息）
  int pos;             // 参数位置（1-based索引）
  TensorArg(Tensor tensor, const char* name, int pos)
    : tensor(std::move(tensor)), name(name), pos(pos) {}
  const Tensor* operator->() const { return &tensor; }
  const Tensor& operator*() const { return tensor; }
};

// TensorGeometryArg结构体：封装TensorGeometry并提供名称和位置信息
struct CAFFE2_API TensorGeometryArg {
  TensorGeometry tensor;  // 封装的张量几何信息
  const char* name;      // 参数名称
  int pos;               // 参数位置（1-based索引）
  /* implicit */ TensorGeometryArg(TensorArg arg)
    : tensor(TensorGeometry{arg.tensor}), name(arg.name), pos(arg.pos) {}
  TensorGeometryArg(TensorGeometry tensor, const char* name, int pos)
    : tensor(tensor), name(name), pos(pos) {}
  const TensorGeometry* operator->() const { return &tensor; }
  const TensorGeometry& operator*() const { return tensor; }
};

// CheckedFrom类型：用于描述哪个函数进行了输入参数检查
// TODO: 考虑将其泛化为调用栈
using CheckedFrom = const char*;

// 未定义张量的处理约定：
// - 单目运算符假定它们的参数是已定义的
// - 接受多个张量的函数会隐式过滤掉未定义的张量
//   （便于编写条件测试：当张量定义时应用测试，否则不应用）
//
// 注意：这意味着n元运算符接受TensorArg列表而非TensorGeometryArg，
// 因为Tensor到TensorGeometry的转换会在遇到未定义张量时抛出异常

// 输出运算符重载：用于打印TensorGeometryArg信息
CAFFE2_API std::ostream& operator<<(std::ostream& out, TensorGeometryArg t);

// 检查张量维度是否为指定值
CAFFE2_API void checkDim(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim);
// 检查张量维度是否在指定范围内[start, end)
CAFFE2_API void checkDimRange(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim_start,
    int64_t dim_end);
// 检查两个张量维度是否相同
CAFFE2_API void checkSameDim(
    CheckedFrom c,
    const TensorGeometryArg& t1,
    const TensorGeometryArg& t2);
// 检查张量是否是连续的
CAFFE2_API void checkContiguous(CheckedFrom c, const TensorGeometryArg& t);
// 检查所有张量是否都是连续的
CAFFE2_API void checkAllContiguous(CheckedFrom c, at::ArrayRef<TensorArg> ts);
// 检查张量大小是否符合指定值
CAFFE2_API void checkSize(
    CheckedFrom c,
    const TensorGeometryArg& t,
    IntList sizes);
// 检查张量特定维度的大小是否符合指定值
CAFFE2_API void checkSize(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim,
    int64_t size);
// 检查张量元素总数是否符合指定值
CAFFE2_API void checkNumel(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t numel);
// 检查两个张量元素总数是否相同
CAFFE2_API void checkSameNumel(
    CheckedFrom c,
    const TensorGeometryArg& t1,
    const TensorGeometryArg& t2);
// 检查所有张量元素总数是否相同
CAFFE2_API void checkAllSameNumel(CheckedFrom c, ArrayRef<TensorArg> tensors);
// 检查张量标量类型是否符合指定值
CAFFE2_API void checkScalarType(
    CheckedFrom c,
    const TensorArg& t,
    ScalarType s);
// 检查张量标量类型是否符合指定列表中的任一类型
CAFFE2_API void checkScalarTypes(
    CheckedFrom c,
    const TensorArg& t,
    at::ArrayRef<ScalarType> l);
// 检查两个张量是否在相同GPU设备上
CAFFE2_API void checkSameGPU(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
// 检查所有张量是否在相同GPU设备上
CAFFE2_API void checkAllSameGPU(CheckedFrom c, ArrayRef<TensorArg> tensors);
// 检查两个张量类型是否相同
CAFFE2_API void checkSameType(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
// 检查所有张量类型是否相同
CAFFE2_API void checkAllSameType(CheckedFrom c, ArrayRef<TensorArg> tensors);
// 检查两个张量大小是否相同
CAFFE2_API void checkSameSize(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
// 检查张量是否已定义
CAFFE2_API void checkDefined(CheckedFrom c, const TensorArg& t);
// 检查所有张量是否已定义
CAFFE2_API void checkAllDefined(CheckedFrom c, at::ArrayRef<TensorArg> t);

// 注意：使用TensorArg会降低性能吗？
// 检查张量后端是否符合指定值
CAFFE2_API void checkBackend(
    CheckedFrom c,
    at::ArrayRef<Tensor> t,
    at::Backend backend);

// 获取张量数据指针（如果张量已定义）
CAFFE2_API void* maybe_data_ptr(const Tensor& tensor);
CAFFE2_API void* maybe_data_ptr(const TensorArg& tensor);

// 检查给定大小和步长的张量几何结构是否是连续的
// 虽然我们现在在张量中缓存了is_contiguous，但这个函数仍然有用，
// 因为它允许在不显式构造张量的情况下检查特定几何结构是否连续，
// 例如当你需要基于子几何结构是否连续来选择内核策略时。
CAFFE2_API bool geometry_is_contiguous(IntList sizes, IntList strides);
}