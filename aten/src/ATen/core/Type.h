#pragma once  // 防止头文件重复包含

// ATen核心功能头文件
#include "ATen/core/ATenGeneral.h"  // ATen通用宏和基础定义
#include <c10/core/Allocator.h>     // 内存分配器接口
#include "ATen/core/Deprecated.h"   // 标记废弃API
#include "ATen/core/Generator.h"    // 随机数生成器
#include <c10/core/Layout.h>        // 张量内存布局定义
#include <c10/core/Scalar.h>        // 标量类型封装
#include <c10/core/ScalarType.h>    // 标量类型枚举
#include "ATen/core/SparseTensorRef.h" // 稀疏张量引用
#include <c10/util/ArrayRef.h>      // 数组视图类
#include <c10/Half.h>               // 半精度浮点支持
#include <c10/core/TensorTypeIdRegistration.h> // 张量类型ID注册
#include "ATen/core/Reduction.h"    // 归约操作类型
#include "ATen/core/TensorOptions.h" // 张量配置选项

#include <c10/util/Optional.h>      // 可选值模板

// 标准库头文件
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>

// 解决Windows平台s_addr冲突
#ifdef _MSC_VER
#ifdef s_addr
#undef s_addr
#endif
#endif

namespace c10 {
struct Storage;  // 前向声明存储结构
}

namespace at {

// 前向声明
class Context;
struct Generator;
class Tensor;

// 空删除器，用于不执行实际释放的智能指针
static inline void noop_deleter(void*) {}

/**
 * @brief 张量类型ID枚举
 * 
 * 定义了所有支持的张量类型，包括：
 * - CPU密集/稀疏张量(8种标量类型)
 * - CUDA密集/稀疏张量(8种标量类型) 
 * - 复数类型(CPU/CUDA)
 * - 特殊类型(未定义类型)
 */
enum class TypeID {
  // CPU密集张量类型(8种)
  CPUByte, CPUChar, CPUDouble, CPUFloat, 
  CPUInt, CPULong, CPUShort, CPUHalf,
  
  // CPU稀疏张量类型(7种)
  SparseCPUByte, SparseCPUChar, SparseCPUDouble,
  SparseCPUFloat, SparseCPUInt, SparseCPULong, SparseCPUShort,
  
  // CUDA密集张量类型(8种)
  CUDAByte, CUDAChar, CUDADouble, CUDAFloat,
  CUDAInt, CUDALong, CUDAShort, CUDAHalf,
  
  // CUDA稀疏张量类型(7种) 
  SparseCUDAByte, SparseCUDAChar, SparseCUDADouble,
  SparseCUDAFloat, SparseCUDAInt, SparseCUDALong, SparseCUDAShort,
  
  // 复数类型
  CPUComplexFloat, CPUComplexDouble,  // CPU复数
  CUDAComplexFloat, CUDAComplexDouble, // CUDA复数
  
  // 特殊类型
  Undefined,  // 未定义类型
  NumOptions  // 类型总数
};

/**
 * @brief 类型系统基类
 * 
 * 定义了张量类型的统一接口，包括：
 * - 类型属性查询
 * - 内存管理
 * - 张量操作
 * - 类型转换
 */
struct CAFFE2_API Type {
  // 构造函数
  explicit Type(TensorTypeId type_id, bool is_variable, bool is_undefined)
      : type_id_(type_id), is_variable_(is_variable), is_undefined_(is_undefined) {}

  virtual ~Type() {}  // 虚析构函数
  
  // === 基础属性查询 ===
  virtual ScalarType scalarType() const = 0;  // 获取标量类型
  virtual caffe2::TypeMeta typeMeta() const = 0;  // 获取类型元信息
  virtual Backend backend() const = 0;  // 获取后端类型(CPU/CUDA等)
  
  // 布局类型(从后端推导)
  Layout layout() const noexcept { return layout_from_backend(backend()); }
  
  // 设备类型判断
  virtual bool is_cuda() const = 0;    // 是否CUDA设备
  virtual bool is_hip() const = 0;     // 是否HIP设备
  virtual bool is_sparse() const = 0;  // 是否稀疏张量
  virtual bool is_distributed() const = 0; // 是否分布式张量
  
  // 特殊属性
  bool is_variable() const noexcept { return is_variable_; }  // 是否自动微分变量
  bool is_undefined() const noexcept { return is_undefined_; } // 是否未定义类型
  
  // === 内存管理 ===
  virtual Allocator* allocator() const = 0;  // 获取内存分配器
  virtual Device getDeviceFromPtr(void* data) const = 0;  // 从指针获取设备
  
  // 存储管理
  virtual Storage storage(bool resizable = false) const = 0;  // 创建可调整大小的存储
  virtual Storage storage(size_t size, bool resizable = false) const = 0; // 指定大小的存储
  virtual Storage storageFromBlob(void* data, int64_t size, 
                               const std::function<void(void*)>& deleter=noop_deleter) const = 0;
  virtual Storage storageWithAllocator(int64_t size, Allocator* allocator) const = 0;
  
  // === 张量操作 ===
  virtual Tensor copy(const Tensor& src, bool non_blocking=false, 
                    c10::optional<Device> to_device={}) const = 0;  // 张量拷贝
  virtual Tensor& copy_(Tensor& self, const Tensor& src, bool non_blocking=false) const = 0;
  
  // 自动微分
  virtual void backward(
      Tensor& self,
      c10::optional<Tensor> gradient,
      bool keep_graph,
      bool create_graph) const = 0;
      
  // === 类型转换 ===
  virtual Type& toBackend(Backend b) const = 0;  // 转换后端
  virtual Type& toScalarType(ScalarType s) const = 0;  // 转换标量类型
  
  // 稀疏/密集转换便捷方法
  Type& toSparse() const {
    return this->toBackend(at::toSparse(this->backend()));
  }
  Type& toDense() const {
    return this->toBackend(at::toDense(this->backend()));
  }
  
  // 设备转换便捷方法
  Type& cpu() const {
    return this->toBackend(at::backendToCPU(this->backend()));
  }
  Type& cuda() const {
    return this->toBackend(at::backendToCUDA(this->backend()));
  }
  Type& hip() const {
    return this->toBackend(at::backendToHIP(this->backend()));
  }
  
  // === 其他功能 ===
  virtual TypeID ID() const = 0;  // 获取类型ID
  TensorTypeId type_id() const { return type_id_; }  // 获取张量类型ID
  
  // 转换为TensorOptions
  TensorOptions options(int16_t device_index = -1) const {
    return TensorOptions().dtype(typeMeta())
                         .device(device_type(), device_index)
                         .layout(layout())
                         .is_variable(is_variable());
  }
  
  operator TensorOptions() const {
    return options();
  }

protected:
  TensorTypeId type_id_;      // 类型唯一标识
  bool is_variable_;          // 是否为自动微分变量
  bool is_undefined_;         // 是否为未定义类型
};
  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) = 0;
// 这是一个抽象基类，定义了张量(Tensor)的各种操作接口
// 所有方法都是纯虚函数(=0)，需要由子类实现具体功能

// 示例(已注释掉)
// virtual Tensor * add(Tensor & a, Tensor & b) = 0; 

/* 数学运算类方法 */
// 计算张量各元素的绝对值，返回新张量
virtual Tensor abs(const Tensor & self) const = 0;
// 原地(in-place)计算张量各元素的绝对值，返回自身引用  
virtual Tensor & abs_(Tensor & self) const = 0;
// 计算张量各元素的反余弦值，返回新张量
virtual Tensor acos(const Tensor & self) const = 0;
// 原地计算张量各元素的反余弦值，返回自身引用
virtual Tensor & acos_(Tensor & self) const = 0;

/* 张量加法操作 */
// 张量相加：self + alpha * other，返回新张量  
virtual Tensor add(const Tensor & self, const Tensor & other, Scalar alpha) const = 0;
// 原地张量相加：self += alpha * other，返回自身引用
virtual Tensor & add_(Tensor & self, const Tensor & other, Scalar alpha) const = 0;
// 张量与标量相加：self + alpha * other，返回新张量
virtual Tensor add(const Tensor & self, Scalar other, Scalar alpha) const = 0;
// 原地张量与标量相加：self += alpha * other，返回自身引用
virtual Tensor & add_(Tensor & self, Scalar other, Scalar alpha) const = 0;

/* 矩阵/向量运算 */
// 矩阵-向量乘加操作：beta*self + alpha*(mat @ vec)
virtual Tensor addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const = 0;
// 原地矩阵-向量乘加操作
virtual Tensor & addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const = 0;
// 外积相加操作：beta*self + alpha*(vec1 ⊗ vec2)  
virtual Tensor addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const = 0;
// 原地外积相加操作
virtual Tensor & addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const = 0;

/* 逻辑判断操作 */
// 沿指定维度判断所有元素是否为真(非零)，可保留维度
virtual Tensor all(const Tensor & self, int64_t dim, bool keepdim) const = 0;
// 判断两个张量是否近似相等(考虑相对误差rtol和绝对误差atol)
virtual bool allclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) const = 0;
// 沿指定维度判断是否有任意元素为真，可保留维度
virtual Tensor any(const Tensor & self, int64_t dim, bool keepdim) const = 0;

/* 极值索引操作 */
// 沿指定维度返回最大值索引，可保留维度
virtual Tensor argmax(const Tensor & self, int64_t dim, bool keepdim) const = 0;
// 返回展平后张量的全局最大值索引
virtual Tensor argmax(const Tensor & self) const = 0;
// 沿指定维度返回最小值索引，可保留维度  
virtual Tensor argmin(const Tensor & self, int64_t dim, bool keepdim) const = 0;
// 返回展平后张量的全局最小值索引
virtual Tensor argmin(const Tensor & self) const = 0;

/* 张量视图操作 */
// 创建具有指定大小和步长的新视图(无存储偏移)
virtual Tensor as_strided(const Tensor & self, IntList size, IntList stride) const = 0;
// 原地修改为指定大小和步长的视图(无存储偏移)
virtual Tensor & as_strided_(Tensor & self, IntList size, IntList stride) const = 0;
// 创建具有指定大小、步长和存储偏移的新视图
virtual Tensor as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const = 0;
// 原地修改为指定大小、步长和存储偏移的视图
virtual Tensor & as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const = 0;

/* 三角函数 */
virtual Tensor asin(const Tensor & self) const = 0;  // 反正弦
virtual Tensor & asin_(Tensor & self) const = 0;    // 原地反正弦
virtual Tensor atan(const Tensor & self) const = 0; // 反正切  
virtual Tensor & atan_(Tensor & self) const = 0;    // 原地反正切

/* 批处理矩阵乘法 */
// 批处理矩阵乘加：beta*self + alpha*(batch1 @ batch2)
virtual Tensor baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const = 0;
// 原地批处理矩阵乘加
virtual Tensor & baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const = 0;

/* 随机数生成 */
// 按self张量的概率进行伯努利采样
virtual Tensor bernoulli(const Tensor & self, Generator * generator) const = 0;
// 原地按给定概率张量p进行伯努利采样
virtual Tensor & bernoulli_(Tensor & self, const Tensor & p, Generator * generator) const = 0;
// 原地按标量概率p进行伯努利采样  
virtual Tensor & bernoulli_(Tensor & self, double p, Generator * generator) const = 0;
// 按标量概率p进行伯努利采样，返回新张量
virtual Tensor bernoulli(const Tensor & self, double p, Generator * generator) const = 0;

/* 直方图统计 */
// 计算值的直方图(需指定箱数、最小最大值)
virtual Tensor bincount(const Tensor & self, const Tensor & weights, int64_t minlength) const = 0;

/* 矩阵乘法 */
// 批处理矩阵乘法
virtual Tensor bmm(const Tensor & self, const Tensor & mat2) const = 0;

/* 取整函数 */
virtual Tensor ceil(const Tensor & self) const = 0;  // 向上取整
virtual Tensor & ceil_(Tensor & self) const = 0;    // 原地向上取整

/* 张量分块 */
// 将张量沿指定维度分成chunks块
virtual std::vector<Tensor> chunk(const Tensor & self, int64_t chunks, int64_t dim) const = 0;

/* 裁剪操作 */
// 将值裁剪到[min, max]范围(可选)
virtual Tensor clamp(const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) const = 0;
virtual Tensor & clamp_(Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) const = 0;
// 只裁剪最大值
virtual Tensor clamp_max(const Tensor & self, Scalar max) const = 0;
virtual Tensor & clamp_max_(Tensor & self, Scalar max) const = 0;
// 只裁剪最小值  
virtual Tensor clamp_min(const Tensor & self, Scalar min) const = 0;
virtual Tensor & clamp_min_(Tensor & self, Scalar min) const = 0;

/* 连续性处理 */
// 返回内存连续的张量(必要时复制)
virtual Tensor contiguous(const Tensor & self) const = 0;

/* 三角函数 */
virtual Tensor cos(const Tensor & self) const = 0;  // 余弦
virtual Tensor & cos_(Tensor & self) const = 0;    // 原地余弦
virtual Tensor cosh(const Tensor & self) const = 0; // 双曲余弦
virtual Tensor & cosh_(Tensor & self) const = 0;    // 原地双曲余弦

/* 累积操作 */
// 沿维度累积求和(可指定输出类型)
virtual Tensor cumsum(const Tensor & self, int64_t dim, ScalarType dtype) const = 0;
// 沿维度累积求和(自动推断类型)
virtual Tensor cumsum(const Tensor & self, int64_t dim) const = 0;
// 沿维度累积求积(可指定输出类型)
virtual Tensor cumprod(const Tensor & self, int64_t dim, ScalarType dtype) const = 0;
// 沿维度累积求积(自动推断类型)  
virtual Tensor cumprod(const Tensor & self, int64_t dim) const = 0;

/* 线性代数 */
virtual Tensor det(const Tensor & self) const = 0;  // 矩阵行列式
// 创建对角嵌入张量
virtual Tensor diag_embed(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) const = 0;
// 创建扁平对角张量
virtual Tensor diagflat(const Tensor & self, int64_t offset) const = 0;
// 返回对角线元素
virtual Tensor diagonal(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) const = 0;

/* 除法运算 */
virtual Tensor div(const Tensor & self, const Tensor & other) const = 0;  // 张量除法
virtual Tensor & div_(Tensor & self, const Tensor & other) const = 0;    // 原地张量除法
virtual Tensor div(const Tensor & self, Scalar other) const = 0;         // 标量除法
virtual Tensor & div_(Tensor & self, Scalar other) const = 0;           // 原地标量除法

/* 点积 */
virtual Tensor dot(const Tensor & self, const Tensor & tensor) const = 0;  // 向量点积

/* 调整大小 */
virtual Tensor & resize_(Tensor & self, IntList size) const = 0;  // 原地调整张量大小

/* 误差函数 */
virtual Tensor erf(const Tensor & self) const = 0;  // 误差函数
virtual Tensor & erf_(Tensor & self) const = 0;     // 原地误差函数
virtual Tensor erfc(const Tensor & self) const = 0; // 互补误差函数
virtual Tensor & erfc_(Tensor & self) const = 0;    // 原地互补误差函数

/* 指数函数 */
virtual Tensor exp(const Tensor & self) const = 0;  // 指数函数
virtual Tensor & exp_(Tensor & self) const = 0;     // 原地指数函数
virtual Tensor expm1(const Tensor & self) const = 0; // exp(x)-1
virtual Tensor & expm1_(Tensor & self) const = 0;    // 原地exp(x)-1

/* 张量扩展 */
// 扩展张量到指定大小(可隐式扩展)
virtual Tensor expand(const Tensor & self, IntList size, bool implicit) const = 0;
// 扩展为与other相同形状
virtual Tensor expand_as(const Tensor & self, const Tensor & other) const = 0;

/* 扁平化 */
// 将[start_dim, end_dim]范围的维度展平
virtual Tensor flatten(const Tensor & self, int64_t start_dim, int64_t end_dim) const = 0;

/* 填充操作 */
// 用标量值填充张量
virtual Tensor & fill_(Tensor & self, Scalar value) const = 0;
// 用张量值填充
virtual Tensor & fill_(Tensor & self, const Tensor & value) const = 0;

/* 取整函数 */
virtual Tensor floor(const Tensor & self) const = 0;  // 向下取整
virtual Tensor & floor_(Tensor & self) const = 0;    // 原地向下取整

/* 外积 */
virtual Tensor ger(const Tensor & self, const Tensor & vec2) const = 0;  // 向量外积

/* 线性方程组求解 */
// 解线性方程组Ax=self，返回解和LU分解
virtual std::tuple<Tensor,Tensor> gesv(const Tensor & self, const Tensor & A) const = 0;

/* 傅里叶变换 */
virtual Tensor fft(const Tensor & self, int64_t signal_ndim, bool normalized) const = 0;  // 傅里叶变换
virtual Tensor ifft(const Tensor & self, int64_t signal_ndim, bool normalized) const = 0; // 逆傅里叶变换
virtual Tensor rfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided) const = 0;  // 实数傅里叶变换
virtual Tensor irfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided, IntList signal_sizes) const = 0;  // 逆实数傅里叶变换

/* 索引操作 */
// 使用张量列表索引
virtual Tensor index(const Tensor & self, TensorList indices) const = 0;
// 按索引复制数据到指定维度
virtual Tensor & index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const = 0;
// 按索引放置值(可选累加)
virtual Tensor index_put(const Tensor & self, TensorList indices, const Tensor & values, bool accumulate) const = 0;
virtual Tensor & index_put_(Tensor & self, TensorList indices, const Tensor & values, bool accumulate) const = 0;

/* 矩阵求逆 */
virtual Tensor inverse(const Tensor & self) const = 0;  // 矩阵逆

/* 张量属性检查 */
virtual Tensor isclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) const = 0;  // 近似相等比较
virtual bool is_distributed(const Tensor & self) const = 0;  // 是否分布式张量
virtual bool is_floating_point(const Tensor & self) const = 0;  // 是否浮点类型
virtual bool is_complex(const Tensor & self) const = 0;  // 是否复数类型
virtual bool is_nonzero(const Tensor & self) const = 0;  // 是否非零
virtual bool is_same_size(const Tensor & self, const Tensor & other) const = 0;  // 形状是否相同
virtual bool is_signed(const Tensor & self) const = 0;  // 是否有符号类型

/* 顺序统计 */
// 返回第k小的元素及其索引
virtual std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) const = 0;

/* 对数函数 */
virtual Tensor log(const Tensor & self) const = 0;  // 自然对数
virtual Tensor & log_(Tensor & self) const = 0;     // 原地自然对数
virtual Tensor log10(const Tensor & self) const = 0; // 以10为底对数
virtual Tensor & log10_(Tensor & self) const = 0;    // 原地以10为底对数
virtual Tensor log1p(const Tensor & self) const = 0; // log(1+x)
virtual Tensor & log1p_(Tensor & self) const = 0;    // 原地log(1+x)
virtual Tensor log2(const Tensor & self) const = 0;  // 以2为底对数
virtual Tensor & log2_(Tensor & self) const = 0;     // 原地以2为底对数
virtual Tensor logdet(const Tensor & self) const = 0;  // 对数行列式

/* softmax相关 */
// 对数softmax(可指定输出类型)
virtual Tensor log_softmax(const Tensor & self, int64_t dim, ScalarType dtype) const = 0;
// 对数softmax(自动推断类型)
virtual Tensor log_softmax(const Tensor & self, int64_t dim) const = 0;
// 对数求和指数
virtual Tensor logsumexp(const Tensor & self, int64_t dim, bool keepdim) const = 0;

/* 矩阵运算 */
virtual Tensor matmul(const Tensor & self, const Tensor & other) const = 0;  // 矩阵乘法
// 矩阵幂运算
virtual Tensor matrix_power(const Tensor & self, int64_t n) const = 0;

/* 最大值操作 */
// 沿维度返回最大值及其索引
virtual std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim, bool keepdim) const = 0;
// 沿维度返回最大值(不保留索引)
virtual Tensor max_values(const Tensor & self, int64_t dim, bool keepdim) const = 0;

/* 均值计算 */
virtual Tensor mean(const Tensor & self, ScalarType dtype) const = 0;  // 全局均值(指定类型)
virtual Tensor mean(const Tensor & self) const = 0;  // 全局均值(自动推断类型)
// 沿维度计算均值(多种重载)
virtual Tensor mean(const Tensor & self, IntList dim, bool keepdim, ScalarType dtype) const = 0;
virtual Tensor mean(const Tensor & self, IntList dim, bool keepdim) const = 0;
virtual Tensor mean(const Tensor & self, IntList dim, ScalarType dtype) const = 0;

/* 中值计算 */
// 沿维度返回中值及其索引
virtual std::tuple<Tensor,Tensor> median(const Tensor & self, int64_t dim, bool keepdim) const = 0;

/* 最小值操作 */
// 沿维度返回最小值及其索引
virtual std::tuple<Tensor,Tensor> min(const Tensor & self, int64_t dim, bool keepdim) const = 0;
// 沿维度返回最小值(不保留索引)
virtual Tensor min_values(const Tensor & self, int64_t dim, bool keepdim) const = 0;

/* 矩阵乘法 */
virtual Tensor mm(const Tensor & self, const Tensor & mat2) const = 0;  // 矩阵乘法

/* 众数计算 */
// 沿维度返回众数及其计数
virtual std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim, bool keepdim) const = 0;

/* 乘法运算 */
virtual Tensor mul(const Tensor & self, const Tensor & other) const = 0;  // 逐元素乘法
virtual Tensor & mul_(Tensor & self, const Tensor & other) const = 0;    // 原地逐元素乘法
virtual Tensor mul(const Tensor & self, Scalar other) const = 0;         // 标量乘法
virtual Tensor & mul_(Tensor & self, Scalar other) const = 0;           // 原地标量乘法

/* 矩阵-向量乘法 */
virtual Tensor mv(const Tensor & self, const Tensor & vec) const = 0;  // 矩阵×向量

/* 多元对数伽马函数 */
virtual Tensor mvlgamma(const Tensor & self, int64_t p) const = 0;  // 多元对数伽马
virtual Tensor & mvlgamma_(Tensor & self, int64_t p) const = 0;     // 原地多元对数伽马

/* 窄化操作 */
// 返回指定维度的窄化副本
virtual Tensor narrow_copy(const Tensor & self, int64_t dim, int64_t start, int64_t length) const = 0;
// 返回指定维度的窄化视图
virtual Tensor narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length) const = 0;

/* 维度置换 */
virtual Tensor permute(const Tensor & self, IntList dims) const = 0;  // 维度重排列

/* 内存固定 */
virtual Tensor pin_memory(const Tensor & self) const = 0;  // 固定内存(用于CUDA)

/* 伪逆 */
virtual Tensor pinverse(const Tensor & self, double rcond) const = 0;  // 伪逆矩阵

/* 重复操作 */
virtual Tensor repeat(const Tensor & self, IntList repeats) const = 0;  // 沿维度重复

/* 形状重塑 */
virtual Tensor reshape(const Tensor & self, IntList shape) const = 0;  // 改变形状
virtual Tensor reshape_as(const Tensor & self, const Tensor & other) const = 0;  // 改为与other相同形状

/* 取整函数 */
virtual Tensor round(const Tensor & self) const = 0;  // 四舍五入
virtual Tensor & round_(Tensor & self) const = 0;     // 原地四舍五入

/* ReLU激活函数 */
virtual Tensor relu(const Tensor & self) const = 0;  // ReLU激活
virtual Tensor & relu_(Tensor & self) const = 0;     // 原地ReLU激活

/* PReLU激活函数 */
virtual Tensor prelu(const Tensor & self, const Tensor & weight) const = 0;  // 参数化ReLU
// PReLU反向传播
virtual std::tuple<Tensor,Tensor> prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight) const = 0;

/* 硬收缩函数 */
virtual Tensor hardshrink(const Tensor & self, Scalar lambd) const = 0;  // 硬收缩
virtual Tensor hardshrink_backward(const Tensor & grad_out, const Tensor & self, Scalar lambd) const = 0;  // 硬收缩反向传播

/* 平方根倒数 */
virtual Tensor rsqrt(const Tensor & self) const = 0;  // 1/sqrt(x)
virtual Tensor & rsqrt_(Tensor & self) const = 0;     // 原地1/sqrt(x)

/* 选择操作 */
// 选择指定维度的索引处的子张量
virtual Tensor select(const Tensor & self, int64_t dim, int64_t index) const = 0;

/* sigmoid函数 */
virtual Tensor sigmoid(const Tensor & self) const = 0;  // sigmoid激活
virtual Tensor & sigmoid_(Tensor & self) const = 0;     // 原地sigmoid

/* 三角函数 */
virtual Tensor sin(const Tensor & self) const = 0;  // 正弦
virtual Tensor & sin_(Tensor & self) const = 0;     // 原地正弦
virtual Tensor sinh(const Tensor & self) const = 0; // 双曲正弦
virtual Tensor & sinh_(Tensor & self) const = 0;    // 原地双曲正弦

/* 分离计算图 */
virtual Tensor detach(const Tensor & self) const = 0;  // 分离出计算图
virtual Tensor & detach_(Tensor & self) const = 0;     // 原地分离

/* 张量尺寸 */
virtual int64_t size(const Tensor & self, int64_t dim) const = 0;  // 获取指定维度大小

/* 切片操作 */
virtual Tensor slice(const Tensor & self, int64_t dim, int64_t start, int64_t end, int64_t step) const = 0;  // 沿维度切片

/* 行列式计算 */
// 返回行列式的符号和对数值
virtual std::tuple<Tensor,Tensor> slogdet(const Tensor & self) const = 0;

/* 稀疏矩阵乘法 */
virtual Tensor smm(const Tensor & self, const Tensor & mat2) const = 0;  // 稀疏矩阵乘法

/* softmax函数 */
virtual Tensor softmax(const Tensor & self, int64_t dim, ScalarType dtype) const = 0;  // softmax(指定类型)
virtual Tensor softmax(const Tensor & self, int64_t dim) const = 0;  // softmax(自动推断类型)

/* 分割操作 */
// 按固定大小分割张量
virtual std::vector<Tensor> split(const Tensor & self, int64_t split_size, int64_t dim) const = 0;
// 按指定大小列表分割张量
virtual std::vector<Tensor> split_with_sizes(const Tensor & self, IntList split_sizes, int64_t dim) const = 0;

/* 压缩维度 */
virtual Tensor squeeze(const Tensor & self) const = 0;  // 压缩所有长度为1的维度
virtual Tensor squeeze(const Tensor & self, int64_t dim) const = 0;  // 压缩指定维度
virtual Tensor & squeeze_(Tensor & self) const = 0;     // 原地压缩所有维度
virtual Tensor & squeeze_(Tensor & self, int64_t dim) const = 0;  // 原地压缩指定维度

/* 稀疏矩阵乘法 */
virtual Tensor sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const = 0;  // 稀疏矩阵乘加

/* 短时傅里叶变换 */
virtual Tensor stft(const Tensor & self, int64_t n_fft, int64_t hop_length, int64_t win_length, const Tensor & window, bool normalized, bool onesided) const = 0;

/* 步长信息 */
virtual int64_t stride(const Tensor & self, int64_t dim) const = 0;  // 获取指定维度步长

/* 求和运算 */
virtual Tensor sum(const Tensor & self, ScalarType dtype) const = 0;  // 全局求和(指定类型)
virtual Tensor sum(const Tensor & self) const = 0;  // 全局求和(自动推断类型)
// 沿维度求和(多种重载)
virtual Tensor sum(const Tensor & self, IntList dim, bool keepdim, ScalarType dtype) const = 0;
virtual Tensor sum(const Tensor & self, IntList dim, bool keepdim) const = 0;
virtual Tensor sum(const Tensor & self, IntList dim, ScalarType dtype) const = 0;

/* 平方根 */
virtual Tensor sqrt(const Tensor & self) const = 0;  // 平方根
virtual Tensor & sqrt_(Tensor & self) const = 0;     // 原地平方根

/* 标准差计算 */
virtual Tensor std(const Tensor & self, bool unbiased) const = 0;  // 全局标准差
// 沿维度计算标准差
virtual Tensor std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const = 0;

/* 乘积运算 */
virtual Tensor prod(const Tensor & self, ScalarType dtype) const = 0;  // 全局乘积(指定类型)
virtual Tensor prod(const Tensor & self) const = 0;  // 全局乘积(自动推断类型)
// 沿维度计算乘积(多种重载)
virtual Tensor prod(const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype) const = 0;
virtual Tensor prod(const Tensor & self, int64_t dim, bool keepdim) const = 0;
virtual Tensor prod(const Tensor & self, int64_t dim, ScalarType dtype) const = 0;

/* 矩阵转置 */
virtual Tensor t(const Tensor & self) const = 0;  // 矩阵转置(2D张量)
virtual Tensor & t_(Tensor & self) const = 0;     // 原地矩阵转置

/* 三角函数 */
virtual Tensor tan(const Tensor & self) const = 0;  // 正切
virtual Tensor & tan_(Tensor & self) const = 0;     // 原地正切
virtual Tensor tanh(const Tensor & self) const = 0; // 双曲正切
virtual Tensor & tanh_(Tensor & self) const = 0;    // 原地双曲正切

/* 转置操作 */
virtual Tensor transpose(const Tensor & self, int64_t dim0, int64_t dim1) const = 0;  // 交换两个维度
virtual Tensor & transpose_(Tensor & self, int64_t dim0, int64_t dim1) const = 0;     // 原地交换维度

/* 张量翻转 */
virtual Tensor flip(const Tensor & self, IntList dims) const = 0;  // 沿指定维度翻转
virtual Tensor roll(const Tensor & self, IntList shifts, IntList dims) const = 0;  // 沿维度滚动
virtual Tensor rot90(const Tensor & self, int64_t k, IntList dims) const = 0;  // 旋转90度(可指定次数)

/* 截断操作 */
virtual Tensor trunc(const Tensor & self) const = 0;  // 截断小数部分
virtual Tensor & trunc_(Tensor & self) const = 0;     // 原地截断

/* 类型转换 */
virtual Tensor type_as(const Tensor & self, const Tensor & other) const = 0;  // 转换为与other相同类型

/* 增加维度 */
virtual Tensor unsqueeze(const Tensor & self, int64_t dim) const = 0;  // 在指定位置增加长度为1的维度
virtual Tensor & unsqueeze_(Tensor & self, int64_t dim) const = 0;     // 原地增加维度

/* 方差计算 */
virtual Tensor var(const Tensor & self, bool unbiased) const = 0;  // 全局方差
// 沿维度计算方差
virtual Tensor var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const = 0;

/* 视图操作 */
virtual Tensor view_as(const Tensor & self, const Tensor & other) const = 0;  // 改为与other相同形状的视图

/* 条件选择 */
virtual Tensor where(const Tensor & condition, const Tensor & self, const Tensor & other) const = 0;  // 按条件选择元素

/* 范数计算 */
virtual Tensor norm(const Tensor & self, Scalar p) const = 0;  // 计算p-范数
// 沿维度计算p-范数
virtual Tensor norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) const = 0;

/* 克隆操作 */
virtual Tensor clone(const Tensor & self) const = 0;  // 创建副本

/* 调整大小 */
virtual Tensor & resize_as_(Tensor & self, const Tensor & the_template) const = 0;  // 调整为与模板相同大小

/* 幂运算 */
virtual Tensor pow(const Tensor & self, Scalar exponent) const = 0;  // 元素幂运算

/* 清零操作 */
virtual Tensor & zero_(Tensor & self) const = 0;  // 将张量清零

/* 减法运算 */
virtual Tensor sub(const Tensor & self, const Tensor & other, Scalar alpha) const = 0;  // 张量减法
virtual Tensor & sub_(Tensor & self, const Tensor & other, Scalar alpha) const = 0;    // 原地张量减法
virtual Tensor sub(const Tensor & self, Scalar other, Scalar alpha) const = 0;        // 标量减法
virtual Tensor & sub_(Tensor & self, Scalar other, Scalar alpha) const = 0;           // 原地标量减法

/* 矩阵乘加 */
virtual Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const = 0;  // beta*self + alpha*(mat1 @ mat2)
virtual Tensor & addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const = 0;    // 原地矩阵乘加

/* 稀疏张量操作 */
virtual Tensor & sparse_resize_(Tensor & self, IntList size, int64_t sparse_dim, int64_t dense_dim) const = 0;  // 调整稀疏张量大小
virtual Tensor & sparse_resize_and_clear_(Tensor & self, IntList size, int64_t sparse_dim, int64_t dense_dim) const = 0;  // 调整大小并清空
virtual Tensor sparse_mask(const Tensor & self, SparseTensorRef mask) const = 0;  // 应用稀疏掩码
virtual Tensor to_dense(const Tensor & self) const = 0;  // 转换为密集张量

/* 稀疏张量属性 */
virtual int64_t sparse_dim(const Tensor & self) const = 0;  // 稀疏维度数
virtual int64_t _dimI(const Tensor & self) const = 0;       // 索引维度数(内部使用)
virtual int64_t dense_dim(const Tensor & self) const = 0;   // 密集维度数
virtual int64_t _dimV(const Tensor & self) const = 0;       // 值维度数(内部使用)
virtual int64_t _nnz(const Tensor & self) const = 0;        // 非零元素数(内部使用)

/* 稀疏张量处理 */
virtual Tensor coalesce(const Tensor & self) const = 0;  // 合并稀疏索引
virtual bool is_coalesced(const Tensor & self) const = 0;  // 是否已合并
virtual Tensor _indices(const Tensor & self) const = 0;    // 获取索引张量(内部使用)
virtual Tensor _values(const Tensor & self) const = 0;     // 获取值张量(内部使用)
virtual Tensor & _coalesced_(Tensor & self, bool coalesced) const = 0;  // 设置合并状态(内部使用)
virtual Tensor indices(const Tensor & self) const = 0;  // 获取索引张量
virtual Tensor values(const Tensor & self) const = 0;   // 获取值张量

/* 元素总数 */
virtual int64_t numel(const Tensor & self) const = 0;  // 返回张量中元素总数

/* 解绑操作 */
virtual std::vector<Tensor> unbind(const Tensor & self, int64_t dim) const = 0;  // 沿维度解绑为元组

/* 转换为稀疏张量 */
virtual Tensor to_sparse(const Tensor & self, int64_t sparse_dim) const = 0;  // 转换为稀疏张量(指定稀疏维度)
virtual Tensor to_sparse(const Tensor & self) const = 0;  // 转换为稀疏张量(自动推断)

/* 类型/设备转换 */
virtual Tensor to(const Tensor & self, const TensorOptions & options, bool non_blocking, bool copy) const = 0;  // 转换为指定选项
virtual Tensor to(const Tensor & self, Device device, ScalarType dtype, bool non_blocking, bool copy) const = 0;  // 转换为指定设备和类型
virtual Tensor to(const Tensor & self, ScalarType dtype, bool non_blocking, bool copy) const = 0;  // 转换为指定类型
virtual Tensor to(const Tensor & self, const Tensor & other, bool non_blocking, bool copy) const = 0;  // 转换为与other相同类型/设备

/* 标量值提取 */
virtual Scalar item(const Tensor & self) const = 0;  // 提取单元素张量的值

/* 数据指针 */
virtual void* data_ptr(const Tensor & self) const = 0;  // 获取底层数据指针

/* 存储设置 */
virtual Tensor & set_(Tensor & self, Storage source) const = 0;  // 设置存储
virtual Tensor & set_(Tensor & self, Storage source, int64_t storage_offset, IntList size, IntList stride) const = 0;  // 设置存储带偏移和形状
virtual Tensor & set_(Tensor & self, const Tensor & source) const = 0;  // 设置为另一个张量
virtual Tensor & set_(Tensor & self) const = 0;  // 清空张量
virtual bool is_set_to(const Tensor & self, const Tensor & tensor) const = 0;  // 是否与另一个张量共享存储

/* 掩码操作 */
virtual Tensor & masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const = 0;  // 按掩码填充标量值
virtual Tensor & masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const = 0;  // 按掩码填充张量值
virtual Tensor & masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const = 0;  // 按掩码散布源张量

/* 视图操作 */
virtual Tensor view(const Tensor & self, IntList size) const = 0;  // 创建新视图

/* 索引放置 */
virtual Tensor & put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const = 0;  // 按索引放置源张量
virtual Tensor & index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const = 0;  // 沿维度按索引相加
virtual Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const = 0;  // 沿维度按索引填充标量
virtual Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const = 0;  // 
  virtual Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const = 0;
  virtual Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const = 0;
  virtual Tensor & scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const = 0;
  virtual Tensor & lt_(Tensor & self, Scalar other) const = 0;
  virtual Tensor & lt_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & gt_(Tensor & self, Scalar other) const = 0;
  virtual Tensor & gt_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & le_(Tensor & self, Scalar other) const = 0;
  virtual Tensor & le_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & ge_(Tensor & self, Scalar other) const = 0;
  virtual Tensor & ge_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & eq_(Tensor & self, Scalar other) const = 0;
  virtual Tensor & eq_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & ne_(Tensor & self, Scalar other) const = 0;
  virtual Tensor & ne_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor __and__(const Tensor & self, Scalar other) const = 0;
  virtual Tensor __and__(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & __iand__(Tensor & self, Scalar other) const = 0;
  virtual Tensor & __iand__(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor __or__(const Tensor & self, Scalar other) const = 0;
  virtual Tensor __or__(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & __ior__(Tensor & self, Scalar other) const = 0;
  virtual Tensor & __ior__(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor __xor__(const Tensor & self, Scalar other) const = 0;
  virtual Tensor __xor__(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & __ixor__(Tensor & self, Scalar other) const = 0;
  virtual Tensor & __ixor__(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor __lshift__(const Tensor & self, Scalar other) const = 0;
  virtual Tensor __lshift__(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & __ilshift__(Tensor & self, Scalar other) const = 0;
  virtual Tensor & __ilshift__(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor __rshift__(const Tensor & self, Scalar other) const = 0;
  virtual Tensor __rshift__(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & __irshift__(Tensor & self, Scalar other) const = 0;
  virtual Tensor & __irshift__(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & lgamma_(Tensor & self) const = 0;
  virtual Tensor & atan2_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & tril_(Tensor & self, int64_t diagonal) const = 0;
  virtual Tensor & triu_(Tensor & self, int64_t diagonal) const = 0;
  virtual Tensor & digamma_(Tensor & self) const = 0;
  virtual Tensor & polygamma_(Tensor & self, int64_t n) const = 0;
  virtual Tensor & erfinv_(Tensor & self) const = 0;
  virtual Tensor & frac_(Tensor & self) const = 0;
  virtual Tensor & renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const = 0;
  virtual Tensor & reciprocal_(Tensor & self) const = 0;
  virtual Tensor & neg_(Tensor & self) const = 0;
  virtual Tensor & pow_(Tensor & self, Scalar exponent) const = 0;
  virtual Tensor & pow_(Tensor & self, const Tensor & exponent) const = 0;
  virtual Tensor & lerp_(Tensor & self, const Tensor & end, Scalar weight) const = 0;
  virtual Tensor & sign_(Tensor & self) const = 0;
  virtual Tensor & fmod_(Tensor & self, Scalar other) const = 0;
  virtual Tensor & fmod_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & remainder_(Tensor & self, Scalar other) const = 0;
  virtual Tensor & remainder_(Tensor & self, const Tensor & other) const = 0;
  virtual Tensor & addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const = 0;
  virtual Tensor addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const = 0;
  virtual Tensor & addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const = 0;
  virtual Tensor & addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const = 0;
  virtual Tensor & random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const = 0;
  virtual Tensor & random_(Tensor & self, int64_t to, Generator * generator) const = 0;
  virtual Tensor & random_(Tensor & self, Generator * generator) const = 0;
  virtual Tensor & uniform_(Tensor & self, double from, double to, Generator * generator) const = 0;
  virtual Tensor & normal_(Tensor & self, double mean, double std, Generator * generator) const = 0;
  virtual Tensor & cauchy_(Tensor & self, double median, double sigma, Generator * generator) const = 0;
  virtual Tensor & log_normal_(Tensor & self, double mean, double std, Generator * generator) const = 0;
  virtual Tensor & exponential_(Tensor & self, double lambd, Generator * generator) const = 0;
  virtual Tensor & geometric_(Tensor & self, double p, Generator * generator) const = 0;
  virtual Tensor diag(const Tensor & self, int64_t diagonal) const = 0;
  virtual Tensor cross(const Tensor & self, const Tensor & other, int64_t dim) const = 0;
  virtual Tensor triu(const Tensor & self, int64_t diagonal) const = 0;
  virtual Tensor tril(const Tensor & self, int64_t diagonal) const = 0;
  virtual Tensor trace(const Tensor & self) const = 0;
  virtual Tensor ne(const Tensor & self, Scalar other) const = 0;
  virtual Tensor ne(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor eq(const Tensor & self, Scalar other) const = 0;
  virtual Tensor eq(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor ge(const Tensor & self, Scalar other) const = 0;
  virtual Tensor ge(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor le(const Tensor & self, Scalar other) const = 0;
  virtual Tensor le(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor gt(const Tensor & self, Scalar other) const = 0;
  virtual Tensor gt(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor lt(const Tensor & self, Scalar other) const = 0;
  virtual Tensor lt(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor take(const Tensor & self, const Tensor & index) const = 0;
  virtual Tensor index_select(const Tensor & self, int64_t dim, const Tensor & index) const = 0;
  virtual Tensor masked_select(const Tensor & self, const Tensor & mask) const = 0;
  virtual Tensor nonzero(const Tensor & self) const = 0;
  virtual Tensor gather(const Tensor & self, int64_t dim, const Tensor & index) const = 0;
  virtual Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const = 0;
  virtual Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const = 0;
  virtual std::tuple<Tensor,Tensor> gels(const Tensor & self, const Tensor & A) const = 0;
  virtual std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const = 0;
  virtual std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors, bool upper) const = 0;
  virtual std::tuple<Tensor,Tensor> eig(const Tensor & self, bool eigenvectors) const = 0;
  virtual std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self, bool some, bool compute_uv) const = 0;
  virtual Tensor cholesky(const Tensor & self, bool upper) const = 0;
  virtual Tensor potrs(const Tensor & self, const Tensor & input2, bool upper) const = 0;
  virtual Tensor potri(const Tensor & self, bool upper) const = 0;
  virtual std::tuple<Tensor,Tensor> pstrf(const Tensor & self, bool upper, Scalar tol) const = 0;
  virtual std::tuple<Tensor,Tensor> qr(const Tensor & self) const = 0;
  virtual std::tuple<Tensor,Tensor> geqrf(const Tensor & self) const = 0;
  virtual Tensor orgqr(const Tensor & self, const Tensor & input2) const = 0;
  virtual Tensor ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const = 0;
  virtual std::tuple<Tensor,Tensor> btrifact(const Tensor & self, bool pivot) const = 0;
  virtual std::tuple<Tensor,Tensor,Tensor> btrifact_with_info(const Tensor & self, bool pivot) const = 0;
  virtual Tensor btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const = 0;
  virtual Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const = 0;
  virtual Tensor lgamma(const Tensor & self) const = 0;
  virtual Tensor digamma(const Tensor & self) const = 0;
  virtual Tensor polygamma(int64_t n, const Tensor & self) const = 0;
  virtual Tensor erfinv(const Tensor & self) const = 0;
  virtual Tensor frac(const Tensor & self) const = 0;
  virtual Tensor dist(const Tensor & self, const Tensor & other, Scalar p) const = 0;
  virtual Tensor reciprocal(const Tensor & self) const = 0;
  virtual Tensor neg(const Tensor & self) const = 0;
  virtual Tensor atan2(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor lerp(const Tensor & self, const Tensor & end, Scalar weight) const = 0;
  virtual Tensor histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) const = 0;
  virtual Tensor sign(const Tensor & self) const = 0;
  virtual Tensor fmod(const Tensor & self, Scalar other) const = 0;
  virtual Tensor fmod(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor remainder(const Tensor & self, Scalar other) const = 0;
  virtual Tensor remainder(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor min(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor min(const Tensor & self) const = 0;
  virtual Tensor max(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor max(const Tensor & self) const = 0;
  virtual Tensor median(const Tensor & self) const = 0;
  virtual std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim, bool descending) const = 0;
  virtual std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const = 0;
  virtual Tensor all(const Tensor & self) const = 0;
  virtual Tensor any(const Tensor & self) const = 0;
  virtual Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const = 0;
  virtual Tensor unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const = 0;
  virtual bool equal(const Tensor & self, const Tensor & other) const = 0;
  virtual Tensor pow(const Tensor & self, const Tensor & exponent) const = 0;
  virtual Tensor pow(Scalar self, const Tensor & exponent) const = 0;
  virtual Tensor alias(const Tensor & self) const = 0;
protected:
  TensorTypeId type_id_;
  bool is_variable_;
  bool is_undefined_;
};

} // namespace at

#include "ATen/core/Tensor.h"
