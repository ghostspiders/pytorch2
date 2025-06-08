#pragma once

#include <atomic>
#include <memory>
#include <numeric>

#include <c10/core/Backend.h>
#include <c10/core/Storage.h>
#include <ATen/core/TensorOptions.h>
#include <c10/core/TensorTypeId.h>
#include <c10/core/TensorTypeIdRegistration.h>
#include <ATen/core/context_base.h>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/Flags.h>
#include <c10/util/Logging.h>

// 一个全局布尔变量，用于控制当Tensor缩小时是否释放内存。
// 这样做的结果是，Tensor将始终保留其迄今为止重塑后的最大容量所分配的内存。
//
// 这个参数会被"大写字母开头"的方法所遵循（这些方法会调用Resize()，例如CopyFrom, ResizeLike）；
// 但不会被Tensor::resize_或ShrinkTo所遵循，这两个方法保证永远不会释放内存。
C10_DECLARE_bool(caffe2_keep_on_shrink);

// 由于在同一运行过程中不同输入之间分配的blob内存可能有很大差异，
// 只有当内存增益大于这个标志（以字节为单位）时，我们才会缩小blob。
// 这仅适用于遵循caffe2_keep_on_shrink参数的函数。
C10_DECLARE_int64(caffe2_max_keep_on_shrink_memory);

namespace caffe2 {

// 由protobuf定义
class DeviceOption;

}

namespace c10 {
class Scalar;
struct Storage;
}
namespace at {
struct Type;
class Tensor;

/**
 * 一个工具函数，用于将vector<int>转换为vector<int64_t>
 * @param src 输入的整数数组引用
 * @return 转换后的int64_t向量
 */
inline std::vector<int64_t> ToVectorint64_t(ArrayRef<int> src) {
  return std::vector<int64_t>(src.begin(), src.end());
}

/**
 * 计算从第k维开始所有维度的乘积
 * @param k 起始维度索引
 * @param dims 维度列表
 * @return 从k维开始的维度乘积
 */
inline int64_t size_from_dim_(int k, IntList dims) {
  int64_t r = 1;
  for (size_t i = k; i < dims.size(); ++i) {
    r *= dims[i];
  }
  return r;
}

/**
 * 计算到第k维之前(不包括第k维)所有维度的乘积
 * @param k 截止维度索引
 * @param dims 维度列表
 * @return 到k维之前的维度乘积
 */
inline int64_t size_to_dim_(int k, IntList dims) {
  AT_ASSERT((unsigned)k <= dims.size());
  int64_t r = 1;
  for (int i = 0; i < k; ++i) {
    r *= dims[i];
  }
  return r;
}

/**
 * 计算第k维和第l维之间(不包括k和l维)所有维度的乘积
 * @param k 起始维度索引
 * @param l 结束维度索引
 * @param dims 维度列表
 * @return k和l维之间的维度乘积
 */
inline int64_t size_between_dim_(int k, int l, IntList dims) {
  AT_ASSERT((unsigned)l < dims.size());
  int64_t r = 1;
  if (k < l) {
    for (int i = k + 1; i < l; ++i) {
      r *= dims[i];
    }
  } else {
    for (int i = l + 1; i < k; ++i) {
      r *= dims[i];
    }
  }
  return r;
}

/**
 * 规范化轴索引，使其在有效范围内
 * 负值表示从后往前计数，例如-1表示最后一个维度
 * @param axis_index 原始轴索引
 * @param ndims 总维度数
 * @return 规范化后的轴索引
 */
inline int canonical_axis_index_(int axis_index, int ndims) {
  AT_ASSERT(axis_index >= -ndims);
  AT_ASSERT(axis_index < ndims);
  if (axis_index < 0) {
    return axis_index + ndims;
  }
  return axis_index;
}

// 内存释放函数指针类型
// 参数：
//   void* - 要释放的内存指针
//   size_t - 要释放的内存大小
using PlacementDtor = void (*)(void*, size_t);

/*
 * 一个在析构时调用额外placement删除器的上下文类
 *
 * 该类接受一个已构造的DataPtr并作为成员存储，在析构时，
 * 会在底层数据指针被DataPtr析构前调用额外的删除器。
 * `data_ptr_` 拥有内存的所有权。
 */
struct CAFFE2_API PlacementDeleteContext {
  at::DataPtr data_ptr_;        // 存储的数据指针，拥有内存所有权
  PlacementDtor placement_dtor_; // 额外的placement删除器函数指针
  size_t size_;                 // 要删除的内存大小

  // 构造函数
  // 参数：
  //   data_ptr: 要管理的数据指针（右值引用，所有权转移）
  //   placement_dtor: 自定义删除器函数
  //   size: 内存大小
  PlacementDeleteContext(
      at::DataPtr&& data_ptr,
      PlacementDtor placement_dtor,
      size_t size)
      : data_ptr_(std::move(data_ptr)),  // 转移所有权
        placement_dtor_(placement_dtor),
        size_(size) {}

  // 静态工厂方法，创建带有自定义删除器的DataPtr
  // 参数：
  //   data_ptr: 原始数据指针
  //   placement_dtor: 自定义删除器
  //   size: 内存大小
  //   device: 设备信息
  // 返回值：
  //   包装后的DataPtr，带有自定义删除逻辑
  static at::DataPtr makeDataPtr(
      at::DataPtr&& data_ptr,
      PlacementDtor placement_dtor,
      size_t size,
      at::Device device);

  // 析构函数
  ~PlacementDeleteContext() {
    // 先调用自定义删除器清理资源
    placement_dtor_(data_ptr_.get(), size_);
    // data_ptr_析构时会自动释放原始内存
  }
};

namespace detail {
  // 这是一个集中确定张量对应TensorTypeId的工具函数
  
  /*
   * 根据TensorOptions计算对应的TensorTypeId
   * 参数：
   *   options - 包含设备类型、布局等信息的张量选项
   * 返回值：
   *   对应的TensorTypeId
   * 注意：
   *   1. 这里使用TensorOptions而非单独的DeviceType和Layout，因为未来可能根据TensorOptions的任何属性进行分发
   *   2. 如果修改此函数逻辑，需要同步更新caffe2/tensor.h中的computeTensorTypeId调用
   */
  inline TensorTypeId computeTensorTypeId(TensorOptions options) {
    switch (options.layout()) {
      case Layout::Strided:  // 密集存储布局
        switch (options.device().type()) {
          case DeviceType::CPU:    // CPU设备
            return CPUTensorId();
          case DeviceType::CUDA:   // CUDA设备
            return CUDATensorId();
          case DeviceType::MKLDNN: // MKLDNN加速
            return MKLDNNTensorId();
          case DeviceType::OPENGL: // OpenGL
            return OpenGLTensorId();
          case DeviceType::OPENCL: // OpenCL
            return OpenCLTensorId();
          case DeviceType::IDEEP:  // IDEEP加速
            return IDEEPTensorId();
          case DeviceType::HIP:     // HIP(ROCm)
            return HIPTensorId();
          default:
            AT_ERROR("不支持该设备类型的密集布局: ", options.device().type());
        }
      case Layout::Sparse:  // 稀疏存储布局
        switch (options.device().type()) {
          case DeviceType::CPU:    // CPU设备
            return SparseCPUTensorId();
          case DeviceType::CUDA:   // CUDA设备
            return SparseCUDATensorId();
          case DeviceType::HIP:    // HIP(ROCm) 
            return SparseHIPTensorId();
          default:
            AT_ERROR("不支持该设备类型的稀疏布局: ", options.device().type());
        }
      default:
        AT_ERROR("不支持的布局类型: ", options.layout());
    }
  }

  /*
   * 根据TensorTypeId反向计算对应的DeviceType
   * 参数：
   *   tid - 张量类型ID
   * 返回值：
   *   对应的设备类型
   */
  inline DeviceType computeDeviceType(TensorTypeId tid) {
    if (tid == CPUTensorId()) {           // CPU张量
      return DeviceType::CPU;
    } else if (tid == CUDATensorId()) {   // CUDA张量
      return DeviceType::CUDA;
    } else if (tid == HIPTensorId()) {    // HIP张量
      return DeviceType::HIP;
    } else if (tid == MKLDNNTensorId()) { // MKLDNN张量
      return DeviceType::MKLDNN;
    } else if (tid == OpenGLTensorId()) {  // OpenGL张量
      return DeviceType::IDEEP;
    } else if (tid == OpenCLTensorId()) {  // OpenCL张量
      return DeviceType::OPENCL;
    } else if (tid == IDEEPTensorId()) {   // IDEEP张量
      return DeviceType::IDEEP;
    } else if (tid == SparseCPUTensorId()) { // 稀疏CPU张量
      return DeviceType::CPU;
    } else if (tid == SparseCUDATensorId()) { // 稀疏CUDA张量
      return DeviceType::CUDA;
    } else if (tid == SparseHIPTensorId()) {  // 稀疏HIP张量
      return DeviceType::HIP;
    } else {
      AT_ASSERTM(false, "未知的TensorTypeId: ", tid);
    }
  }
} // namespace detail

/**
 * 张量的底层表示结构，包含指向存储对象(Storage/StorageImpl)的指针
 * 以及描述该数据视图的元数据(如尺寸和步长)。
 *
 * 关于张量内存表示的基本特征：
 *
 *  - 包含指向存储结构(Storage/StorageImpl)的指针，该结构记录了
 *    实际数据的指针、数据类型和设备信息。这使得多个张量可以共享
 *    同一底层数据，从而高效实现张量的不同视图。
 *
 *  - 张量结构本身记录了视图特定的元数据，如尺寸、步长和在存储中的偏移量。
 *    存储的每个视图可以有不同的尺寸或偏移量。
 *
 *  - 该类采用侵入式引用计数。使用引用计数是为了支持及时释放大张量；
 *    采用侵入式引用计数是为了能对原始指针进行引用计数操作，这在跨语言边界
 *    传递张量时通常更为方便。
 *
 *  - 出于向后兼容的原因，张量可能处于未初始化状态。张量可能在以下两种情况下
 *    未初始化：
 *
 *      - 数据类型(DTYPE)未初始化：这类张量的数据类型未初始化。
 *        这种情况最常见于用户编写 Tensor x(CPU) 时。数据类型会在首次调用
 *        mutable_data<T>() 时初始化。
 *
 *      - 存储(STORAGE)未初始化：这类张量具有非零尺寸，但存储的数据指针为空。
 *        这种情况最常见于用户调用 Resize() 或 FreeMemory() 时。
 *        因为Caffe2传统上采用延迟分配策略：数据的实际分配会延迟到首次调用
 *        mutable_data<T>() 时才进行。零尺寸张量总是存储初始化的，
 *        因为这种情况下不需要分配内存。
 *
 *    这两种未初始化状态的所有组合都是可能的。考虑以下典型的Caffe2 API操作序列：
 *
 *      Tensor x(CPU); // x是存储初始化的，数据类型未初始化
 *      x.Resize(4); // x变为存储未初始化，数据类型未初始化
 *      x.mutable_data<float>(); // x变为存储初始化，数据类型初始化
 *      x.FreeMemory(); // x变为存储未初始化，但数据类型保持初始化
 *
 *    张量的所有其他字段总是初始化的。特别是尺寸总是有效的。
 *    (历史上，声明为 Tensor x(CPU) 的张量尺寸也未初始化，编码为 numel == -1，
 *    但现在我们默认零尺寸，即 numel == 0)。
 *
 *    未初始化的存储必须是唯一拥有的，以保持模型简单。因此，我们将拒绝可能导致
 *    未初始化存储变为共享的操作(或导致共享存储变为未初始化的操作，例如 FreeMemory)。
 *
 *    在实践中，存储未初始化且数据类型未初始化的张量存在时间极短：基本上，
 *    在调用 Resize() 后，几乎总是立即调用 mutable_data()。
 *    大多数函数设计时并未考虑处理存储未初始化且数据类型未初始化的张量。
 *
 *    我们计划消除所有未初始化状态，使每个张量的所有字段都完全初始化。
 *    请不要编写依赖这些未初始化状态的新代码。
 */
struct CAFFE2_API TensorImpl : public c10::intrusive_ptr_target {
// 禁用默认构造函数
TensorImpl() = delete;

/**
 * 构造一个1维0尺寸的张量，使用给定的参数配置
 * @param type_id 张量类型ID (CPU/CUDA等)
 * @param data_type 数据类型元信息
 * @param allocator 内存分配器，将在后续resize时用于分配数据 
 * @param is_variable 标记是否为可变张量(用于自动微分)
 */
TensorImpl(TensorTypeId type_id, const caffe2::TypeMeta& data_type, Allocator* allocator, bool is_variable);

/**
 * 构造一个1维0尺寸的张量，直接使用给定的存储对象
 * @param storage 已分配的存储对象(移动语义转移所有权)
 * @param type_id 张量类型ID
 * @param is_variable 标记是否为可变张量
 */
TensorImpl(Storage&& storage, TensorTypeId type_id, bool is_variable);

private:
/**
 * 私有构造函数 - 因为data_type实际上可以从storage中获取
 * 单独传入data_type是为了简化初始化列表编写
 * (避免storage被移出导致的问题)
 */
TensorImpl(Storage&& storage, TensorTypeId type_id, 
          const caffe2::TypeMeta& data_type, bool is_variable);

public:
// 禁用拷贝构造和拷贝赋值(因为引用计数的特殊性)
TensorImpl(const TensorImpl&) = delete;
TensorImpl& operator=(const TensorImpl&) = delete;

// 允许移动构造和移动赋值
TensorImpl(TensorImpl&&) = default;
TensorImpl& operator=(TensorImpl&&) = default;

/**
 * 释放资源(减少引用计数)和任何外部分配
 * 重写自intrusive_ptr_target，用于实现弱引用张量
 */
virtual void release_resources() override;

/**
 * 理想情况下，type_id() 应该是我们进行调度分发时唯一需要查询的键值，
 * 而不需要额外检查三个不同的变量。以下是当前存在的障碍：
 *
 *  - 要消除 ScalarType 依赖，我们需要为每个 ScalarType+Backend 组合
 *    分配一个独立的 TensorTypeId，并在初始化 TensorImpl 时正确设置。
 *
 *  - 要消除 is_variable 标志，我们需要定义两类 TensorTypeId：
 *    一类对应变量(Variable)，另一类对应普通张量。
 *    短期可能不会消除此标志，因为将变量状态硬编码到 type_id() 中
 *    会使得实现"线程局部 no_grad"技巧更加困难
 *    (该技巧通过设置线程局部变量将变量当作普通张量处理)。
 *
 * TODO: type() 作为方法名非常直观，但我们实际上不希望用户直接使用它。
 * 考虑重命名为其他名称。
 */

/**
 * 返回该张量对应的 TensorTypeId。未来这将作为操作符分发的唯一依据，
 * 但目前尚未用于实际调度。
 *
 * 注意：type_id() 和 type() 不是一一对应的关系。例如：
 * - 所有 CPU 张量共享同一个 type_id()
 * - 但存在多种 Type (CPUFloatTensor、CPUDoubleTensor 等)
 */ 


// 返回张量的类型ID (TensorTypeId)
TensorTypeId type_id() const { return type_id_; }

/**
 * 返回该张量尺寸的引用。只要张量存在且未被重新调整大小，
 * 该引用保持有效。
 */
virtual IntList sizes() const;

/**
 * 返回该张量步长(stride)的引用。只要张量存在且未重新调整步长，
 * 该引用保持有效。 
 */
virtual IntList strides() const;

/**
 * 返回该张量的维度数量。注意：
 * - 0维张量表示标量(Scalar)
 * - 1维张量表示向量
 */
virtual int64_t dim() const;

/**
 * 返回张量的底层存储(Storage)。多个张量可能共享同一个存储。
 * Storage是一个功能受限的类，支持的操作远少于Tensor。
 *
 * 警告：尽量避免直接使用此方法，应优先使用Tensor API进行操作。
 */
virtual const Storage& storage() const;

// TODO: 待移除的友元声明  
friend struct Type;

/**
 * 返回张量中元素的总数。
 *
 * 警告：在旧版Caffe2 API中，可通过numel() == -1判断张量是否未初始化。
 * 此行为已改变，现在numel()始终准确返回张量各维度大小的乘积。
 */
virtual int64_t numel() const {
#ifdef DEBUG  // 调试模式下验证计算结果
    AT_ASSERT(compute_numel() == numel_);
#endif
    return numel_;
}

/**
 * 判断张量是否在内存中连续存储。
 *
 * 具有非平凡步长(stride)的张量是不连续的。
 * 具体定义见compute_contiguous()函数。
 */
virtual bool is_contiguous() const {
#ifdef DEBUG  // 调试模式下验证计算结果
    AT_ASSERT(compute_contiguous() == is_contiguous_);
#endif
    return is_contiguous_;
}

// 判断是否为稀疏张量（非虚函数，出于性能考虑避免动态分派）
bool is_sparse() const {
    auto tid = type_id();
    // 注意：当前Variable与对应张量共享TensorTypeId
    return tid == SparseCPUTensorId() || 
           tid == SparseCUDATensorId() || 
           tid == SparseHIPTensorId();
}

// 判断是否位于CUDA设备（非虚函数，性能优化）
bool is_cuda() const {
    auto tid = type_id();
    // 注意：当前Variable与对应张量共享TensorTypeId  
    return tid == CUDATensorId() || tid == SparseCUDATensorId();
}

// 判断是否位于HIP设备（非虚函数，性能优化）  
bool is_hip() const {
    auto tid = type_id();
    // 注意：当前Variable与对应张量共享TensorTypeId
    return tid == HIPTensorId() || tid == SparseHIPTensorId();
}

// 获取设备索引（非虚函数，常见情况避免动态分派）
int64_t get_device() const {
    const auto tid = type_id();
    if (tid == CUDATensorId() || tid == HIPTensorId()) {
        // TODO: #12934 考虑缓存设备信息以避免虚函数调用
        return storage().device().index();
    }
    return get_device_slow();  // 慢速路径
}

 // 返回张量所在的设备对象
Device device() const {
    // 为性能考虑特殊处理常见情况
    // TODO: 当前实现有些复杂，考虑在TensorImpl缓存设备信息(#12934)以加速所有情况下的device()调用
    const auto tid = type_id();
    if (tid == CPUTensorId() || tid == CUDATensorId() || tid == HIPTensorId()) {
      // 注意：使用storage()而非storage_，因为要考虑Variable的情况
      const auto& mystorage = storage();
      if (mystorage) {
        return mystorage.device();
      }
    }
    const auto device_type = detail::computeDeviceType(tid);
    bool not_cpu = device_type != DeviceType::CPU;
    return Device(device_type, not_cpu ? get_device() : -1);
}

// 返回张量的内存布局类型（非虚函数，性能优化）
Layout layout() const {
    if (is_sparse()) {
      return kSparse;  // 稀疏布局
    } else {
      return kStrided; // 密集(跨步)布局
    }
}

/**
 * 如果condition_when_zero_dim为true且张量是1维1元素张量，
 * 则将其重塑为0维张量(标量)。
 *
 * 此辅助函数由生成的包装代码调用，用于"修正"传统代码生成的不正确形状。
 * 例如，假设有一个传统函数'add'生成的张量与输入形状相同；
 * 但当输入是0维时，它生成的是1维1元素张量。
 * 调用result->maybe_zero_dim(lhs->dim() == 0 && rhs->dim() == 0) 
 * 可以在输入为0维时正确将维度重置为0。
 *
 * 随着TH越来越多地正确处理0维情况，此函数将变得不那么必要。
 * 目前它常被已正确处理0维情况的函数调用，此时只是死代码。
 * 在美好的未来，此函数将被完全移除。
 */
virtual TensorImpl* maybe_zero_dim(bool condition_when_zero_dim);

/**
 * 判断张量是否是从C++或Python数字自动包装而来。
 * 例如，当写't + 2'时，2会自动包装为设置了`is_wrapped_number_`的Tensor。
 *
 * 在混合类型操作中，如果有任何非包装数字的Tensor，
 * 包装数字不参与结果类型计算。这很有用，因为我们希望't + 2'能
 * 与任何类型的张量一起工作，而不仅仅是LongTensor(这是Python整数表示的默认类型)。
 *
 * 其他情况下，它们的行为与非包装等价物相同。
 * 参见TensorIterator.h中的[结果类型计算]。
 *
 * 为什么选择包装数字方案，而不是额外提供add(Tensor, Scalar)函数？
 * 这大大减少了我们需要为add操作编写的代码量，因为Tensor-Scalar加法
 * 实际上就是当RHS是0维时的Tensor-Tensor加法(除了类型提升行为)。
 *
 * 警告：不能在Variable上调用此方法。
 * 参见注释[我们后悔让Variable持有Tensor]
 */
bool is_wrapped_number() const {
    AT_ASSERT(!is_variable());
    return is_wrapped_number_;
}

/**
 * 设置张量是否是从C++或Python数字自动包装而来。
 * 除非你在编写绑定代码，否则可能不需要调用此方法。
 *
 * 警告：不能在Variable上调用此方法。
 * 参见注释[我们后悔让Variable持有Tensor]
 */
void set_wrapped_number(bool value) {
    AT_ASSERT(!is_variable());
    AT_ASSERT(dim() == 0);  // 只有0维张量才能是包装数字
    is_wrapped_number_ = value;
}

// ~~~~~ 自动微分API ~~~~~
// 部分方法在TensorImpl.cpp中定义，因为Tensor是不完整类型
//
// 注释[Tensor与Variable在C++中的区别]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 自动微分方法仅对Variable::Impl子类有效。
// 这是由于一些有问题的设计选择：Variable包含一个Tensor(所以它们不是同一事物)，
// 但Variable又是Tensor的子类(这样可以在Tensor上编写同时适用于Variable和Tensor的代码)。
// Variable不满足Tensor的里氏替换原则；通常你要么全部使用Variable，
// 要么全部使用Tensor，而不是混合使用。我们计划在未来修复这个问题。
//
// 注释[我们后悔让Variable持有Tensor]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Tensor有许多字段。这些字段总是有效的吗？不一定：
// Variable::Impl子类不使用这些字段，而是将它们"转发"到内部持有的'tensor'。
// 它甚至不保持外部Tensor字段更新，因为最终用户可能获取内部tensor并直接调整大小
// (使任何我们跟踪的外部字段过时)。
//
// 可以想象，这是一个非常糟糕的状态。它使得在TensorImpl上实现一切变得复杂：
// 如果直接访问TensorImpl的字段，必须将其"虚化"，以使其在Variable上正确工作
// (因为需要重写方法以避免查看我们的字段，而是查看数据tensor的字段)。
// 任何没有虚化的内容，在Variable上调用时都不会工作。
//
// 解决方法是让Variable::Impl不再持有tensor，而是直接"成为"一个tensor。

/**
 * 设置张量是否需要梯度。
 * 只能在Variable上调用此方法。
 * 参见注释[Tensor与Variable在C++中的区别]。
 */
virtual void set_requires_grad(bool requires_grad) {
    AT_ERROR("Tensor未实现set_requires_grad");
}

/**
 * 判断张量是否需要梯度。需要梯度的张量会跟踪在其上执行的操作历史，
 * 以便我们可以自动微分回它们。需要梯度但没有历史的张量是"叶子"张量，
 * 我们会将梯度累积到其中。
 *
 * 只能在Variable上调用此方法。
 * 参见注释[Tensor与Variable在C++中的区别]。
 */
virtual bool requires_grad() const {
    AT_ERROR("Tensor未实现requires_grad");
}

/**
 * 返回梯度的可变引用。通常用作`t.grad() = x`来设置全新的梯度。
 *
 * 只能在Variable上调用此方法。
 * 参见注释[Tensor与Variable在C++中的区别]。
 */
virtual Tensor& grad();

/**
 * 返回张量的累积梯度。当此张量是叶子张量时，在反向传播时会写入此梯度。
 *
 * 只能在Variable上调用此方法。
 * 参见注释[Tensor与Variable在C++中的区别]。
 */
virtual const Tensor& grad() const;

/**
 * 返回指向张量实际数据的类型化指针。这会检查请求的类型(来自模板参数)
 * 是否与张量的内部类型匹配。
 *
 * 不能在数据类型未初始化的张量上调用data()，即使大小为0。
 *
 * 警告：如果张量不连续，执行索引计算时必须使用步长来确定元素位置。
 * 建议使用'TensorAccessor'来处理这些计算。
 *
 * 警告：不能在Variable上调用此方法。
 * 参见注释[我们后悔让Variable持有Tensor]
 */
template <typename T>
inline T* data() const {
    AT_ASSERT(!is_variable());
    AT_ASSERTM(storage_initialized(),
        "张量元素数非零但数据尚未分配。Caffe2使用延迟分配，"
        "需要调用mutable_data()或raw_mutable_data()来实际分配内存。");
    AT_ASSERTM(storage_.IsType<T>(),
        "张量类型不匹配，调用者期望元素类型为",
        caffe2::TypeMeta::TypeName<T>(), "，而张量包含",
        data_type_.name(), "。");
    return storage_.unsafe_data<T>() + storage_offset_;
}

/**
 * 返回指向张量实际数据的void*指针。
 *
 * 不能在数据类型未初始化的张量上调用，即使大小为0。
 *
 * 警告：此指针指向的数据可能不连续；不要假设itemsize() * numel()
 * 足以计算从此张量可有效读取的字节数。
 *
 * 警告：不能在Variable上调用此方法。
 * 参见注释[我们后悔让Variable持有Tensor]
 */
inline void* data() const {
    AT_ASSERT(!is_variable());
    AT_ASSERT(storage_initialized());
    AT_ASSERT(dtype_initialized());
    return static_cast<void*>(
        static_cast<char*>(storage_.data()) +
        data_type_.itemsize() * storage_offset_);
}

/**
 * 类似data()，但适用于Variable。
 * Variable和Tensor合并后此函数将消失。
 * 参见注释[我们后悔让Variable持有Tensor]
 */
virtual void* slow_data() const {
    return data();
}

/**
 * 类似data<T>()，但不进行检查。调用者需确保满足data()的所有要求。
 *
 * 警告：不能在Variable上调用此方法。
 * 参见注释[我们后悔让Variable持有Tensor]
 */
template <typename T>
inline T* unsafe_data() const {
    return storage_.unsafe_data<T>() + storage_offset_;
}

/**
 * 返回张量的TypeMeta，描述其数据类型(如int、float等)
 */
const caffe2::TypeMeta& dtype() const {
    return data_type_;
}

/**
 * 返回此张量单个元素的大小(字节)
 */
size_t itemsize() const {
    AT_ASSERT(dtype_initialized());
    return data_type_.itemsize();
}

/**
 * 返回此张量在存储中的元素偏移量。大多数张量的storage_offset()为0，
 * 但例如张量的索引会有非零的storage_offset()。
 *
 * 警告：这不是以字节为单位计算的。
 *
 * XXX：唯一阻止此函数成为虚函数的是Variable。
 */
virtual int64_t storage_offset() const {
    return storage_offset_;
}

/**
 * 判断张量是否为空(即numel() == 0)
 */
inline bool is_empty() const {
    return numel() == 0;
}

/**
 * 改变张量的维度。这是真正的resize：
 * 如果旧尺寸仍然有效，将被保留(某些调用点利用此不变性，
 * 例如squeeze的实现主要希望尺寸保持不变)。
 * 新维度被赋予0尺寸和0步长；这可能不是你想要的——
 * 之后应该调用set_size/set_stride。
 *
 * TODO：应该用更不易误用的`set_sizes_and_strides`替代此方法
 */
virtual void resize_dim(int64_t ndim) {
    sizes_.resize(ndim, 0);
    strides_.resize(ndim, 0);
    refresh_numel();
    refresh_contiguous();
}

/**
 * 改变某维度的尺寸。这不会更新步长；
 * 因此，大多数尺寸变化不会保持连续性。调用此方法时可能还需要调用set_stride()。
 *
 * TODO：应该用更不易误用的`set_sizes_and_strides`替代此方法
 */
virtual void set_size(int64_t dim, int64_t new_size) {
    sizes_.at(dim) = new_size;
    refresh_numel();
    refresh_contiguous();
}

/**
 * 改变某维度的步长。
 *
 * TODO：应该用更不易误用的`set_sizes_and_strides`替代此方法
 */
virtual void set_stride(int64_t dim, int64_t new_stride) {
    strides_[dim] = new_stride;
    refresh_numel();
    refresh_contiguous();
}

/**
 * 设置此张量在存储中的偏移量。
 *
 * 警告：这不检查张量对于存储中新位置是否越界；
 * 调用者需负责检查(必要时调整大小)
 */
virtual void set_storage_offset(int64_t storage_offset) {
    storage_offset_ = storage_offset;
}

/**
 * 类似set_sizes_and_strides但假设步长是连续的。
 *
 * 警告：此函数不检查请求的尺寸/步长是否在已分配存储的范围内；
 * 这是调用者的责任。
 *
 * 警告：不能在Variable上调用此方法。
 * 参见注释[我们后悔让Variable持有Tensor]
 */
void set_sizes_contiguous(at::IntList new_size) {
    AT_ASSERT(!is_variable());
    auto old_dim = sizes_.size();
    auto new_dim = new_size.size();

    sizes_.resize(new_dim);
    for (size_t dim = 0; dim < new_dim; ++dim) {
        sizes_[dim] = new_size[dim];
    }

    update_to_contiguous_strides(old_dim);
    refresh_numel();
}

/**
 * 设置张量的尺寸和步长。
 *
 * 警告：此函数不检查请求的尺寸/步长是否在已分配存储的范围内；
 * 这是调用者的责任。
 *
 * 警告：不能在Variable上调用此方法。
 * 参见注释[我们后悔让Variable持有Tensor]
 */
void set_sizes_and_strides(at::IntList new_size, at::IntList new_stride) {
    AT_ASSERT(!is_variable());
    AT_CHECK(new_size.size() == new_stride.size(),
        "尺寸维度(", new_size.size(), ")必须与步长维度(", 
        new_stride.size(), ")匹配");
    auto new_dim = new_size.size();

    sizes_.resize(new_dim);
    for (size_t dim = 0; dim < new_dim; ++dim) {
        sizes_[dim] = new_size[dim];
    }

    strides_.resize(new_dim);
    if (new_dim > 0) {
        for (size_t dim = new_dim - 1; ; dim--) {
            if (new_stride[dim] >= 0) {
                strides_[dim] = new_stride[dim];
            } else {
                // XXX：此行为令人意外，可能需要移除以支持负步长。
                // 一些pytorch函数依赖于此行为：例如torch.cat
                if (dim == new_dim - 1) {
                    strides_[dim] = 1;
                } else {
                    // 保持步长单调递增以匹配NumPy
                    strides_[dim] = std::max<int64_t>(sizes_[dim + 1], 1) * strides_[dim + 1];
                }
            }
            if (dim == 0) break;
        }
    }

    refresh_numel();
    refresh_contiguous();
}

/**
 * 返回张量某维度的尺寸
 */
virtual int64_t size(int64_t d) const;

/**
 * 返回张量某维度的步长
 */
virtual int64_t stride(int64_t d) const;

/**
 * 判断张量是否是Variable。
 * 参见注释[Tensor与Variable在C++中的区别]
 */
bool is_variable() const { return is_variable_; };

private:
// 作为优化，get_device处理典型的CUDA张量情况，如果张量将设备存储在其他地方
// (VariableImpl, SparseTensorImpl)，则调用get_device_slow。此方法执行虚函数
// 调用，使其比特殊处理的CUDA张量情况慢10-20纳秒。
virtual int64_t get_device_slow() const {
    AT_ERROR("get_device未实现于具有", 
        toString(tensorTypeIdToBackend(type_id())), "后端的张量");
}

public:
/**
 * 张量的设备类型，如DeviceType::CPU或DeviceType::CUDA
 */
at::DeviceType device_type() const {
    AT_ASSERT(!is_variable());
    return storage_.device_type();
}

/**
 * 张量的设备；例如Device(at::kCUDA, 1)(索引为1的CUDA设备)
 */
at::Device GetDevice() const {
    return storage_.device();
}

/**
 * @brief 从源张量复制数据，提供上下文来执行底层memcpy操作。
 * 此方法尊重caffe2_keep_on_shrink标志。
 *
 * CopyFrom后，目标张量保证与src具有相同的初始化状态和数据类型。
 * 此方法保留源张量的设备类型(例如，如果在CPU上分配张量然后从CUDA张量
 * CopyFrom，将执行CUDA到CPU的传输)。
 *
 * 'async'参数对CUDA张量触发异步复制
 */
void CopyFrom(const TensorImpl& src, bool async = false) {
    AT_ASSERT(!is_variable());
    AT_ASSERTM(src.is_contiguous(),
        "目前仅支持复制连续存储的源张量");
    AT_ASSERTM(src.storage_initialized(),
        "不能从未初始化的张量复制");

    if ((void*)&src == (void*)this) return;  // 自我复制

    // 测试是否需要分配新存储
    // 未初始化的存储保证唯一拥有，所以这种情况下不需要交换
    if (storage_initialized()) {
        // 如果数据类型改变，需要重新分配存储
        if (data_type_ != src.dtype()) {
            // 注意：复制保留device_type
            // 此存储将由下面的mutable_data调用初始化
            storage_ = at::Storage(device_type(), src.dtype());
        }
    }
    data_type_ = src.dtype();
    Resize(src.sizes());

    if (numel() > 0) {
        if (data_type_.copy()) {  // 非POD类型
            AT_ASSERTM(device_type() == ::at::DeviceType::CPU,
                "非POD类型复制时源和目标张量必须都在CPU上，但目标在",
                device_type());
            AT_ASSERTM(src.device_type() == ::at::DeviceType::CPU,
                "非POD类型复制时源和目标张量必须都在CPU上，但源在",
                src.device_type());
            data_type_.copy()(src.data(), raw_mutable_data(data_type_), numel());
        } else {  // POD类型
            // 以下复制使用当前(线程局部)流进行复制，并从传入的device()字段获取GPU id
            //
            // TODO：可能需要更多强制措施来避免当前设置设备错误时意外切换到同步复制
            //
            // 具体来说，可能需要显式切换到不同的上下文设备，
            // 以避免依赖用户正确同步事物
            //
            // 注意：raw_mutable_data在此初始化设备
            void* new_data = raw_mutable_data(data_type_);
            at::CopyBytes(numel() * itemsize(),
                src.data(), src.device(),
                new_data, device(), async);
        }
    }
}

/**
 * @brief 扩展张量的最外层维度，保留现有数据。
 *
 * 底层数据可能会重新分配以容纳新元素，此时张量的容量会按growthPct比例增长。
 * 这确保Extend的摊销时间复杂度为O(1)。
 *
 * 如果底层设备(CUDA)支持，此操作会自动异步执行。
 */
void Extend(int64_t num, float growthPct) {
    AT_ASSERT(sizes_.size() >= 1u);
    AT_ASSERTM(num >= 0, "Extend的`num`必须非负");
    AT_ASSERTM(is_contiguous_,
        "目前Extend仅支持连续存储的张量");
    auto newDims = sizes_;
    newDims[0] += num;
    if (!storage_.data()) {  // 未初始化存储
        Resize(newDims);
        return;
    }
    auto newNumel = std::accumulate(newDims.begin(), newDims.end(),
        static_cast<int64_t>(1), std::multiplies<int64_t>());
    if (newNumel * storage_.itemsize() <= storage_.capacity()) {  // 容量足够
        sizes_ = newDims;
        numel_ = newNumel;
        return;
    }
    // 需要扩容
    auto newCapacity = sizes_;
    newCapacity[0] = std::max<size_t>(
        newDims[0], std::ceil(sizes_[0] * (growthPct + 100) / 100));
    auto oldData = std::move(storage_.data_ptr());
    auto oldSize = numel_;
    auto oldDims = sizes_;
    Resize(newCapacity);
    auto* newData = raw_mutable_data(data_type_);
    if (data_type_.copy()) {  // 非POD类型
        AT_ASSERTM(device_type() == ::at::DeviceType::CPU,
            "非POD类型仅能在CPU上工作");
        data_type_.copy()(oldData.get(), newData, oldSize);
    } else {  // POD类型
        // 使用当前(线程局部)流进行异步复制
        at::CopyBytes(oldSize * itemsize(),
            oldData.get(), device(),
            newData, device(), true); // 非阻塞
    }
    reserved_ = true;
    sizes_ = newDims;
    numel_ = newNumel;
}

/**
 * @brief 为底层张量预留空间。
 *
 * 必须在Resize()之后调用，因为我们只指定第一个维度。
 * 这不会将旧数据复制到新分配的空间。
 */
template <class T>
void ReserveSpace(const T& outer_dim) {
    AT_ASSERTM(is_contiguous_,
        "目前ReserveSpace仅支持连续存储的张量");
    AT_ASSERTM(storage_.unique(), "不能在共享存储上调用ReserveSpace");
    auto newCapacity = sizes_;
    newCapacity[0] = outer_dim;
    auto newNumel = std::accumulate(newCapacity.begin(), newCapacity.end(),
        static_cast<int64_t>(1), std::multiplies<int64_t>());
    if (newNumel * storage_.itemsize() <= storage_.capacity()) return;  // 容量足够
    // 需要扩容，丢弃旧数据
    storage_.data_ptr().clear();
    auto oldSize = numel_;
    auto oldDims = sizes_;
    Resize(newCapacity);
    // 分配新内存但不复制数据
    raw_mutable_data(data_type_);
    sizes_ = oldDims;
    numel_ = oldSize;
    reserved_ = true;
}

/**
 * @brief 调整张量尺寸。
 *
 * Resize接受指定张量维度的int向量。可以传入空向量表示它是标量
 * (即包含单个元素)。
 *
 * 调用Resize后底层存储可能被删除：如果新形状导致张量元素数量变化，
 * 旧内存会被删除，下次调用mutable_data()时将分配新内存。但如果形状不同
 * 但总元素数相同，则保留底层存储。
 *
 * 此方法尊重caffe2_keep_on_shrink标志。查看此方法的内部逻辑以了解
 * 此标志在什么情况下起作用。
 */
template <typename... Ts>
void Resize(Ts... dim_source) {
    bool size_changed = SetDims(dim_source...);
    if (size_changed) {
        // 如果需要，我们将释放数据。下次mutable_data()调用将创建新存储
        bool reset_tensor = false;
        if (reserved_) {
            // 如果张量是预留的，除非容量小于新大小，否则不释放内存
            reset_tensor = storage_.capacity() < (storage_offset_ + numel_) * storage_.itemsize();
        } else {
            reset_tensor = storage_.capacity() <
                    (storage_offset_ + numel_) * storage_.itemsize() ||
                !FLAGS_caffe2_keep_on_shrink ||
                storage_.capacity() -
                        (storage_offset_ + numel_) * storage_.itemsize() >
                    static_cast<size_t>(FLAGS_caffe2_max_keep_on_shrink_memory);
        }

        if (reset_tensor && storage_initialized()) {
            FreeMemory();
        }
    }
}

/**
 * 调整张量形状而不触及底层存储。
 * 这要求张量的总大小保持不变。
 */
inline void Reshape(const std::vector<int64_t>& dims) {
    AT_ASSERTM(is_contiguous_,
        "目前Reshape仅支持连续存储的张量");
    int64_t new_size = 1;
    for (auto d : dims) {
        AT_ASSERT(d >= 0);
        new_size *= d;
    }
    AT_ASSERTM(new_size == numel_,
        "新旧大小不相等。不能使用Reshape而应使用Resize。"
        // TODO(jiayq): 待处理差异稳定后移除以下警告
        "旧版caffe2混合使用Reshape和Resize但此行为已更改。"
        "如果看到此错误，很可能需要将相应代码从Reshape改为Resize。");
    auto old_dim = sizes_.size();
    sizes_ = dims;
    update_to_contiguous_strides(old_dim);
}

/**
 * 释放张量持有的内存但保留尺寸和类型信息。
 * 后续调用mutable_data将触发新内存分配。
 */
inline void FreeMemory() {
    // 分离旧存储并创建新存储
    storage_ = at::Storage(storage_.device(), data_type_);
    storage_offset_ = 0;
}

/**
 * @brief 与另一个张量共享数据。
 *
 * 两个张量共享数据时，它们的尺寸必须已经相同。我们不隐式执行Resize使
 * 两个张量形状相同的原因是，我们希望允许不同形状但相同元素数量的张量
 * 仍然能够共享数据。这允许例如一个n维张量与其扁平化版本共享相同底层存储。
 *
 * 源张量应已分配数据。
 */
void ShareData(const TensorImpl& src) {
    // 目前我们假设device_type相同，因为在非模板代码中它们本质相同。
    // 应该添加断言，虽然可能轻微影响性能
    AT_ASSERTM(src.numel_ == numel_,
        "大小不匹配 - 共享数据前是否调用了reshape?");
    // 源张量可能尚未调用mutable_data()，此时ShareData()没有意义
    // TODO: 消除所有未初始化状态后添加断言
    if (!src.dtype_initialized()) {
        C10_LOG_EVERY_MS(WARNING, 1000) <<
            "源张量没有数据类型(是否在张量上调用了mutable_data<T>?)";
    }
    AT_ASSERTM(src.storage_initialized(),
        "源张量无内容但大小>0");
    // 执行共享
    /* 由于我们每当需要更改data_type/capacity时会创建新Storage，
     * 这仍保持原始语义
     */
    storage_ = src.storage();
    data_type_ = src.dtype();
    storage_offset_ = src.storage_offset();
}

/**
 * 与原始外部指针共享数据
 */
void ShareExternalPointer(
    at::DataPtr&& data_ptr,
    const caffe2::TypeMeta& data_type,
    size_t capacity) {
    AT_ASSERTM(data_type.id() != caffe2::TypeIdentifier::uninitialized(),
        "要与原始外部指针共享，需要传入已初始化的data_type(TypeMeta)");
    if (!capacity) {
        capacity = numel_ * data_type.itemsize();
    }
    if (storage_.unique()) {  // 存储唯一拥有
        storage_.UniqueStorageShareExternalPointer(
            std::move(data_ptr), data_type, capacity);
        data_type_ = data_type;
        storage_offset_ = 0;
    } else {  // 共享存储
        int64_t numel = capacity / data_type.itemsize();
        // 创建新Storage
        storage_ = at::Storage(data_type, numel, std::move(data_ptr), nullptr, true);
        data_type_ = data_type;
        storage_offset_ = 0;
    }
}

/**
 * 返回底层存储的可变原始指针。由于需要知道数据类型以进行分配，
 * 传入TypeMeta对象指定必要信息。这在概念上等同于调用mutable_data<T>()，
 * 其中TypeMeta参数meta从类型T派生。此函数与mutable_data<T>()的不同之处在于
 * 类型T可以通过TypeMeta对象在运行时指定。
 *
 * 如果现有数据不匹配所需类型，它将被删除并创建新存储。
 */
inline void* raw_mutable_data(const caffe2::TypeMeta& meta) {
    // 对于0大小张量，可以返回任何指针(包括nullptr)
    if (data_type_ == meta && storage_initialized()) {
        return static_cast<void*>(
            static_cast<char*>(storage_.data()) + 
            storage_offset_ * meta.itemsize());
    } else {
        bool had_special_dtor = data_type_.placementDelete() != nullptr;
        storage_offset_ = 0;
        if (storage_.unique()) {  // 存储唯一拥有
            storage_.set_dtype(meta);
        } else {  // 共享存储
            if (data_type_ != meta) {
                storage_ = at::Storage(storage_.device(), meta);
            }
        }
        data_type_ = meta;

        // 如果当前数据没有特殊析构器且新数据没有特殊构造函数，
        // 我们可以重用现有缓冲区
        if (numel_ == 0 ||
            (meta.placementNew() == nullptr && !had_special_dtor &&
             storage_.numel() >= numel_)) {
            AT_ASSERT(storage_offset_ == 0); // 因为我们刚刚重新分配
            return storage_.data();
        }
        const at::Allocator* allocator = storage_.allocator();
        // TODO: 摆脱StaticContext
        if (allocator == nullptr) {
            allocator = caffe2::GetAllocator(storage_.device_type());
        }
        if (meta.placementNew()) {  // 需要placement new的类型
            // 对于需要placement new的类型，我们将调用它，
            // 并确保在数据释放时调用正确的析构过程
            auto size = numel_;
            auto dtor = data_type_.placementDelete();
            auto data_ptr = allocator->allocate(numel_ * storage_.itemsize());
            storage_.set_data_ptr(PlacementDeleteContext::makeDataPtr(
                std::move(data_ptr), dtor, size, storage_.device()));
            data_type_.placementNew()(storage_.data(), numel_);
        } else {  // 基本类型
            // 对于基本类型，new和delete更简单
            storage_.set_data_ptr(
                allocator->allocate(numel_ * storage_.itemsize()));
        }
        storage_.set_numel(numel_);
        AT_ASSERT(storage_offset_ == 0); // 因为我们刚刚重新分配
        return storage_.data();
    }
}

/**
 * 返回底层存储的类型化指针。
 *
 * 对于基本类型，如果容量足够，我们重用可能的现有存储。
 */
template <typename T>
inline T* mutable_data() {
    if (storage_initialized() && storage_.IsType<T>()) {
        return static_cast<T*>(storage_.data()) + storage_offset_;
    }
    // 在此静态检查 - 否则TypeMeta会在尝试调用TypeMeta::ctor()时抛出运行时错误
    static_assert(std::is_default_constructible<T>::value,
        "张量不能持有非默认构造类型");
    return static_cast<T*>(raw_mutable_data(caffe2::TypeMeta::Make<T>()));
}

/**
 * 判断张量存储是否已初始化。调用Resize()或FreeMemory()后张量可能变为未初始化
 */
bool storage_initialized() const noexcept {
    return storage_.data() || numel_ == 0;
}

/**
 * 判断张量数据类型是否已初始化。使用Caffe2风格构造函数分配的张量在首次调用
 * mutable_data<T>()前数据类型未初始化。
 */
bool dtype_initialized() const noexcept {
    return data_type_ != caffe2::TypeMeta();
}

private:
// Caffe2的Resize()方法支持Resize({2,2})和可变参数Resize(2, 2)两种调用方式。
// 这些重载提供所有支持的调用配置，同时作为重载(而非模板)以便隐式转换仍然有效。
//
// ArrayRef的SetDims在内部实现为模板，因此我们可以处理不同类型的ArrayRef
// (Caffe2中有一些Resize用法传入int而非int64_t)

// 设置尺寸的内部模板实现
template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
bool SetDimsTemplate(at::ArrayRef<T> src) {
    auto old_numel = numel_;
    auto old_dim = sizes_.size();
    sizes_.resize(src.size());
    int64_t new_numel = 1;
    for (size_t i = 0; i < src.size(); ++i) {
        new_numel *= src[i];
        sizes_[i] = src[i];
    }
    update_to_contiguous_strides(old_dim);
    numel_ = new_numel;
    return numel_ != old_numel;
}

// 各种SetDims重载
bool SetDims(at::ArrayRef<int64_t> s) { return SetDimsTemplate(s); }
bool SetDims(at::ArrayRef<int> s) { return SetDimsTemplate(s); }
bool SetDims(at::ArrayRef<size_t> s) { return SetDimsTemplate(s); }
bool SetDims() { return SetDims(at::IntList{}); }
bool SetDims(const int64_t d0) { return SetDims(at::IntList{d0}); }
bool SetDims(const int64_t d0, const int64_t d1) { return SetDims(at::IntList{d0, d1}); }
bool SetDims(const int64_t d0, const int64_t d1, const int64_t d2) { 
    return SetDims(at::IntList{d0, d1, d2}); 
}
bool SetDims(const int64_t d0, const int64_t d1, const int64_t d2, const int64_t d3) {
    return SetDims(at::IntList{d0, d1, d2, d3});
}

// 更新为连续步长
inline void update_to_contiguous_strides(size_t old_dim) {
    strides_.resize(sizes_.size(), 0);
    if (dim() > 0) {
        int last_idx = dim() - 1;
        strides_[last_idx] = 1;
        for (auto i = last_idx - 1; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * std::max<int64_t>(sizes_[i + 1], 1);
        }
    }
    is_contiguous_ = true;
}

// 计算基于张量尺寸的元素数量
int64_t compute_numel() const {
    int64_t n = 1;
    for (auto s : sizes()) {
        n *= s;
    }
    return n;
}

// 计算张量是否连续
bool compute_contiguous() const;

protected:
/**
 * 重新计算张量的缓存元素数量。如果修改了尺寸，调用此方法。
 */
void refresh_numel() {
    AT_ASSERT(!is_variable());
    numel_ = compute_numel();
}

/**
 * 重新计算张量的缓存连续性。如果修改了尺寸或步长，调用此方法。
 */
void refresh_contiguous() {
    AT_ASSERT(!is_variable());
    is_contiguous_ = compute_contiguous();
}

public:
at::Storage storage_; // TODO: 修复我的可见性

protected:
// 我们可以通过组合SmallVector结构节省一两个字，
// 因为它们的大小是冗余的，如果需要溢出缓冲区空间，
// 我们可以将两个指针保持在一起。然而，这将需要从头实现另一个结构，
// 所以只有在确实需要时才这样做。
at::SmallVector<int64_t,5> sizes_;
at::SmallVector<int64_t,5> strides_;

int64_t storage_offset_ = 0;
// 如果sizes和strides为空，numel为1！！然而，大多数情况下，
// 我们会立即将sizes设置为{0}并将numel重置为0。
// (不能在默认初始化器中这样做，因为无法拼写"分配一个元素数组"给strides_)
int64_t numel_ = 1;

// 不变式：当storage非空时，此type meta必须与storage中的type meta一致
caffe2::TypeMeta data_type_;

// 在这里可以有八个字节大小的字段，之后应该将其打包到位字段中
TensorTypeId type_id_;
bool is_contiguous_ = true;
bool is_variable_ = false;
bool is_wrapped_number_ = false;
// 我们决定保留reserved_，在拆分后它将存在于Tensor中
// 逻辑是如果曾经调用过Extend()或ReserveSpace()，
// 则后续Resize()不会释放存储
bool reserved_ = false;
};

// 注释[TensorImpl大小约束]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 更改了TensorImpl的大小？如果大小减小了，很好！
// 调整下面的文档和预期大小。变大了？继续阅读...
//
// 结构体大小很重要。在Facebook的一些生产系统中，训练运行期间
// 有4亿个活跃张量。算一下：你添加到Tensor的每个64位字
// 在RAM中都是额外的3.2GB。
//
// 如果你是Facebook员工，可以使用以下命令检查相关运行是否超出限制：
// https://fburl.com/q5enpv98
//
// 作为参考，我们在每个TensorImpl 160字节(20字)时OOM。
// 这不包括步长非内联分配和StorageImpl空间的开销，
// 这是在我们将sizes和strides直接内联到TensorImpl作为SmallVectors之前。
//
// 我们在32位系统上的内存使用不理想，但目前没有检查
// (以避免当32位数字错误时引发愤怒循环)
//
// 当前分解：
//
//    vtable指针
//    强引用计数           TODO: 将这些打包到一个字中
//    弱引用计数
//    存储指针
//    sizes SmallVector (开始)
//    sizes SmallVector (结束)
//    sizes SmallVector (容量)
//    sizes SmallVector (预分配0)
//    sizes SmallVector (预分配1)
//    sizes SmallVector (预分配2)
//    sizes SmallVector (预分配3)
//    sizes SmallVector (预分配4)
//    strides SmallVector (开始)
//    strides SmallVector (结束)
//    strides SmallVector (容量)
//    strides SmallVector (预分配0)
//    strides SmallVector (预分配1)
//    strides SmallVector (预分配2)
//    strides SmallVector (预分配3)
//    strides SmallVector (预分配4)
//    存储偏移量
//    元素数量
//    数据类型指针
//    杂项位字段
//
static_assert(sizeof(void*) != sizeof(int64_t) || // 如果是64位...
              sizeof(TensorImpl) == sizeof(int64_t) * 24,
              "你在64位架构上更改了TensorImpl的大小。"
              "参见注释[TensorImpl大小约束]了解如何继续。");

} // namespace at
