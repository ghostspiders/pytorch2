#pragma once

#include <ATen/core/DefaultDtype.h>
#include <c10/core/Backend.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeUtils.h>
#include <c10/Device.h>

#include <c10/util/Optional.h>
#include <c10/util/C++17.h>
#include <c10/macros/Macros.h>

#include <cstddef>
#include <iosfwd>
#include <utility>

namespace at {
/// 一个封装Tensor构造选项的类。TensorOptions的设计初衷是为了支持Python风格的API，
/// 用于在工厂函数中指定构造选项，例如：
///
///     torch.zeros(2, 3, dtype=torch.int32)
///
/// 由于C++原生不支持关键字参数，因此需要另一种方式来指定类似关键字的参数。TensorOptions是一个
/// 构建器类，可用于构造这种关键字参数的"字典"：支持TensorOptions的函数通常将其作为可选的最后一个参数。
///
/// 警告：在PyTorch中，工厂函数有`torch::`变体，例如torch::zeros对应at::zeros。
/// 这些变体返回Variables（而标准ATen函数返回普通Tensors）。如果混淆这些函数，会导致问题。
///
/// 相比于直接使用该类的构造函数，更推荐使用构造函数函数，然后链式调用setter方法：
///
///     at::device(at::kCUDA).dtype(kInt)
///     at::dtype(at::kInt)
///
/// 此外，在任何需要TensorOptions的地方，可以直接传递at::kCUDA/at::kInt，
/// 它们会隐式转换为TensorOptions。
///
/// 以下是创建具有特定属性的2x2零张量的推荐方式（都隐式使用了TensorOptions）：
///
///     at::zeros({2,2}, at::kCUDA);  // 在CUDA设备上
///     at::zeros({2,2}, at::kLong);  // 指定long类型
///     at::zeros({2,2}, at::device(at::kCUDA).dtype(at::kLong()));  // 链式调用
///     at::zeros({2,2}, at::device({at::kCUDA, 1}));  // 指定1号设备
///     at::zeros({2,2}, at::requires_grad());  // 需要梯度计算
///

/// 说明 [ TensorOptions构造函数 ]
///
/// TensorOptions类似于包含以下可选字段的字典：
/// {requires_grad, is_variable, device, dtype, layout}。它被广泛用于C++内部和API中
/// 指定张量属性，例如工厂方法`at::empty({10}, options)`、张量转换`tensor.to(...)`等。
///
/// 为了提供与Python一致的简洁API（在Python中可以直接用`torch.empty(sizes, X)`，
/// X可以是`torch.device`、`torch.dtype`或`torch.layout`），我们需要让TensorOptions能
/// 隐式转换自`ScalarType dtype`、`Layout layout`和`Device device`。因此我们为这三类
/// 类型分别提供了隐式构造函数。
///
/// 对于简单的枚举类`ScalarType`和`Layout`这已经足够。但`Device`是一个具有隐式构造函数
/// `Device(DeviceType, DeviceIndex = -1)`和`Device(std::string)`的普通类（为了与Python API
/// 保持一致，例如"cuda:1"可以传递给任何接受`torch.device("cuda:1")`的地方）。为了支持
/// `at::empty({10}, {kCUDA, 1})`和`tensor.to(kCUDA)`等语法，我们需要确保TensorOptions
/// 能用任何可构造Device的参数隐式构造。因此我们实现了：
///
///    /* 隐式 */ TensorOptions(T&& device) : TensorOptions() {
///      this->set_device(device);
///    }
///
///    template <typename... Args,
///             typename = std::enable_if_t<std::is_constructible<Device, Args&&...>::value>>
///    /* 隐式 */ TensorOptions(Args&&... args)
///     : TensorOptions(Device(std::forward<Args>(args)...)) {}
///
/// 但这会带来问题。考虑`TensorOptions({kCUDA, 1})`的情况：编译器会报错，因为
/// `{kCUDA, 1}`可以同时转换为`TensorOption`和`Device`，导致拷贝构造函数和
/// `Device`构造函数之间的歧义。
///
/// 解决方法是将`Device`构造函数模板化。由于重载决议在模板决议之前进行，
/// 这个问题就得到了解决。

/// 封装Tensor构造选项的类，用于指定张量的设备/数据类型/布局等属性
struct CAFFE2_API TensorOptions {
  // 默认构造函数，初始化所有选项为未设置状态
  TensorOptions()
    : requires_grad_(false)    // 默认不需要梯度计算
    , is_variable_(false)     // 默认不是Variable类型
    , has_device_(false)      // 设备未设置标志
    , has_dtype_(false)       // 数据类型未设置标志  
    , has_layout_(false)      // 布局未设置标志
    , has_requires_grad_(false) // requires_grad未设置标志
    , has_is_variable_(false)   // is_variable未设置标志
    {}

  /// 通过Layout构造TensorOptions的隐式构造函数
  /* implicit */ TensorOptions(Layout layout) : TensorOptions() {
    this->set_layout(layout);  // 调用私有方法设置布局
  }

  /// 通过Device构造TensorOptions的隐式构造函数（模板化以避免构造歧义）
  /// 使用SFINAE确保T可以转换为Device类型
  template<typename T,
           typename = c10::guts::enable_if_t<std::is_same<c10::guts::decay_t<T>, Device>::value>>
  /* implicit */ TensorOptions(T&& device) : TensorOptions() {
    this->set_device(std::forward<T>(device));  // 完美转发设备参数
  }

  /// 通过Device构造参数包转发构造TensorOptions
  /// 支持任何可以构造Device的参数组合
  template <typename... Args,
            typename = c10::guts::enable_if_t<std::is_constructible<Device, Args&&...>::value>>
   /* implicit */ TensorOptions(Args&&... args)
    : TensorOptions(Device(std::forward<Args>(args)...)) {}  // 转发参数构造Device

  /// 通过Backend类型构造TensorOptions的隐式构造函数
  /* implicit */ TensorOptions(Backend backend)
      : TensorOptions(Device(backendToDeviceType(backend))) {}  // 转换Backend为Device

  /// 通过TypeMeta数据类型构造TensorOptions的隐式构造函数  
  /* implicit */ TensorOptions(caffe2::TypeMeta dtype) : TensorOptions() {
    this->set_dtype(dtype);  // 设置数据类型
  }

  /// 通过ScalarType数据类型构造TensorOptions的隐式构造函数（传统接口）
  /* implicit */ TensorOptions(ScalarType dtype) : TensorOptions() {
    this->set_dtype(dtype);  // 设置标量类型
  }

  /// 相等比较运算符，比较两个TensorOptions的所有属性
  bool operator==(const TensorOptions& other) const noexcept {
    return
        has_dtype_ == other.has_dtype_ &&  // 比较数据类型标志
        has_layout_ == other.has_layout_ &&  // 比较布局标志
        has_device_ == other.has_device_ &&  // 比较设备标志
        has_requires_grad_ == other.has_requires_grad_ &&  // 比较requires_grad标志
        has_is_variable_ == other.has_is_variable_ &&  // 比较is_variable标志
        (!has_dtype_ || dtype_ == other.dtype_) &&  // 比较实际数据类型
        (!has_layout_ || layout_ == other.layout_) &&  // 比较实际布局
        (!has_device_ || device_ == other.device_) &&  // 比较实际设备
        (!requires_grad_ || requires_grad_ == other.requires_grad_) &&  // 比较requires_grad
        (!is_variable_ || is_variable_ == other.is_variable_);  // 比较is_variable
  }

  /// 不等比较运算符，调用相等运算符取反
  bool operator!=(const TensorOptions& other) const noexcept {
    return !(*this == other);
  }

  /// 设置设备选项并返回新副本
  /// @param device 要设置的设备，传入nullopt表示清除设备设置
  /// @return 包含新设备设置的新TensorOptions对象
  C10_NODISCARD TensorOptions device(c10::optional<Device> device) const noexcept {
    TensorOptions r = *this;  // 创建副本
    r.set_device(device);     // 调用私有方法设置设备
    return r;                 // 返回新对象
  }

  /// 设置设备选项的可变参数模板版本
  /// 支持直接传入设备构造参数而不需要显式创建Device对象
  template<typename ... Args>
  C10_NODISCARD TensorOptions device(Args&&... args) const noexcept {
    // 使用in_place构造optional<Device>避免临时对象
    return device(c10::optional<Device>(c10::in_place, std::forward<Args>(args)...));
  }

  /// 设置CUDA设备索引的传统接口（不推荐使用）
  /// @param device_index CUDA设备索引
  /// @return 包含新设备设置的新TensorOptions对象
  C10_NODISCARD TensorOptions device_index(int16_t device_index) const noexcept {
    return device(Device::Type::CUDA, device_index);  // 固定设备类型为CUDA
  }

  /// 设置数据类型(TypeMeta)并返回新副本
  /// @param dtype 要设置的数据类型，传入nullopt表示清除数据类型设置
  /// @return 包含新数据类型设置的新TensorOptions对象
  C10_NODISCARD TensorOptions dtype(c10::optional<caffe2::TypeMeta> dtype) const noexcept {
    TensorOptions r = *this;  // 创建副本
    r.set_dtype(dtype);       // 调用私有方法设置数据类型
    return r;                 // 返回新对象
  }

  /// 设置数据类型(ScalarType)的传统接口
  /// @param dtype 要设置的标量类型，传入nullopt表示清除数据类型设置
  /// @return 包含新数据类型设置的新TensorOptions对象
  C10_NODISCARD TensorOptions dtype(c10::optional<ScalarType> dtype) const noexcept {
    TensorOptions r = *this;  // 创建副本
    r.set_dtype(dtype);       // 调用私有方法设置标量类型
    return r;                 // 返回新对象
  }

  /// 模板方法设置具体类型的数据类型
  /// 示例：options.dtype<float>()
  /// @return 返回当前对象的引用以支持链式调用
  template <typename T>
  TensorOptions& dtype() {
    dtype_ = caffe2::TypeMeta::Make<T>();  // 通过类型创建TypeMeta
    has_dtype_ = true;                     // 设置数据类型标志
    return *this;                          // 返回当前对象引用
  }

  /// 设置布局并返回新副本
  /// @param layout 要设置的布局，传入nullopt表示清除布局设置
  /// @return 包含新布局设置的新TensorOptions对象
  C10_NODISCARD TensorOptions layout(c10::optional<Layout> layout) const noexcept {
    TensorOptions r = *this;  // 创建副本
    r.set_layout(layout);     // 调用私有方法设置布局
    return r;                 // 返回新对象
  }

  /// 设置requires_grad属性并返回新副本
  /// @param requires_grad 是否需要梯度，传入nullopt表示清除设置
  /// @return 包含新requires_grad设置的新TensorOptions对象
  C10_NODISCARD TensorOptions requires_grad(c10::optional<bool> requires_grad) const noexcept {
    TensorOptions r = *this;      // 创建副本
    r.set_requires_grad(requires_grad);  // 调用私有方法设置requires_grad
    return r;                     // 返回新对象
  }

  /// 设置is_variable属性并返回新副本
  /// @param is_variable 是否是Variable，传入nullopt表示清除设置
  /// @return 包含新is_variable设置的新TensorOptions对象
  C10_NODISCARD TensorOptions is_variable(c10::optional<bool> is_variable) const noexcept {
    TensorOptions r = *this;    // 创建副本
    r.set_is_variable(is_variable);  // 调用私有方法设置is_variable
    return r;                   // 返回新对象
  }

  /// 获取设备信息
  /// @return 如果设置了设备则返回该设备，否则返回默认CPU设备
  Device device() const noexcept {
    return has_device_ ? device_ : Device(kCPU);
  }

  /// 检查是否设置了设备
  /// @return 是否设置了设备标志
  bool has_device() const noexcept {
    return has_device_;
  }

  /// 获取设备信息的optional版本
  /// @return 如果设置了设备则返回包含设备的optional，否则返回nullopt
  c10::optional<Device> device_opt() const noexcept {
    return has_device_ ? c10::make_optional(device_) : c10::nullopt;
  }

  /// 获取设备索引
  /// @return 当前设备的索引号
  int32_t device_index() const noexcept {
    return device().index();
  }

  /// 获取数据类型信息
  /// @return 如果设置了数据类型则返回该类型，否则返回默认float类型
  caffe2::TypeMeta dtype() const noexcept {
    return has_dtype_ ? dtype_ : get_default_dtype();
  }

  /// 检查是否设置了数据类型
  /// @return 是否设置了数据类型标志
  bool has_dtype() const noexcept {
    return has_dtype_;
  }

  /// 获取数据类型信息的optional版本
  /// @return 如果设置了数据类型则返回包含类型的optional，否则返回nullopt
  c10::optional<caffe2::TypeMeta> dtype_opt() const noexcept {
    return has_dtype_ ? c10::make_optional(dtype_) : c10::nullopt;
  }

  /// 获取布局信息
  /// @return 如果设置了布局则返回该布局，否则返回默认strided布局
  Layout layout() const noexcept {
    return has_layout_ ? layout_ : kStrided;
  }

  /// 检查是否设置了布局
  /// @return 是否设置了布局标志
  bool has_layout() const noexcept {
    return has_layout_;
  }

  /// 获取布局信息的optional版本
  /// @return 如果设置了布局则返回包含布局的optional，否则返回nullopt
  c10::optional<Layout> layout_opt() const noexcept {
    return has_layout_ ? c10::make_optional(layout_) : c10::nullopt;
  }

  /// 获取requires_grad属性
  /// @return 如果设置了requires_grad则返回该值，否则返回false
  bool requires_grad() const noexcept {
    return has_requires_grad_ ? requires_grad_ : false;
  }

  /// 检查是否设置了requires_grad
  /// @return 是否设置了requires_grad标志
  bool has_requires_grad() const noexcept {
    return has_requires_grad_;
  }

  /// 获取requires_grad属性的optional版本
  /// @return 如果设置了requires_grad则返回包含值的optional，否则返回nullopt
  c10::optional<bool> requires_grad_opt() const noexcept {
    return has_requires_grad_ ? c10::make_optional(requires_grad_)
                              : c10::nullopt;
  }

  /// 获取is_variable属性
  /// @return 如果设置了is_variable则返回该值，否则返回false
  bool is_variable() const noexcept {
    return has_is_variable_ ? is_variable_ : false;
  }

  /// 检查是否设置了is_variable
  /// @return 是否设置了is_variable标志
  bool has_is_variable() const noexcept {
    return has_is_variable_;
  }

  /// 获取is_variable属性的optional版本
  /// @return 如果设置了is_variable则返回包含值的optional，否则返回nullopt
  c10::optional<bool> is_variable_opt() const noexcept {
    return has_is_variable_ ? c10::make_optional(is_variable_) : c10::nullopt;
  }

  /// 计算并返回实际的Backend类型
  /// 根据设备和布局确定具体是哪种Backend
  /// @return 计算得到的Backend枚举值
  Backend backend() const noexcept {
    Backend backend;
    if (device().type() == Device::Type::CPU) {
      // CPU设备：根据布局选择密集或稀疏CPU后端
      backend = (layout() == kStrided) ? Backend::CPU : Backend::SparseCPU;
    } else {
      // CUDA设备：根据布局选择密集或稀疏CUDA后端
      backend = (layout() == kStrided) ? Backend::CUDA : Backend::SparseCUDA;
    }
    return backend;
  }

 private:
  // 注意：如果添加新选项，必须相应调整Tensor::options的实现

  // 使用独立bool标志而非optional以节省空间
  caffe2::TypeMeta dtype_ = caffe2::TypeMeta::Make<float>(); // 数据类型(64位)
  Device device_ = at::kCPU;       // 设备信息(32位)
  Layout layout_ = at::kStrided;   // 内存布局(8位)

  // 使用位域压缩存储布尔标志(共7位)
  bool requires_grad_     : 1;  // 是否需要梯度
  bool is_variable_       : 1;  // 是否为Variable
  bool has_device_        : 1;  // 是否设置了设备
  bool has_dtype_         : 1;  // 是否设置了数据类型
  bool has_layout_        : 1;  // 是否设置了布局
  bool has_requires_grad_ : 1;  // 是否设置了requires_grad
  bool has_is_variable_   : 1;  // 是否设置了is_variable

  // 私有设置方法（禁止在临时对象上调用，通过引用限定符&实现）

  /// 设置设备（私有方法）
  /// @param device 要设置的设备optional
  void set_device(c10::optional<Device> device) & noexcept {
    if (device) {
      device_ = *device;      // 解引用optional设置设备
      has_device_ = true;     // 设置标志位
    } else {
      has_device_ = false;    // 清除标志位
    }
  }

  /// 设置数据类型TypeMeta（私有方法）
  /// @param dtype 要设置的数据类型optional
  void set_dtype(c10::optional<caffe2::TypeMeta> dtype) & noexcept {
    if (dtype) {
      dtype_ = *dtype;       // 解引用optional设置数据类型
      has_dtype_ = true;      // 设置标志位
    } else {
      has_dtype_ = false;     // 清除标志位
    }
  }

  /// 设置数据类型ScalarType（私有方法，传统接口）
  /// @param dtype 要设置的标量类型optional
  void set_dtype(c10::optional<ScalarType> dtype) & noexcept {
    if (dtype) {
      dtype_ = scalarTypeToTypeMeta(*dtype);  // 转换标量类型为TypeMeta
      has_dtype_ = true;           // 设置标志位
    } else {
      has_dtype_ = false;          // 清除标志位
    }
  }

  /// 设置布局（私有方法）
  /// @param layout 要设置的布局optional
  void set_layout(c10::optional<Layout> layout) & noexcept {
    if (layout) {
      layout_ = *layout;      // 解引用optional设置布局
      has_layout_ = true;     // 设置标志位
    } else {
      has_layout_ = false;    // 清除标志位
    }
  }

  /// 设置requires_grad（私有方法）
  /// @param requires_grad 要设置的requires_grad optional
  void set_requires_grad(c10::optional<bool> requires_grad) & noexcept {
    if (requires_grad) {
      requires_grad_ = *requires_grad;  // 解引用optional设置值
      has_requires_grad_ = true;       // 设置标志位
    } else {
      has_requires_grad_ = false;      // 清除标志位
    }
  }

  /// 设置is_variable（私有方法）
  /// @param is_variable 要设置的is_variable optional
  void set_is_variable(c10::optional<bool> is_variable) & noexcept {
    if (is_variable) {
      is_variable_ = *is_variable;    // 解引用optional设置值
      has_is_variable_ = true;        // 设置标志位
    } else {
      has_is_variable_ = false;       // 清除标志位
    }
  }
};

// 静态断言确保内存占用不超过128位（64位系统为16字节）
static_assert(sizeof(TensorOptions) <= sizeof(int64_t) * 2,
               "TensorOptions必须控制在128位以内");

// 工厂函数定义区

/// 创建指定数据类型的TensorOptions便捷函数
/// @param dtype 数据类型TypeMeta
/// @return 包含该数据类型的TensorOptions
inline TensorOptions dtype(caffe2::TypeMeta dtype) {
  return TensorOptions().dtype(dtype);  // 创建默认选项后设置数据类型
}

/// 创建指定数据类型的TensorOptions便捷函数（传统ScalarType接口）
/// @param dtype 标量类型
/// @return 包含该数据类型的TensorOptions
inline TensorOptions dtype(ScalarType dtype) {
  return TensorOptions().dtype(scalarTypeToTypeMeta(dtype));  // 转换后设置
}

/// 创建指定布局的TensorOptions便捷函数
/// @param layout 布局类型
/// @return 包含该布局的TensorOptions
inline TensorOptions layout(Layout layout) {
  return TensorOptions().layout(layout);  // 创建默认选项后设置布局
}

/// 创建指定设备的TensorOptions便捷函数
/// @param device 设备对象
/// @return 包含该设备的TensorOptions
inline TensorOptions device(Device device) {
  return TensorOptions().device(std::move(device));  // 移动语义传递设备
}

/// 创建指定CUDA设备索引的TensorOptions便捷函数
/// @param device_index CUDA设备索引
/// @return 包含该设备的TensorOptions
inline TensorOptions device_index(int16_t device_index) {
  return TensorOptions().device_index(device_index);  // 设置CUDA设备索引
}

/// 创建指定requires_grad的TensorOptions便捷函数
/// @param requires_grad 是否需要梯度计算
/// @return 包含该设置的TensorOptions
inline TensorOptions requires_grad(bool requires_grad = true) {
  return TensorOptions().requires_grad(requires_grad);  // 设置requires_grad
}

/// TensorOptions的流输出运算符声明
CAFFE2_API std::ostream& operator<<(
    std::ostream& stream,
    const TensorOptions& options);

/// 模板化的数据类型工厂函数
/// 示例：auto opts = dtype<float>();
/// @return 包含指定类型的TensorOptions
template <typename T>
inline TensorOptions dtype() {
  return dtype(caffe2::TypeMeta::Make<T>());  // 通过类型创建TypeMeta
}

} // namespace at