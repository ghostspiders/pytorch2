#pragma once  // 防止头文件重复包含

// 包含必要的头文件
#include <ATen/core/Scalar.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorImpl.h>
#include <ATen/core/UndefinedTensorImpl.h>
#include <ATen/core/blob.h>
#include <c10/util/intrusive_ptr.h>  // 包含intrusive_ptr智能指针实现
#include <ATen/core/thread_pool.h>
#include <type_traits>

namespace c10 {
struct IValue;

namespace ivalue {

// 定义Shared为intrusive_ptr的别名模板
template <typename T>
using Shared = c10::intrusive_ptr<T>;

// 常量字符串结构体，继承自intrusive_ptr_target以支持引用计数
struct CAFFE2_API ConstantString final : c10::intrusive_ptr_target {
 private:
  const std::string str_;  // 内部存储的字符串
  
 public:
  // 构造函数，接收字符串并移动存储
  ConstantString(std::string str) : str_(std::move(str)) {}
  
  // 静态工厂方法，创建ConstantString的intrusive_ptr
  static c10::intrusive_ptr<ConstantString> create(std::string str_);
  
  // 获取存储的字符串常量引用
  const std::string & string() const {
    return str_;
  }
  
  // 类型转换运算符，可直接转换为string引用
  operator const std::string & () const {
    return string();
  }
  
  // 友元函数，重载输出运算符
  CAFFE2_API friend std::ostream& operator<<(
      std::ostream& out,
      const ConstantString& v);
};

// 泛型列表结构体，同样支持引用计数
template <typename Elem>
struct C10_EXPORT List : c10::intrusive_ptr_target {
 private:
  std::vector<Elem> elements_;  // 内部使用vector存储元素

 public:
  typedef Elem ElemType;  // 元素类型别名

  // 构造函数，接收元素vector并移动存储
  List(std::vector<Elem> elements_) : elements_(std::move(elements_)) {}
  
  // 静态工厂方法，创建List的intrusive_ptr
  static c10::intrusive_ptr<List<Elem>> create(std::vector<Elem> elements_) {
    return c10::make_intrusive<List<Elem>>(std::move(elements_));
  }
  
  // 获取元素vector的常量引用
  const std::vector<Elem>& elements() const {
    return elements_;
  }
  
  // 类型转换运算符，可直接转换为vector常量引用
  operator const std::vector<Elem>&() const {
    return elements();
  }

  // 获取元素vector的非常量引用
  std::vector<Elem>& elements() {
    return elements_;
  }
  
  // 类型转换运算符，可直接转换为vector非常量引用
  operator std::vector<Elem>&() {
    return elements();
  }
};

// 前向声明Future结构体
struct Future;

// Tuple元组结构体，继承自List<IValue>
// C10_EXPORT宏确保在不同平台(DLL/SO)上能正确导出符号
struct C10_EXPORT Tuple : public List<IValue> {
  // 继承基类的构造函数
  using List<IValue>::List;
  
  // 静态工厂方法，创建Tuple的intrusive_ptr
  static c10::intrusive_ptr<Tuple> create(std::vector<IValue> elements_) {
    return c10::make_intrusive<Tuple>(std::move(elements_));
  }
};

// 定义各种常用列表类型的别名
using IntList = List<int64_t>;        // 64位整数列表
using TensorList = List<at::Tensor>;  // 张量列表
using DoubleList = List<double>;      // 双精度浮点数列表
using BoolList = List<bool>;          // 布尔值列表
using GenericList = List<IValue>;     // 通用IValue列表

} // namespace ivalue结束

// IValue是解释器使用的通用标记联合(tagged union)类型
// 用于保存所有值类型。它是一个16字节的对象：
// - 8字节有效载荷(payload)
// - 8字节标记(tag)，其中：
//   * 4字节用于确定类型
//   * 1字节标记该类型是否是c10::intrusive_ptr_target的子类
//     从而决定是否需要retain/release调用

// 定义所有支持的标签类型的宏
// 每个标签对应IValue可以存储的一种类型
#define TORCH_FORALL_TAGS(_) \
  _(None)       /* 空值 */ \
  _(Tensor)     /* 张量 */ \
  _(Double)     /* 双精度浮点数 */ \
  _(Int)        /* 整数 */ \
  _(Bool)       /* 布尔值 */ \
  _(Tuple)      /* 元组 */ \
  _(IntList)    /* 整数列表 */ \
  _(DoubleList) /* 双精度浮点数列表 */ \
  _(BoolList)   /* 布尔值列表 */ \
  _(String)     /* 字符串 */ \
  _(TensorList) /* 张量列表 */ \
  _(Blob)       /* 二进制大对象 */ \
  _(GenericList) /* 通用列表 */ \
  _(Future)     /* 未来值(用于异步计算) */ \
  _(Device)     /* 设备类型 */

// IValue 是一个通用的值类型，可以存储多种不同类型的数据
struct CAFFE2_API IValue final {
  // 默认构造函数，初始化为None类型
  IValue()
  : payload{0}
  , tag(Tag::None)
  , is_intrusive_ptr(false) {}
  
  // 拷贝构造函数
  IValue(const IValue& rhs)
      : payload(rhs.payload),
        tag(rhs.tag),
        is_intrusive_ptr(rhs.is_intrusive_ptr) {
    // 如果是侵入式指针，增加引用计数
    if (is_intrusive_ptr) {
      c10::raw::intrusive_ptr::incref(payload.as_intrusive_ptr);
    }
  }
  
  // 移动构造函数
  IValue(IValue&& rhs) noexcept : IValue() {
    swap(rhs);
  }
  
  // 析构函数
  ~IValue() {
    // 如果是侵入式指针，减少引用计数
    if (is_intrusive_ptr) {
      c10::raw::intrusive_ptr::decref(payload.as_intrusive_ptr);
    }
  }
  
  // 移动赋值运算符
  IValue & operator=(IValue && rhs) & noexcept {
    IValue(std::move(rhs)).swap(*this); // 交换内容，同时将rhs置为None
    return *this;
  }
  
  // 拷贝赋值运算符
  IValue & operator=(IValue const & rhs) & {
    IValue(rhs).swap(*this);
    return *this;
  }

  // 检查当前IValue是否是另一个IValue的别名
  bool isAliasOf(const IValue& rhs) const {
    if (this->tag != rhs.tag) {
      // 类型不同肯定不是别名
      return false;
    }

    if (!this->is_intrusive_ptr) {
      // 基本类型没有别名概念
      return false;
    }

    AT_ASSERT(rhs.is_intrusive_ptr);

    // 对于Tensor类型，比较内部存储
    if (this->isTensor()) {
      const auto thisTensor = this->toTensor();
      const auto rhsTensor = rhs.toTensor();
      return thisTensor.is_alias_of(rhsTensor);
    }

    // 其他类型直接比较指针值
    return this->payload.as_intrusive_ptr == rhs.payload.as_intrusive_ptr;
  }
  
  // 交换两个IValue的内容
  void swap(IValue & rhs) noexcept {
    std::swap(payload, rhs.payload);
    std::swap(is_intrusive_ptr, rhs.is_intrusive_ptr);
    std::swap(tag, rhs.tag);
  }

  // 以下是各种子类型的访问器

  // Tensor类型的构造函数
  IValue(at::Tensor t)
  : tag(Tag::Tensor), is_intrusive_ptr(t.defined())  {
    // 注意：未定义的Tensor不进行引用计数，所以虽然它被标记为Tensor，
    // 但is_intrusive_ptr设置为false。
    // 这不是可选的优化：我们的incref调用在未定义的Tensor上不会正常工作。
    payload.as_intrusive_ptr = t.unsafeReleaseTensorImpl();
  }
  
  // 检查是否是Tensor类型
  bool isTensor() const { return Tag::Tensor == tag; }
  
  // 右值转换为Tensor
  at::Tensor toTensor() && {
    AT_ASSERT(isTensor());
    return at::Tensor(moveToIntrusivePtr<at::TensorImpl, at::UndefinedTensorImpl>());
  }
  
  // 左值转换为Tensor
  at::Tensor toTensor() const & {
    AT_ASSERT(isTensor());
    return at::Tensor(toIntrusivePtr<at::TensorImpl, at::UndefinedTensorImpl>());
  }

  // 转换为IValue引用（用于泛型编程）
  const IValue& toIValue() const {
    return *this;
  }
  IValue& toIValue() {
    return *this;
  }

  // Blob类型的构造函数
  IValue(caffe2::Blob blob) : tag(Tag::Blob), is_intrusive_ptr(true) {
    // TODO (Tensor合并后) 如果传入的Blob包含Tensor，应该提取并存储为Tensor
    payload.as_intrusive_ptr =
        c10::make_intrusive<caffe2::Blob>(std::move(blob)).release();
  }
  
  // 检查是否是Blob类型
  bool isBlob() const {
    return Tag::Blob == tag;
  }
  
  // 转换为Blob引用（左值）
  caffe2::Blob& toBlob() & {
    AT_ASSERT(isBlob());
    return *static_cast<caffe2::Blob*>(payload.as_intrusive_ptr);
  }
  
  // 转换为Blob引用（常量左值）
  const caffe2::Blob& toBlob() const& {
    AT_ASSERT(isBlob());
    return *static_cast<const caffe2::Blob*>(payload.as_intrusive_ptr);
  }
    return *static_cast<caffe2::Blob*>(payload.as_intrusive_ptr);
  }

   // Tuple（元组）类型相关方法
  IValue(c10::intrusive_ptr<ivalue::Tuple> v);  // 元组构造函数
  bool isTuple() const { return Tag::Tuple == tag; }  // 检查是否为元组类型
  c10::intrusive_ptr<ivalue::Tuple> toTuple() && {    // 右值转换为元组
    AT_ASSERT(isTuple());
    return moveToIntrusivePtr<ivalue::Tuple>();
  }
  c10::intrusive_ptr<ivalue::Tuple> toTuple() const & {  // 左值转换为元组
    AT_ASSERT(isTuple());
    return toIntrusivePtr<ivalue::Tuple>();
  }

  // Double（双精度浮点数）类型相关方法
  IValue(double d)       // 双精度浮点数构造函数
  : tag(Tag::Double), is_intrusive_ptr(false) {
    payload.as_double = d;
  }
  bool isDouble() const { return Tag::Double == tag; }  // 检查是否为双精度浮点数
  double toDouble() const {    // 转换为双精度浮点数值
    AT_ASSERT(isDouble());
    return payload.as_double;
  }

  // Future（未来值）类型相关方法
  IValue(c10::intrusive_ptr<ivalue::Future> v);  // Future构造函数
  bool isFuture() const { return Tag::Future == tag; }  // 检查是否为Future
  c10::intrusive_ptr<ivalue::Future> toFuture() && {   // 右值转换为Future
    AT_ASSERT(isFuture());
    return moveToIntrusivePtr<ivalue::Future>();
  }
  c10::intrusive_ptr<ivalue::Future> toFuture() const & {  // 左值转换为Future
    AT_ASSERT(isFuture());
    return toIntrusivePtr<ivalue::Future>();
  }

  // Int（整数）类型相关方法
  IValue(int64_t i)     // 64位整数构造函数
  : tag(Tag::Int), is_intrusive_ptr(false) {
    payload.as_int = i;
  }

  // 允许传入字面量(如3,4)而不会产生歧义
  IValue(int32_t i)     // 32位整数构造函数
  : IValue(static_cast<int64_t>(i)) {}

  bool isInt() const { return Tag::Int == tag; }  // 检查是否为整数

  int64_t toInt() const {    // 转换为整数值
    AT_ASSERT(isInt());
    return payload.as_int;
  }

  // Bool（布尔值）类型相关方法
  IValue(bool b)        // 布尔值构造函数
  : tag(Tag::Bool), is_intrusive_ptr(false) {
    payload.as_bool = b;
  }
  bool isBool() const { return Tag::Bool == tag; }  // 检查是否为布尔值
  bool toBool() const {     // 转换为布尔值
    AT_ASSERT(isBool());
    return payload.as_bool;
  }

  // IntList（整数列表）类型相关方法
  IValue(c10::intrusive_ptr<ivalue::IntList> v);  // IntList构造函数(侵入式指针)
  IValue(std::vector<int64_t> v);                 // IntList构造函数(向量)
  IValue(at::ArrayRef<int64_t> v)                 // IntList构造函数(数组引用)
  : IValue(v.vec()) {}
  bool isIntList() const { return Tag::IntList == tag; }  // 检查是否为整数列表
  c10::intrusive_ptr<ivalue::IntList> toIntList() && {    // 右值转换为IntList
    AT_ASSERT(isIntList());
    return moveToIntrusivePtr<ivalue::IntList>();
  }
  c10::intrusive_ptr<ivalue::IntList> toIntList() const & {  // 左值转换为IntList
    AT_ASSERT(isIntList());
    return toIntrusivePtr<ivalue::IntList>();
  }

  // 各种列表类型的引用访问方法
  const std::vector<int64_t>& toIntListRef() const;    // 获取整数列表引用
  const std::vector<double>& toDoubleListRef() const;  // 获取双精度列表引用
  const std::vector<bool>& toBoolListRef() const;      // 获取布尔列表引用
  const std::vector<at::Tensor>& toTensorListRef() const; // 获取张量列表引用
  const std::vector<IValue>& toGenericListRef() const; // 获取通用值列表引用
  const std::string& toStringRef() const;             // 获取字符串引用

  // ConstantString（常量字符串）类型相关方法
  IValue(c10::intrusive_ptr<ivalue::ConstantString> v); // 字符串构造函数(侵入式指针)
  IValue(std::string v);                                // 字符串构造函数(string对象)
  bool isString() const { return Tag::String == tag; }  // 检查是否为字符串
  c10::intrusive_ptr<ivalue::ConstantString> toString() && {  // 右值转换为字符串
    AT_ASSERT(isString());
    return moveToIntrusivePtr<ivalue::ConstantString>();
  }
  c10::intrusive_ptr<ivalue::ConstantString> toString() const & { // 左值转换为字符串
    AT_ASSERT(isString());
    return toIntrusivePtr<ivalue::ConstantString>();
  }

  // DoubleList（双精度列表）类型相关方法
  IValue(c10::intrusive_ptr<ivalue::DoubleList> v);  // DoubleList构造函数
  IValue(std::vector<double> v);                    // DoubleList构造函数(向量)
  bool isDoubleList() const { return Tag::DoubleList == tag; }  // 检查是否为双精度列表
  c10::intrusive_ptr<ivalue::DoubleList> toDoubleList() && {    // 右值转换
    AT_ASSERT(isDoubleList());
    return moveToIntrusivePtr<ivalue::DoubleList>();
  }
  c10::intrusive_ptr<ivalue::DoubleList> toDoubleList() const & {  // 左值转换
    AT_ASSERT(isDoubleList());
    return toIntrusivePtr<ivalue::DoubleList>();
  }

  // BoolList（布尔列表）类型相关方法
  IValue(c10::intrusive_ptr<ivalue::BoolList> v);  // BoolList构造函数
  IValue(std::vector<bool> v);                     // BoolList构造函数(向量)
  bool isBoolList() const { return Tag::BoolList == tag; }  // 检查是否为布尔列表
  c10::intrusive_ptr<ivalue::BoolList> toBoolList() && {    // 右值转换
    AT_ASSERT(isBoolList());
    return moveToIntrusivePtr<ivalue::BoolList>();
  }
  c10::intrusive_ptr<ivalue::BoolList> toBoolList() const & {  // 左值转换
    AT_ASSERT(isBoolList());
    return toIntrusivePtr<ivalue::BoolList>();
  }

  // TensorList（张量列表）类型相关方法
  IValue(c10::intrusive_ptr<ivalue::TensorList> v);  // TensorList构造函数
  IValue(std::vector<at::Tensor> v);                // TensorList构造函数(向量)
  bool isTensorList() const { return Tag::TensorList == tag; }  // 检查是否为张量列表
  c10::intrusive_ptr<ivalue::TensorList> toTensorList() && {   // 右值转换
    AT_ASSERT(isTensorList());
    return moveToIntrusivePtr<ivalue::TensorList>();
  }
  c10::intrusive_ptr<ivalue::TensorList> toTensorList() const & {  // 左值转换
    AT_ASSERT(isTensorList());
    return toIntrusivePtr<ivalue::TensorList>();
  }
  // GenericList（通用列表）类型相关方法
  IValue(c10::intrusive_ptr<ivalue::GenericList> v);  // 通用列表构造函数(侵入式指针)
  IValue(std::vector<IValue> v);                     // 通用列表构造函数(向量)
  bool isGenericList() const { return Tag::GenericList == tag; }  // 检查是否为通用列表
  c10::intrusive_ptr<ivalue::GenericList> toGenericList() && {   // 右值转换为通用列表
    AT_ASSERT(isGenericList());
    return moveToIntrusivePtr<ivalue::GenericList>();
  }
  c10::intrusive_ptr<ivalue::GenericList> toGenericList() const & {  // 左值转换为通用列表
    AT_ASSERT(isGenericList());
    return toIntrusivePtr<ivalue::GenericList>();
  }

  // None（空值）类型相关方法
  bool isNone() const {  // 检查是否为None
    return Tag::None == tag;
  }
  std::string toNone() const {  // 转换为None字符串表示
    AT_ASSERT(isNone());
    return "None";
  }
  
  // Scalar（标量）类型相关方法，会被编码为Int或Double
  IValue(at::Scalar s) : IValue() {  // 标量构造函数
    if(s.isFloatingPoint()) {  // 浮点类型存储为Double
      *this = s.toDouble();
    } else {                   // 整数类型存储为Int
      *this = s.toLong();
    }
  }
  bool isScalar() const {  // 检查是否为标量(包含Double/Int/Bool)
    return isDouble() || isInt() || isBool();
  }
  at::Scalar toScalar() const {  // 转换为标量值
    if(isDouble())
      return toDouble();
    else if(isInt())
      return toInt();
    else if (isBool())
      return int(toBool());
    throw std::runtime_error("IValue is not a Scalar");
  }

  // Device（设备）类型相关方法
  IValue(c10::Device d)  // 设备构造函数
  : tag(Tag::Device), is_intrusive_ptr(false) {
    payload.as_device.type = d.type();    // 存储设备类型
    payload.as_device.index = d.index();  // 存储设备索引
  }
  bool isDevice() const { return Tag::Device == tag; }  // 检查是否为设备
  c10::Device toDevice() const {  // 转换为设备对象
    AT_ASSERT(isDevice());
    return c10::Device(payload.as_device.type, payload.as_device.index);
  }

  // ScalarType（标量类型）转换
  at::ScalarType toScalarType() const {
    return static_cast<at::ScalarType>(toInt());  // 从整数转换为标量类型
  }

  // Layout（布局）转换
  at::Layout toLayout() const {
    return static_cast<at::Layout>(toInt());  // 从整数转换为布局类型
  }

  // 调试用：获取类型标签名称
  std::string tagKind() const {
    switch(tag) {
      #define DEFINE_CASE(x) case Tag::x: return #x;
      TORCH_FORALL_TAGS(DEFINE_CASE)  // 遍历所有标签类型
      #undef DEFINE_CASE
    }
    return "Invalid Tag";
  }

  // 通用模板转换方法，用于pop/push等特殊函数
  // 建议优先使用直接命名的方法(toTensor等)，更易于理解
  
  // 注意：如果出现链接错误说某个方法缺失，可以将其改为... && = delete;
  // 以获得更好的错误信息
  template<typename T>
  T to() &&;        // 右值模板转换
  template<typename T>
  T to() const &;   // 左值模板转换

  template<typename T>
  optional<T> toOptional();  // 转换为可选值

  // 浅比较两个IValue的对象标识
  bool isSameIdentity(IValue& rhs);

  // 输出运算符重载
  CAFFE2_API friend std::ostream& operator<<(
      std::ostream& out,
      const IValue& v);

  // 检查是否为指针类型
  bool isPtrType() const {
    return is_intrusive_ptr;
  }

 private:
  // 注意：IValue标签是私有的，未来可能使用不同的编码方式(如NaN boxing)
  // 这会使获取所有类型的标签比检查特定类型更耗时
  // 建议客户端尽可能使用isX方法
  enum class Tag : uint32_t {
#define DEFINE_TAG(x) x,
    TORCH_FORALL_TAGS(DEFINE_TAG)  // 定义所有标签类型
#undef DEFINE_TAG
  };

  // 移动转换为侵入式指针
  template<class T, class NullType = c10::detail::intrusive_target_default_null_type<T>>
  c10::intrusive_ptr<T, NullType> moveToIntrusivePtr() {
    auto t = c10::intrusive_ptr<T, NullType>::reclaim(static_cast<T*>(payload.as_intrusive_ptr));
    clearToNone();  // 转换后清空为None
    return t;
  }
  
  // 转换为侵入式指针(不移动)
  template<typename T, class NullType = c10::detail::intrusive_target_default_null_type<T>>
  c10::intrusive_ptr<T, NullType> toIntrusivePtr() const {
    auto r = c10::intrusive_ptr<T, NullType>::reclaim(static_cast<T*>(payload.as_intrusive_ptr));
    auto p = r;
    r.release();
    return p;
  }
  
  // 清空为None状态
  void clearToNone() {
    payload.as_int = 0;
    tag = Tag::None;
    is_intrusive_ptr = false;
  }
  
  // 联合体存储各种类型的值
  union {
    int64_t as_int;          // 存储整数
    double as_double;        // 存储双精度浮点数
    bool as_bool;            // 存储布尔值
    c10::intrusive_ptr_target* as_intrusive_ptr;  // 存储侵入式指针
    struct {                 // 存储设备信息
      DeviceType type;       // 设备类型
      DeviceIndex index;     // 设备索引
    } as_device;
  } payload;
  Tag tag;                  // 类型标签
  bool is_intrusive_ptr;    // 是否为侵入式指针
};

// Future（未来值）实现
struct C10_EXPORT ivalue::Future final : c10::intrusive_ptr_target {
 private:
  // 从this指针创建侵入式指针
  c10::intrusive_ptr<Future> intrusive_from_this() {
    c10::raw::intrusive_ptr::incref(this); // 增加引用计数
    return c10::intrusive_ptr<Future>::reclaim(this);
  }

 public:
  // 等待Future完成
  void wait() {
    if (completed()) {
      return;
    }
    c10::global_work_queue().workOnTasksUntilCompleted(intrusive_from_this());
    AT_ASSERT(completed());
  }

  // 标记Future为已完成状态
  void markCompleted(IValue value) {
    {
      // 加锁保护，防止与addCallback竞争
      std::unique_lock<std::mutex> lock(mutex_);
      AT_ASSERT(!completed());
      completed_ = true;
      value_ = std::move(value);
    }

    // 执行所有回调函数
    for (auto& callback : callbacks) {
      callback();
    }
    callbacks.clear();
  }

  // 获取Future的值
  IValue value() {
    std::unique_lock<std::mutex> lock(mutex_);
    AT_ASSERT(completed());
    return value_;
  }

  // 添加完成回调
  void addCallback(std::function<void(void)> callback) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (completed()) {  // 如果已完成立即执行回调
      lock.unlock();
      callback();
      return;
    }
    callbacks.push_back(callback);  // 否则保存回调
  }

  // 检查是否已完成
  bool completed() {
    return completed_;
  }

  // 获取互斥锁
  std::mutex& get_mutex() {
    return mutex_;
  }

  // 输出运算符重载
  CAFFE2_API friend std::ostream& operator<<(
      std::ostream& out,
      const Future& v);

 private:
  std::mutex mutex_;                  // 互斥锁
  IValue value_;                      // 存储的值
  std::atomic_bool completed_ = {false}; // 完成标志
  std::vector<std::function<void(void)>> callbacks; // 回调函数列表
};

// 定义各种类型的to模板特化
#define DEFINE_TO(type, method_name) \
template<> \
inline type IValue::to<type>() && { \
  return std::move(*this).method_name(); \
} \
template<> \
inline type IValue::to<type>() const & { \
  return this->method_name(); \
}
// 为各种类型定义to模板特化
DEFINE_TO(at::Tensor, toTensor)
DEFINE_TO(c10::intrusive_ptr<ivalue::Tuple>, toTuple)
DEFINE_TO(double, toDouble)
// ...其他类型特化...

// 各种类型的构造函数实现
inline IValue::IValue(c10::intrusive_ptr<ivalue::Tuple> v)
: tag(Tag::Tuple), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}

// ...其他构造函数实现...

// 列表引用访问方法实现
inline const std::vector<int64_t>& IValue::toIntListRef() const {
  return toIntList()->elements();
}
// ...其他列表引用访问方法...

// 转换为可选值
template<typename T>
inline optional<T> IValue::toOptional() {
  if (this->isNone()) {
    return nullopt;
  }
  return this->to<T>();
}

// 对象标识比较
inline bool IValue::isSameIdentity(IValue& rhs) {
  // 比较规则：
  // 1. None与None、False与False、True与True比较为true
  // 2. 张量类型需要考虑未定义张量情况
  // 3. 未定义张量与None比较为true
  // 4. 引用类型比较指针地址
  // 5. 其他情况返回false
  
  if (this->isNone() && rhs.isNone()) {
    return true;
  } else if (this->isBool() && rhs.isBool()) {
    return this->toBool() == rhs.toBool();
  } else if (this->isTensor() && rhs.isTensor()) {
    return this->payload.as_intrusive_ptr == rhs.payload.as_intrusive_ptr;
  } else if (this->isTensor() && rhs.isNone()) {
    return !this->is_intrusive_ptr;  // 未定义张量与None比较
  } else if (this->isNone() && rhs.isTensor()) {
    return !rhs.is_intrusive_ptr;    // None与未定义张量比较
  } else {
    return this->is_intrusive_ptr && rhs.is_intrusive_ptr
        && this->payload.as_intrusive_ptr == rhs.payload.as_intrusive_ptr;
  }
}