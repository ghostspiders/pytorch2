#pragma once

#include <c10/core/dispatch/DispatchKey.h>
#include <c10/util/Array.h>
#include <c10/util/Metaprogramming.h>
#include <c10/DeviceType.h>

namespace caffe2 {
class Tensor;
}  // namespace caffe2

namespace c10 {

namespace details {

/**
 * 检查类型 Arg 是否是 Tensor 或其引用类型。
 * 如果是，则成员常量 value 为 true，否则为 false。
 */
template <class Arg>
using is_tensor_arg = std::
    is_same<caffe2::Tensor, guts::remove_cv_t<guts::remove_reference_t<Arg>>>;

/**
 * 将 DeviceType 转换为 DeviceTypeId。
 * 用于统一设备类型的标识。
 */
inline DeviceTypeId to_device_type_id(DeviceType device_type) {
  switch (device_type) {
    case DeviceType::CPU:
      return DeviceTypeId::CPU;
    case DeviceType::CUDA:
      return DeviceTypeId::CUDA;
    default:
      return DeviceTypeId::UNDEFINED;
  }
}

/**
 * 将 Tensor 转换为 DispatchKey 的辅助工具。
 * 提取 Tensor 的设备类型、布局和数据类型，组合成 DispatchKey。
 */
struct tensor_to_dispatch_key final {
    template<class TensorType>
    TensorParameterDispatchKey operator()(const TensorType& tensor) const {
      return TensorParameterDispatchKey{
          to_device_type_id(tensor.GetDeviceType()),
          LayoutId(0),
          tensor.dtype().id()};
    }
};

/**
 * 从可变参数列表中提取所有 Tensor 参数的类型信息。
 * 返回一个包含 Tensor 参数 DispatchKey 的数组。
 */
template<class... Args> auto getTensorTypeIds_(const Args&... args)
-> guts::array<TensorParameterDispatchKey, guts::typelist::count_if<is_tensor_arg, guts::typelist::typelist<Args...>>::value> {
  return guts::filter_map<TensorParameterDispatchKey, is_tensor_arg>(tensor_to_dispatch_key(), args...);
}

/**
 * 检查类型 T 是否定义了 Signature 类型。
 * 用于编译期验证算子定义是否合法。
 */
template<class T, typename = void>
struct has_signature_defined : std::false_type {};
template<class T>
struct has_signature_defined<T, guts::void_t<
  typename T::Signature
>> : std::true_type {};

/**
 * 检查类型 T 是否定义了 parameter_names 成员。
 * 用于验证算子是否提供了参数名称列表。
 */
template<class T, typename = void>
struct has_parameter_names_defined : std::false_type {};
template<class T>
struct has_parameter_names_defined<T, guts::void_t<
  decltype(T::parameter_names)
>> : std::true_type {};

/**
 * 检查类型 T 是否定义了 name 成员。
 * 用于验证算子是否提供了名称。
 */
template<class T, typename = void>
struct has_name_defined : std::false_type {};
template<class T>
struct has_name_defined<T, guts::void_t<
        decltype(T::name)
>> : std::true_type {};

/**
 * 封装算子签名的解析功能。
 * 提取算子的函数类型、返回值类型、参数类型等信息。
 */
template<class OpSchemaDef> class OpSignatureSchema final {
  static_assert(details::has_signature_defined<OpSchemaDef>::value, "Operator schema doesn't define a valid Signature member type.");
  static_assert(guts::is_function_type<typename OpSchemaDef::Signature>::value, "Signature member of operator schema must be a function type.");

  using signature_traits = guts::function_traits<typename OpSchemaDef::Signature>;
public:
  using func_type = typename signature_traits::func_type;  // 算子函数类型
  using return_type = typename signature_traits::return_type;  // 返回值类型
  using parameter_types = typename signature_traits::parameter_types;  // 参数类型列表

  static constexpr size_t num_args = guts::typelist::size<parameter_types>::value;  // 参数数量
  static constexpr size_t num_tensor_args = guts::typelist::count_if<details::is_tensor_arg, parameter_types>::value;  // Tensor 参数数量

private:
  static_assert(details::has_parameter_names_defined<OpSchemaDef>::value, "Operator schema doesn't define parameter_names member.");
  static_assert(std::is_same<const guts::array<const char*, num_args>, decltype(OpSchemaDef::parameter_names)>::value, "Operator schema defines parameter_names member, but it isn't the correct type. Must be a static constexpr guts::array of const char* with one entry for each parameter.");

public:
  static constexpr const guts::array<const char*, num_args>& parameter_names() {
    return OpSchemaDef::parameter_names;  // 返回参数名称列表
  }
};

/**
 * 检查类型 T 是否定义了 dispatch_key 方法。
 * 用于判断算子是否自定义了 DispatchKey 生成逻辑。
 */
template<class T, typename = void>
struct has_function_dispatch_key_defined : std::false_type {};
template<class T>
struct has_function_dispatch_key_defined<T, guts::void_t<
  decltype(&T::dispatch_key)
>> : std::true_type {};

/**
 * 默认的 DispatchKey 生成器。
 * 根据 Tensor 参数自动生成 DispatchKey。
 */
template<class OpSchemaDef>
class OpDispatchKeySchema<OpSchemaDef, guts::enable_if_t<!has_function_dispatch_key_defined<OpSchemaDef>::value>> final {
  using signature = OpSignatureSchema<OpSchemaDef>;
public:
  using dispatch_key_type = DispatchKey<signature::num_tensor_args>;

  template<class... Args>
  static inline dispatch_key_type dispatch_key(const Args&... args) {
    // 检查参数类型是否匹配算子签名
    static_assert(std::is_same<
      guts::typelist::map_t<guts::remove_cv_t, guts::typelist::map_t<guts::remove_reference_t, guts::typelist::typelist<Args...>>>,
      guts::typelist::map_t<guts::remove_cv_t, guts::typelist::map_t<guts::remove_reference_t, typename signature::parameter_types>>
      >::value, "Invalid argument types passed to OpSchema::dispatch_key()");
    return dispatch_key_type {
      details::getTensorTypeIds_(args...)  // 提取 Tensor 参数的类型信息
    };
  }
};

/**
 * 自定义 DispatchKey 生成器。
 * 使用算子定义的 dispatch_key 方法生成 DispatchKey。
 */
template<class OpSchemaDef>
class OpDispatchKeySchema<OpSchemaDef, guts::enable_if_t<has_function_dispatch_key_defined<OpSchemaDef>::value>> final {
  using signature = OpSignatureSchema<OpSchemaDef>;

  static_assert(guts::is_function_type<decltype(OpSchemaDef::dispatch_key)>::value, "Operator schema defines dispatch_key member, but it isn't a function.");

  using dispatch_key_traits = guts::function_traits<decltype(OpSchemaDef::dispatch_key)>;

public:
  using dispatch_key_type = typename dispatch_key_traits::return_type;

private:
  // 检查 DispatchKey 类型是否支持比较和哈希
  static_assert(guts::is_equality_comparable<dispatch_key_type>::value, "Operator schema specified custom dispatch_key() derivation function, but the returned dispatch key type doesn't have the equality operator defined. Please define it.");
  static_assert(guts::is_hashable<dispatch_key_type>::value, "Operator schema specified custom dispatch_key() derivation function, but the returned dispatch key type doesn't have an overload for std::hash. Please define it.");

  // 检查自定义 dispatch_key 方法的参数是否匹配算子签名
  static_assert(std::is_same<
    guts::typelist::map_t<guts::remove_cv_t, guts::typelist::map_t<guts::remove_reference_t, typename dispatch_key_traits::parameter_types>>,
    guts::typelist::map_t<guts::remove_cv_t, guts::typelist::map_t<guts::remove_reference_t, typename signature::parameter_types>>
    >::value, "Operator schema defines custom dispatch_key() derivation function, but the arguments don't match the operator signature.");

public:
  template<class... Args>
  static inline dispatch_key_type dispatch_key(const Args&... args) {
    // 检查参数类型是否匹配算子签名
    static_assert(std::is_same<
      guts::typelist::map_t<guts::remove_cv_t, guts::typelist::map_t<guts::remove_reference_t, guts::typelist::typelist<Args...>>>,
      guts::typelist::map_t<guts::remove_cv_t, guts::typelist::map_t<guts::remove_reference_t, typename signature::parameter_types>>
      >::value, "Invalid argument types passed to OpSchema::dispatch_key()");
    return OpSchemaDef::dispatch_key(args...);  // 调用自定义的 dispatch_key 方法
  }
};

/**
 * 封装算子的元信息，如算子名称。
 */
template<class OpSchemaDef>
class OpMetadataSchema final {
private:
    static_assert(has_name_defined<OpSchemaDef>::value, "The operator schema has to define a 'static constexpr const char* name = ...' member to specify the operator name.");
    static_assert(std::is_same<const char* const, decltype(OpSchemaDef::name)>::value, "The 'name' member of the operator schema must have type 'static constexpr const char*'");

public:
    static constexpr const char* name() {
        return OpSchemaDef::name;  // 返回算子名称
    }
};

}  // namespace details

/**
 * 对外的算子 Schema 接口。
 * 组合了元信息、签名解析和派发功能。
 */
template <class OpSchemaDef>
class CAFFE2_API OpSchema final {
public:
  using metadata = details::OpMetadataSchema<OpSchemaDef>;  // 元信息
  using signature = details::OpSignatureSchema<OpSchemaDef>;  // 签名解析
  using dispatch = details::OpDispatchKeySchema<OpSchemaDef>;  // 派发逻辑
};

}  // namespace c10