#pragma once

// The legacy mechanism for dispatching operators in ATen is a Type
// object, which is essentially a giant virtual dispatch table
// for every operation we support dynamically dispatching over.
//
// We intend to deprecate this design for a more extensible one
// that permits addition of extra operators *out-of-band*.  However,
// for the time being, it's the only mechanism which works for
// dispatching PyTorch operators, so we are supporting it for now.
//
// The use of Type in ATen/core poses another problem: on a
// mobile build, we don't want to assume that Type is available.
// But all methods on Tensor which route to PyTorch operators
// need to somehow *get* a Type, and then do a virtual call on it.
// How are we going to get the Type?  Why, by another indirection!
//
// This registry is the mechanism for getting a concrete Type.
// For a regular build, we register all types here; for a
// mobile build, there are no registrations and instead we
// return a stub which errors for all functions.
//
// NB: We don't use Registry for this, because we don't want to
// pay for a hash table lookup every time we do an operation.

#include <c10/core/Backend.h>
#include <c10/core/ScalarType.h>
#include <ATen/core/VariableHooksInterface.h>
#include <c10/util/Exception.h>
#include <ATen/core/LegacyDeviceTypeInit.h>
#include <ATen/core/TensorImpl.h>

namespace at {

// 前向声明Type类
struct Type;

/**
 * @struct LegacyTypeDeleter
 * @brief 传统类型删除器，用于管理Type对象的生命周期
 * @note 包含一个函数指针，用于自定义Type对象的删除逻辑
 */
struct CAFFE2_API LegacyTypeDeleter {
  using TypeDeleterFun = void(Type*);  ///< Type删除函数类型定义
  TypeDeleterFun *fn_ = nullptr;      ///< 实际的删除函数指针
  
  LegacyTypeDeleter() = default;       ///< 默认构造函数
  
  /* implicit */ LegacyTypeDeleter(TypeDeleterFun *fn) : fn_(fn) {}  ///< 带函数指针的构造函数
  
  /**
   * @brief 重载()运算符，执行删除操作
   * @param ptr 要删除的Type指针
   */
  void operator()(Type * ptr) {
    if (fn_) {
      (*fn_)(ptr);  // 调用自定义删除函数
    }
  }
};

/**
 * @class LegacyTypeDispatch
 * @brief 传统类型分发系统，负责管理和分发不同类型的计算后端
 */
class CAFFE2_API LegacyTypeDispatch {
 public:
  using TypeUniquePtr = std::unique_ptr<Type, LegacyTypeDeleter>;  ///< Type唯一指针类型定义
  
  /**
   * @brief 获取原始非变量类型指针
   * @param p 计算后端类型
   * @param s 标量类型
   * @return 对应的Type指针
   * @warning 调用前必须确保所需类型已初始化
   */
  Type* getNonVariableTypeRaw(Backend p, ScalarType s) {
    return type_registry[static_cast<int>(p)][static_cast<int>(s)].get();
  }
  
  /**
   * @brief 获取可选的非变量类型指针
   * @param p 计算后端类型
   * @param s 标量类型
   * @return 对应的Type指针，可能为nullptr
   * @note 会自动初始化所需的设备类型和标量类型
   */
  Type * getNonVariableTypeOpt(Backend p, ScalarType s) {
    if (p != Backend::Undefined) {
      initForDeviceType(backendToDeviceType(p));  // 初始化设备类型
      initForScalarType(s);                      // 初始化标量类型
    }
    auto type = getNonVariableTypeRaw(p, s);

    if(!type) {
      // 对于Undefined类型，返回统一的Undefined Type
      if (p == Backend::Undefined || s == ScalarType::Undefined) {
        return getNonVariableTypeRaw(Backend::Undefined, ScalarType::Undefined);
      }
    }

    return type;
  }

  /**
   * @brief 获取非变量类型引用
   * @param p 计算后端类型
   * @param s 标量类型
   * @return 对应的Type引用
   * @throw 如果类型未启用则抛出错误
   */
  Type & getNonVariableType(Backend p, ScalarType s) {
    auto* type = getNonVariableTypeOpt(p, s);
    if (!type) AT_ERROR(toString(p), toString(s), "Type is not enabled.");
    return *type;
  }

  /**
   * @brief 获取原始类型指针
   * @param p 计算后端类型
   * @param s 标量类型
   * @param is_variable 是否为变量类型
   * @return 对应的Type指针
   */
  Type* getTypeRaw(Backend p, ScalarType s, bool is_variable) {
    auto baseType = getNonVariableTypeRaw(p, s);
    if (is_variable) {
      return &detail::getVariableHooks().getVariableTypeFromBaseType(*baseType);
    } else {
      return baseType;
    }
  }

  /**
   * @brief 获取变量类型引用
   * @param p 计算后端类型
   * @param s 标量类型
   * @return 对应的变量Type引用
   */
  Type & getVariableType(Backend p, ScalarType s) {
    auto& baseType = getNonVariableType(p, s);
    return detail::getVariableHooks().getVariableTypeFromBaseType(baseType);
  }

  /**
   * @brief 获取类型引用
   * @param p 计算后端类型
   * @param s 标量类型
   * @param is_variable 是否为变量类型
   * @return 对应的Type引用
   */
  Type & getType(Backend p, ScalarType s, bool is_variable) {
    if (is_variable) {
      return getVariableType(p, s);
    } else {
      return getNonVariableType(p, s);
    }
  }

  /**
   * @brief 注册类型
   * @param b 计算后端类型
   * @param s 标量类型
   * @param t 要注册的Type对象
   */
  void registerType(Backend b, ScalarType s, TypeUniquePtr&& t) {
    type_registry[static_cast<int>(b)][static_cast<int>(s)] = std::move(t);
    detail::getVariableHooks().registerVariableTypeFor(this, b, s);
  }

private:
  /**
   * @brief 初始化特定设备类型
   * @param p 设备类型
   */
  void initForDeviceType(DeviceType p) {
    static std::once_flag cpu_once;
    static std::once_flag cuda_once;
    if (p == DeviceType::CPU) {
      std::call_once(cpu_once, [] {
        getLegacyDeviceTypeInit().initCPU();  // 初始化CPU
      });
    } else if (p == DeviceType::CUDA) {
      std::call_once(cuda_once, [] {
        getLegacyDeviceTypeInit().initCUDA(); // 初始化CUDA
      });
    } else if (p == DeviceType::HIP) {
      std::call_once(cuda_once, [] {
        getLegacyDeviceTypeInit().initHIP();  // 初始化HIP
      });
    }
  }

  /**
   * @brief 初始化特定标量类型
   * @param s 标量类型
   */
  void initForScalarType(ScalarType s) {
    static std::once_flag once;
    // 只有复数类型需要初始化
    if (isComplexType(s)) {
      std::call_once(once, [] {
        getLegacyDeviceTypeInit().initComplex();  // 初始化复数支持
      });
    }
  }

  // 类型注册表，使用二维数组存储不同后端和标量类型的Type对象
  // 注意：在CUDA初始化完成前，所有CUDA后端对应的位置都是nullptr
  TypeUniquePtr type_registry
    [static_cast<int>(Backend::NumOptions)]
    [static_cast<int>(ScalarType::NumOptions)];
};

/**
 * @brief 获取全局传统类型分发实例
 * @return 全局LegacyTypeDispatch引用
 */
CAFFE2_API LegacyTypeDispatch& globalLegacyTypeDispatch();

/**
 * @brief 获取Tensor对应的传统Type对象
 * @param tensor 要查询的Tensor实现
 * @return 对应的Type引用
 * @note 这是内部实现细节，不推荐最终用户使用
 */
inline Type& legacyTensorType(const TensorImpl& tensor) {
  // 可以直接使用getTypeRaw，因为创建TensorImpl必须先初始化对应的Type
  // TODO: 通过Caffe2路径创建的TensorImpl可能不满足这个前提条件
  return *globalLegacyTypeDispatch().getTypeRaw(
    tensorTypeIdToBackend(tensor.type_id()),
    typeMetaToScalarType(tensor.dtype()),
    tensor.is_variable()
  );
}

} // namespace at