#pragma once

/**
 * @file LegacyDeviceTypeInit.h
 * @brief 传统设备类型初始化机制
 * @details 该机制被LegacyTypeDispatch和LegacyTHDispatch共同使用，
 * 用于向后兼容不同设备类型的初始化操作。
 */

#include <c10/DeviceType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Registry.h>
#include <ATen/core/ScalarType.h>

namespace at {

/**
 * @class LegacyDeviceTypeInitInterface
 * @brief 传统设备类型初始化接口类
 * @note 该接口定义了各种设备类型的初始化方法，默认实现会抛出错误。
 * 具体实现应由各设备类型库提供。
 */
struct CAFFE2_API LegacyDeviceTypeInitInterface {
  virtual ~LegacyDeviceTypeInitInterface() = default;
  
  /**
   * @brief 初始化CPU设备
   * @throws 如果没有链接ATen库会抛出错误
   */
  virtual void initCPU() const { 
      AT_ERROR("cannot use CPU without ATen library");
  }
  
  /**
   * @brief 初始化CUDA设备
   * @throws 如果没有链接ATen CUDA库会抛出错误
   */
  virtual void initCUDA() const {
    AT_ERROR("cannot use CUDA without ATen CUDA library");
  }
  
  /**
   * @brief 初始化HIP设备
   * @throws 如果没有链接ATen HIP库会抛出错误
   */
  virtual void initHIP() const {
    AT_ERROR("cannot use HIP without ATen HIP library");
  }
  
  /**
   * @brief 初始化复数类型支持
   * @throws 如果没有链接ATen复数库会抛出错误
   */
  virtual void initComplex() const {
    AT_ERROR("cannot use complex without ATen Complex library");
  }
};

/**
 * @struct LegacyDeviceTypeInitArgs
 * @brief 传统设备类型初始化参数结构体
 * @note 目前为空结构体，保留未来扩展能力
 */
struct CAFFE2_API LegacyDeviceTypeInitArgs {};

/**
 * @macro C10_DECLARE_REGISTRY
 * @brief 声明传统设备类型初始化注册表
 * @param LegacyDeviceTypeInitRegistry 注册表名称
 * @param LegacyDeviceTypeInitInterface 注册的接口类型
 * @param LegacyDeviceTypeInitArgs 构造接口时传入的参数类型
 */
C10_DECLARE_REGISTRY(
    LegacyDeviceTypeInitRegistry,
    LegacyDeviceTypeInitInterface,
    LegacyDeviceTypeInitArgs);

/**
 * @def REGISTER_LEGACY_TYPE_INIT
 * @brief 注册传统类型初始化的宏
 * @param clsname 要注册的类名，同时作为注册表键名和实现类名
 */
#define REGISTER_LEGACY_TYPE_INIT(clsname) \
  C10_REGISTER_CLASS(LegacyDeviceTypeInitRegistry, clsname, clsname)

/**
 * @brief 获取传统设备类型初始化接口的单例引用
 * @return 返回LegacyDeviceTypeInitInterface的常量引用
 * @note 该函数保证线程安全，且会延迟初始化
 */
CAFFE2_API const LegacyDeviceTypeInitInterface& getLegacyDeviceTypeInit();

} // namespace at