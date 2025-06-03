/*!
 *  Copyright (c) 2017 by Contributors
 * \file dlpack.h
 * \brief DLPack通用头文件
 * 
 * DLPack是一个开放的张量数据结构标准，用于不同框架间的张量数据交换
 */
#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_

// 处理C++兼容性
#ifdef __cplusplus
#define DLPACK_EXTERN_C extern "C"  // 如果是C++，使用extern "C"修饰
#else
#define DLPACK_EXTERN_C             // 纯C环境不需要特殊处理
#endif

/*! \brief 当前DLPack版本号 (0.10) */
#define DLPACK_VERSION 010

/*! \brief Windows平台的DLL导出/导入修饰符 */
#ifdef _WIN32
#ifdef DLPACK_EXPORTS
#define DLPACK_DLL __declspec(dllexport)  // 导出符号
#else
#define DLPACK_DLL __declspec(dllimport)  // 导入符号
#endif
#else
#define DLPACK_DLL  // 非Windows平台空定义
#endif

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief DLContext中使用的设备类型枚举
 */
typedef enum {
  kDLCPU = 1,        // CPU设备
  kDLGPU = 2,        // CUDA GPU设备
  kDLCPUPinned = 3,  // 固定内存(pinned memory)，CPU和GPU都可访问
  kDLOpenCL = 4,     // OpenCL设备
  kDLMetal = 8,      // Apple Metal设备
  kDLVPI = 9,        // Verilog仿真器接口
  kDLROCM = 10,      // AMD ROCm GPU设备
} DLDeviceType;

/*!
 * \brief 张量和操作符的设备上下文结构体
 */
typedef struct {
  DLDeviceType device_type;  // 设备类型
  int device_id;             // 设备ID
} DLContext;

/*!
 * \brief DLDataType的类型代码枚举
 */
typedef enum {
  kDLInt = 0U,    // 有符号整数类型
  kDLUInt = 1U,   // 无符号整数类型
  kDLFloat = 2U,  // 浮点类型
} DLDataTypeCode;

/*!
 * \brief 张量可以持有的数据类型描述
 *
 * 示例:
 *   - float: type_code = 2, bits = 32, lanes=1
 *   - float4(4个float向量化): type_code = 2, bits = 32, lanes=4 
 *   - int8: type_code = 0, bits = 8, lanes=1
 */
typedef struct {
  /*!
   * \brief 基础类型代码
   * 使用uint8_t而非DLDataTypeCode以最小化内存占用，
   * 但值应为DLDataTypeCode枚举值之一
   */
  uint8_t code;
  /*!
   * \brief 位数，常见值为8,16,32
   */
  uint8_t bits;
  /*! \brief 类型中的通道数，用于向量化类型 */
  uint16_t lanes;
} DLDataType;

/*!
 * \brief 普通C张量对象，不管理内存
 */
typedef struct {
  /*!
   * \brief 指向已分配数据的不透明数据指针
   * 可以是CUDA设备指针或OpenCL中的cl_mem句柄
   * 此指针始终按照CUDA要求256字节对齐
   */
  void* data;
  /*! \brief 张量的设备上下文 */
  DLContext ctx;
  /*! \brief 维度数量 */
  int ndim;
  /*! \brief 指针的数据类型 */
  DLDataType dtype;
  /*! \brief 张量的形状 */
  int64_t* shape;
  /*!
   * \brief 张量的步幅(strides)
   * 可以为NULL，表示张量是紧凑的(contiguous)
   */
  int64_t* strides;
  /*! \brief 到数据起始指针的字节偏移量 */
  uint64_t byte_offset;
} DLTensor;

/*!
 * \brief 带内存管理的C张量对象，管理DLTensor的内存
 * 此数据结构旨在方便其他框架借用DLTensor，
 * 不是用于传输张量。当借用框架不再需要张量时，
 * 应调用deleter通知主机资源不再需要
 */
typedef struct DLManagedTensor {
  /*! \brief 被内存管理的DLTensor */
  DLTensor dl_tensor;
  /*! 
   * \brief 原始宿主框架的上下文指针
   * 在其中使用DLManagedTensor的框架上下文，
   * 也可以为NULL
   */
  void * manager_ctx;
  /*! 
   * \brief 析构函数签名 void (*)(void*)
   * 应调用此函数来销毁持有DLManagedTensor的manager_ctx
   * 如果调用者无法提供合理的析构函数，可以为NULL
   */
  void (*deleter)(struct DLManagedTensor * self);
} DLManagedTensor;

#ifdef __cplusplus
}  // DLPACK_EXTERN_C
#endif
#endif  // DLPACK_DLPACK_H_