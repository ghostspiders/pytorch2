#include "ATen/DLConvertor.h"
#include "ATen/Functions.h"
#include <iostream>
#include <sstream>

using namespace std;
namespace at {

// 将ATen的Type转换为DLPack的数据类型
static DLDataType getDLDataType(const Type& type) {
  DLDataType dtype;
  dtype.lanes = 1;  // 通道数设为1(不支持向量化)
  dtype.bits = type.elementSizeInBytes() * 8;  // 计算位数(字节转比特)
  
  // 根据标量类型设置数据类型代码
  switch (type.scalarType()) {
    case ScalarType::Byte:
      dtype.code = DLDataTypeCode::kDLUInt;  // 无符号整型
      break;
    case ScalarType::Char:
      dtype.code = DLDataTypeCode::kDLInt;   // 有符号整型
      break;
    case ScalarType::Double:
    case ScalarType::Float:
    case ScalarType::Half:
      dtype.code = DLDataTypeCode::kDLFloat; // 浮点型
      break;
    case ScalarType::Int:
    case ScalarType::Long:
    case ScalarType::Short:
      dtype.code = DLDataTypeCode::kDLInt;   // 有符号整型
      break;
    case ScalarType::ComplexHalf:
    case ScalarType::ComplexFloat:
    case ScalarType::ComplexDouble:
      throw std::logic_error("Complex types are not supported by dlpack");  // 不支持复数类型
    case ScalarType::Undefined:
    case ScalarType::NumOptions:
      throw std::logic_error("Invalid ScalarType");  // 无效类型
  }
  return dtype;
}

// 获取DLPack上下文信息(设备类型和设备ID)
static DLContext getDLContext(const Type& type, const int64_t& device_id) {
  DLContext ctx;
  ctx.device_id = device_id;
  // 根据是否是CUDA张量设置设备类型
  if (type.is_cuda()) {
    ctx.device_type = DLDeviceType::kDLGPU;  // GPU设备
  } else {
    ctx.device_type = DLDeviceType::kDLCPU;  // CPU设备
  }
  return ctx;
}

// 将DLPack设备类型转换为ATen设备类型
static DeviceType getATenDeviceType(const DLContext& ctx) {
  switch (ctx.device_type) {
    case DLDeviceType::kDLCPU:
      return DeviceType::CPU;    // CPU
    case DLDeviceType::kDLGPU:
      return DeviceType::CUDA;   // CUDA GPU
    case DLDeviceType::kDLOpenCL:
      return DeviceType::OPENCL; // OpenCL
    case DLDeviceType::kDLROCM:
      return DeviceType::HIP;    // ROCm
    default:
      throw std::logic_error("Unsupported device_type");  // 不支持的设备类型
  }
  return DeviceType::CPU; // 不会执行到这里
}

// 将DLPack数据类型转换为ATen标量类型
ScalarType toScalarType(const DLDataType& dtype) {
  ScalarType stype;
  if (dtype.lanes != 1) throw std::logic_error("Vector types not supported");  // 不支持向量类型
  
  // 根据数据类型代码和位数确定标量类型
  switch (dtype.code) {
    case DLDataTypeCode::kDLUInt:  // 无符号整型
      if (dtype.bits == 8) {
        stype = ScalarType::Byte;  // 8位无符号
      } else {
        throw std::logic_error("Unsupported UInt bits");
      }
      break;
    case DLDataTypeCode::kDLInt:   // 有符号整型
      switch (dtype.bits) {
        case 8:  stype = ScalarType::Char;  break;  // 8位有符号
        case 16: stype = ScalarType::Short; break;  // 16位有符号
        case 32: stype = ScalarType::Int;   break;  // 32位有符号
        case 64: stype = ScalarType::Long;  break;  // 64位有符号
        default: throw std::logic_error("Unsupported Int bits");
      }
      break;
    case DLDataTypeCode::kDLFloat: // 浮点型
      switch (dtype.bits) {
        case 16: stype = ScalarType::Half;   break;  // 半精度浮点
        case 32: stype = ScalarType::Float;  break;  // 单精度浮点
        case 64: stype = ScalarType::Double; break;  // 双精度浮点
        default: throw std::logic_error("Unsupported Float bits");
      }
      break;
    default:
      throw std::logic_error("Unsupported data type code");
  }
  return stype;
}

// 用于管理DLManagedTensor的结构体
struct ATenDLMTensor {
  Tensor handle;         // 保存原始ATen张量
  DLManagedTensor tensor; // DLPack张量结构
};

// DLPack张量删除器
void deleter(DLManagedTensor * arg) {
  delete static_cast<ATenDLMTensor*>(arg->manager_ctx);  // 释放管理上下文
}

// 将ATen张量转换为DLPack张量
DLManagedTensor* toDLPack(const Tensor& src) {
  // 创建管理结构
  ATenDLMTensor * atDLMTensor(new ATenDLMTensor);
  atDLMTensor->handle = src;  // 保存原始张量
  atDLMTensor->tensor.manager_ctx = atDLMTensor;  // 设置管理上下文
  atDLMTensor->tensor.deleter = &deleter;        // 设置删除器
  
  // 填充DLTensor结构
  atDLMTensor->tensor.dl_tensor.data = src.data_ptr();  // 数据指针
  int64_t device_id = src.is_cuda() ? src.get_device() : 0;  // 设备ID
  atDLMTensor->tensor.dl_tensor.ctx = getDLContext(src.type(), device_id);  // 设备上下文
  atDLMTensor->tensor.dl_tensor.ndim = src.dim();  // 维度数
  atDLMTensor->tensor.dl_tensor.dtype = getDLDataType(src.type());  // 数据类型
  atDLMTensor->tensor.dl_tensor.shape = const_cast<int64_t*>(src.sizes().data());  // 形状
  atDLMTensor->tensor.dl_tensor.strides = const_cast<int64_t*>(src.strides().data());  // 步幅
  atDLMTensor->tensor.dl_tensor.byte_offset = 0;  // 字节偏移
  
  return &(atDLMTensor->tensor);
}

// 从DLPack张量创建ATen张量
Tensor fromDLPack(const DLManagedTensor* src) {
  // 获取设备类型和标量类型
  DeviceType device_type = getATenDeviceType(src->dl_tensor.ctx);
  ScalarType stype = toScalarType(src->dl_tensor.dtype);
  
  // 设置删除器，当ATen张量不再需要时调用DLPack的删除器
  auto deleter = [src](void * self) {
    src->deleter(const_cast<DLManagedTensor*>(src));
  };
  
  // 从DLPack数据创建ATen张量
  return at::from_blob(
      src->dl_tensor.data,  // 数据指针
      IntList(src->dl_tensor.shape, src->dl_tensor.ndim),  // 形状
      IntList(src->dl_tensor.strides, src->dl_tensor.ndim),  // 步幅
      deleter,  // 删除器
      at::device(device_type).dtype(stype)  // 设备和数据类型
  );
}
} // namespace at