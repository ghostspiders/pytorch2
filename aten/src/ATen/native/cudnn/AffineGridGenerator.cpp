#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>

#if !AT_CUDNN_ENABLED()

namespace at { namespace native {

// 如果没有启用 cuDNN 支持，则直接报错

// 前向传播：仿射变换网格生成器
Tensor cudnn_affine_grid_generator_forward(
    const Tensor& theta,
    int64_t N, int64_t C, int64_t H, int64_t W) {
  AT_ERROR("cudnn_affine_grid_generator_forward: ATen not compiled with cuDNN support");
}

// 反向传播：仿射变换网格生成器
Tensor cudnn_affine_grid_generator_backward(
    const Tensor& grad_theta,
    int64_t N, int64_t C, int64_t H, int64_t W) {
  AT_ERROR("cudnn_affine_grid_generator_backward: ATen not compiled with cuDNN support");
}

}}

#else // AT_CUDNN_ENABLED()

#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/cuda/Exceptions.h>

#include <ATen/TensorUtils.h>

namespace at { namespace native {

namespace {

// 设置采样器描述符
void setSamplerDescriptor(SpatialTransformerDescriptor& desc,
                          cudnnDataType_t dataType,
                          int N, int C, int H, int W)
{
  int inputSize[4] = {N, C, H, W}; // 输入尺寸
  desc.set(dataType, 4, inputSize); // 设置描述符
}

}  // namespace

// 前向传播：仿射变换网格生成器
Tensor cudnn_affine_grid_generator_forward(
    const Tensor& theta_t,
    int64_t N, int64_t C, int64_t H, int64_t W)
{
  setCuDNNStreamToCurrent(); // 设置当前 CUDA 流

  // 检查 theta 的连续性和尺寸
  TensorArg theta{ theta_t.contiguous(), "theta", 1 };
  CheckedFrom c = "cudnn_affine_grid_generator_forward";
  checkContiguous(c, theta);
  checkSize(c, theta, {N, 2, 3});

  // 创建网格张量
  auto grid_t = at::empty({0}, theta->options());
  grid_t.resize_({N, H, W, 2});

  // 获取数据类型
  auto dataType = getCudnnDataType(*theta);
  SpatialTransformerDescriptor desc;
  setSamplerDescriptor(desc, dataType, N, C, H, W); // 设置采样器描述符
  AT_CUDNN_CHECK(cudnnSpatialTfGridGeneratorForward(getCudnnHandle(), desc.desc(),
                                                 theta->data_ptr(),
                                                 grid_t.data_ptr()));
  return grid_t;
}

// 反向传播：仿射变换网格生成器
Tensor cudnn_affine_grid_generator_backward(
    const Tensor& grad_grid_t,
    int64_t N, int64_t C, int64_t H, int64_t W)
{
  setCuDNNStreamToCurrent(); // 设置当前 CUDA 流

  // 检查 grad_grid 的连续性和尺寸
  TensorArg grad_grid{ grad_grid_t.contiguous(), "grad_grid", 1 };
  CheckedFrom c = "cudnn_affine_grid_generator_backward";
  checkContiguous(c, grad_grid);
  checkSize(c, grad_grid, {N, H, W, 2});

  // 创建梯度张量
  auto grad_theta_t = at::empty({0}, grad_grid->options());
  grad_theta_t.resize_({N, 2, 3});

  // 获取数据类型
  auto dataType = getCudnnDataType(grad_theta_t);
  SpatialTransformerDescriptor desc;
  setSamplerDescriptor(desc, dataType, N, C, H, W); // 设置采样器描述符
  AT_CUDNN_CHECK(cudnnSpatialTfGridGeneratorBackward(getCudnnHandle(), desc.desc(),
                                                  grad_grid->data_ptr(),
                                                  grad_theta_t.data_ptr()));
  return grad_theta_t;
}

}}  // namespace at::native

#endif // AT_CUDNN_ENABLED()