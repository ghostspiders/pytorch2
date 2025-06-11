#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>

// 条件编译：检查是否启用cuDNN支持
#if !AT_CUDNN_ENABLED()

namespace at { namespace native {

// 注意：[ATen预处理器哲学]
// 未启用cuDNN时的桩函数实现

// 网格采样器前向传播
Tensor cudnn_grid_sampler_forward(
    const Tensor& input_t, const Tensor& grid_t) {
  AT_ERROR("cudnn_grid_sampler_forward: ATen not compiled with cuDNN support");
}

// 网格采样器反向传播
std::tuple<Tensor, Tensor> cudnn_grid_sampler_backward(
    const Tensor& input_t, const Tensor& grid_t,
    const Tensor& grad_output_t) {
  AT_ERROR("cudnn_grid_sampler_backward: ATen not compiled with cuDNN support");
}

}}

#else // AT_CUDNN_ENABLED（启用cuDNN时的实际实现）

#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/cuda/Exceptions.h>

#include <ATen/TensorUtils.h>

namespace at { namespace native {

namespace { // 匿名命名空间（内部辅助函数）

// 设置空间变换器描述符
void setSamplerDescriptor(SpatialTransformerDescriptor& desc, cudnnDataType_t dataType, const at::Tensor& tensor)
{
  int inputSize[4] = {0};
  // 将Tensor维度转换为cuDNN需要的格式
  for (int i = 0; i < tensor.dim(); ++i) {
    inputSize[i] = (int) tensor.size(i);
  }
  desc.set(dataType, 4, inputSize);
}

// 检查网格尺寸是否有效
void checkGridSize(CheckedFrom c, TensorArg grid, TensorArg input)
{
  // 检查网格张量：应为4维 [N, H, W, 2]
  checkContiguous(c, grid);  // 必须连续内存
  checkDim(c, grid, 4);      // 必须是4维张量
  // 检查维度0（批大小）匹配
  checkSize(c, grid, 0, input->size(0));
  // 检查最后一维必须为2（坐标点）
  checkSize(c, grid, 3, 2);
}

}  // 匿名命名空间结束

// 网格采样器前向传播实现
Tensor cudnn_grid_sampler_forward(
    const Tensor& input_t, const Tensor& grid_t)
{
  // 准备输入参数（确保连续内存）
  TensorArg input{ contiguousIfZeroInStrides(input_t), "input", 1 },
            grid{ grid_t.contiguous(), "grid", 2 };
  CheckedFrom c = "cudnn_grid_sampler_forward";
  
  setCuDNNStreamToCurrent();  // 设置当前CUDA流
  checkAllSameGPU(c, {input, grid});  // 检查同设备
  checkAllSameType(c, {input, grid}); // 检查同数据类型
  checkGridSize(c, grid, input);      // 检查网格尺寸
  checkDim(c, input, 4);              // 输入必须是4维 [N, C, H, W]

  // 创建输出张量 [N, C, grid_H, grid_W]
  auto output_t = at::empty({0}, input->options());
  output_t.resize_({input->size(0), input->size(1), grid->size(1), grid->size(2)});

  // 创建cuDNN描述符
  TensorDescriptor idesc{ *input };   // 输入描述符
  TensorDescriptor odesc{ output_t }; // 输出描述符
  SpatialTransformerDescriptor desc;  // 空间变换描述符

  auto handle = getCudnnHandle();     // 获取cuDNN句柄
  auto dataType = getCudnnDataType(*input); // 获取数据类型
  setSamplerDescriptor(desc, dataType, output_t); // 设置采样器描述符

  // 准备缩放因子
  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  
  // 调用cuDNN前向传播函数
  AT_CUDNN_CHECK(cudnnSpatialTfSamplerForward(
      handle, desc.desc(),             // cuDNN句柄和描述符
      &one, idesc.desc(), input->data_ptr(),  // 输入数据和缩放因子
      grid->data_ptr(),                // 网格数据
      &zero, odesc.desc(), output_t.data_ptr() // 输出数据和缩放因子
  ));

  return output_t;
}

// 网格采样器反向传播实现
std::tuple<Tensor, Tensor> cudnn_grid_sampler_backward(
    const Tensor& input_t, const Tensor& grid_t,
    const Tensor& grad_output_t)
{
  // 准备输入参数（确保连续内存）
  TensorArg input{ contiguousIfZeroInStrides(input_t), "input", 1 },
            grid{ grid_t.contiguous(), "grid", 2 },
            grad_output{ contiguousIfZeroInStrides(grad_output_t), "grad_output", 3 };
  CheckedFrom c = "cudnn_grid_sampler_backward";
  
  setCuDNNStreamToCurrent();  // 设置当前CUDA流
  checkAllSameGPU(c, {input, grad_output, grid}); // 检查同设备
  checkGridSize(c, grid, input);  // 检查网格尺寸
  checkDim(c, input, 4);          // 输入维度检查
  checkDim(c, grad_output, 4);    // 梯度输出维度检查

  // 创建输入梯度和网格梯度张量
  auto grad_input_t = at::empty({0}, input->options());
  grad_input_t.resize_(input->sizes());  // [N, C, H, W]
  auto grad_grid_t = at::empty({0}, grid->options());
  grad_grid_t.resize_(grid->sizes());    // [N, H, W, 2]

  // 创建cuDNN描述符
  TensorDescriptor idesc{ *input };      // 输入描述符
  TensorDescriptor odesc{ *grad_output }; // 梯度输出描述符
  TensorDescriptor gdesc{ grad_input_t }; // 梯度输入描述符
  SpatialTransformerDescriptor desc;     // 空间变换描述符

  auto handle = getCudnnHandle();        // 获取cuDNN句柄
  auto dataType = getCudnnDataType(*input); // 获取数据类型
  setSamplerDescriptor(desc, dataType, *grad_output); // 设置采样器描述符

  // 准备缩放因子
  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  
  // 调用cuDNN反向传播函数
  AT_CUDNN_CHECK(cudnnSpatialTfSamplerBackward(
    handle, desc.desc(),               // cuDNN句柄和描述符
    &one, idesc.desc(), input->data_ptr(), // 输入数据
    &zero, gdesc.desc(), grad_input_t.data_ptr(), // 输入梯度
    &one, odesc.desc(), grad_output->data_ptr(), // 梯度输出
    grid->data_ptr(),                  // 原始网格
    &zero, grad_grid_t.data_ptr()      // 网格梯度
  ));

  return std::tuple<Tensor, Tensor>{ grad_input_t, grad_grid_t };
}

}}  // namespace at::native

#endif // AT_CUDNN_ENABLED结束