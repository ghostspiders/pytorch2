#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>

#if !AT_CUDNN_ENABLED()

namespace at { namespace native {

// 如果没有启用 cuDNN 支持，则直接报错

// cuDNN 批量归一化前向传播
std::tuple<Tensor, Tensor, Tensor> cudnn_batch_norm(
    const Tensor& input, const Tensor& weight,
    const Tensor& bias, const Tensor& running_mean, const Tensor& running_var,
    bool training, double exponential_average_factor, double epsilon) {
  AT_ERROR("cudnn_batch_norm: ATen not compiled with cuDNN support");
}

// cuDNN 批量归一化反向传播
std::tuple<Tensor, Tensor, Tensor> cudnn_batch_norm_backward(
    const Tensor& input, const Tensor& grad_output, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_var,
    double epsilon) {
  AT_ERROR("cudnn_batch_norm_backward: ATen not compiled with cuDNN support");
}

}}  // namespace at::native

#else // AT_CUDNN_ENABLED

#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/cuda/Exceptions.h>

#include <ATen/TensorUtils.h>

namespace at { namespace native {

namespace {

// 扩展尺度张量
Tensor expandScale(const Tensor& t, int64_t dim) {
  std::vector<int64_t> size{ 1, t.numel() };
  while (static_cast<int64_t>(size.size()) < dim) {
    size.emplace_back(1);
  }
  return t.view(size);
}

}  // namespace

// cuDNN 批量归一化前向传播
std::tuple<Tensor, Tensor, Tensor> cudnn_batch_norm(
    const Tensor& input_t, const Tensor& weight_t,
    const Tensor& bias_t, const Tensor& running_mean_t, const Tensor& running_var_t,
    bool training, double exponential_average_factor, double epsilon)
{
  // 检查输入张量
  TensorArg input{ input_t, "input", 1 },
            weight{ weight_t, "weight", 2 },
            bias{ bias_t, "bias", 3 },
            running_mean{ running_mean_t, "running_mean", 4 },
            running_var{ running_var_t, "running_var", 5 };
  CheckedFrom c = "cudnn_batch_norm";
  setCuDNNStreamToCurrent(); // 设置当前 CUDA 流

  // 检查输入张量是否定义
  checkAllDefined(c, {input, weight, bias});
  if (!training) {
    checkAllDefined(c, {running_mean, running_var});
  }
  // 检查输入张量是否在同一 GPU 上
  checkAllSameGPU(c, {input, weight, bias, running_mean, running_var});
  // 检查输入张量的数据类型
  if (input->type().scalarType() == ScalarType::Half) {
    checkScalarType(c, weight, ScalarType::Float);
  } else {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {weight, bias, running_mean, running_var});
  // 检查输入张量是否连续
  checkAllContiguous(c, {input, weight, bias, running_mean, running_var});
  // 检查输入张量的维度
  checkDimRange(c, input, 2, 6 /* exclusive */);
  // 检查特征数量是否一致
  auto num_features = input->size(1);
  for (auto t : {weight, bias, running_mean, running_var}) {
    if (t->defined()) {
      checkNumel(c, t, num_features);
    }
  }

  // 选择批量归一化模式
  cudnnBatchNormMode_t mode;
  if (input->dim() == 2) {
    mode = CUDNN_BATCHNORM_PER_ACTIVATION; // 每个激活函数单独归一化
  } else {
    mode = CUDNN_BATCHNORM_SPATIAL; // 空间批量归一化
  }

  // 创建输出张量
  auto output_t = at::empty(input->sizes(), input->options());
  TensorArg output{ output_t, "output", 0 };

  // 获取 cuDNN 句柄和数据类型
  auto handle = getCudnnHandle();
  auto dataType = getCudnnDataType(*input);
  TensorDescriptor idesc{ *input, 4 };  // 输入描述符
  TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // 权重、偏置、均值等的描述符

  // 创建常量
  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  Tensor save_mean, save_var;

  // 前向传播
  if (training) {
    int64_t num_features = input_t.size(1);
    save_mean = at::empty({ num_features }, weight_t.options());
    save_var = at::empty({ num_features }, weight_t.options());
    AT_CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
      handle, mode, &one, &zero,
      idesc.desc(), input->data_ptr(),
      idesc.desc(), output->data_ptr(),
      wdesc.desc(),
      weight->data_ptr(),
      bias->data_ptr(),
      exponential_average_factor,
      at::maybe_data_ptr(running_mean),
      at::maybe_data_ptr(running_var),
      epsilon,
      save_mean.data_ptr(),
      save_var.data_ptr()));
  } else {
    AT_CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
      handle, mode, &one, &zero,
      idesc.desc(), input->data_ptr(),
      idesc.desc(), output->data_ptr(),
      wdesc.desc(),
      weight->data_ptr(),
      bias->data_ptr(),
      running_mean->data_ptr(),
      running_var->data_ptr(),
      epsilon));
  }

  // 返回输出张量、保存的均值和方差
  return std::tuple<Tensor, Tensor, Tensor>{output_t, save_mean, save_var};
}

// cuDNN 批量归一化反向传播
std::tuple<Tensor, Tensor, Tensor> cudnn_batch_norm_backward(
    const Tensor& input_t, const Tensor& grad_output_t, const Tensor& weight_t,
    // 未使用：但需要传递以便双反向传播
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean_t, const Tensor& save_var_t,
    double epsilon)
{
  // 检查输入张量
  TensorArg input{ input_t, "input", 1 },
            grad_output{ grad_output_t, "grad_output", 2 },
            weight{ weight_t, "weight", 3 },
            save_mean{ save_mean_t, "save_mean", 4 },
            save_var{ save_var_t, "save_var", 5 };
  CheckedFrom c = "cudnn_batch_norm_backward";
  setCuDNNStreamToCurrent(); // 设置当前 CUDA 流

  // 检查输入张量是否定义
  checkAllDefined(c, {input, grad_output, weight, save_mean, save_var});
  // 检查输入张量是否在同一 GPU 上
  checkAllSameGPU(c, {input, grad_output, weight, save_mean, save_var});
  // 检查输入张量的数据类型
  if (input->type().scalarType() == ScalarType::Half) {
    checkScalarType(c, weight, ScalarType::Float);
  } else {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {input, grad_output});
  checkAllSameType(c, {weight, save_mean, save_var});
  // 检查输入张量是否连续
  checkAllContiguous(c, {input, grad_output, save_mean, save_var});
  // 检查输入张量的维度
  checkDimRange(c, input, 2, 6 /* exclusive */);
  checkSameSize(c, input, grad_output); // 检查输入和梯度输出的大小是否一致
  // 检查特征数量是否一致
  auto num_features = input->size(1);
  for (auto t : {weight, save_mean, save_var}) {
    checkNumel(c, t, num_features);
  }

  // 选择批量归一化模式
  cudnnBatchNormMode_t mode;
  if (input->dim() == 2) {
    mode = CUDNN_BATCHNORM_PER_ACTIVATION; // 每个激活函数单独归一化
  } else {
    mode = CUDNN_BATCHNORM_SPATIAL; // 空间批量归一化
  }

  //