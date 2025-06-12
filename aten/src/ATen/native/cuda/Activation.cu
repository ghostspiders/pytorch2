#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"  // CUDA应用工具
#include "ATen/cuda/detail/IndexUtils.cuh" // 索引工具
#include "ATen/native/Activation.h"       // 激活函数基类
#include "ATen/native/cuda/Loops.cuh"     // CUDA循环模板

namespace at { namespace native {

// ==============================================================
// PReLU 前向传播
// ==============================================================

// 共享权重的PReLU实现（所有通道使用同一个权重）
template <typename scalar_t>
void prelu_cuda_kernel_share_weights(
  const Tensor& input,
  Tensor& result,
  const scalar_t* weight_data) {  // 权重指针

  // 使用CUDA张量应用模板处理每个元素
  at::cuda::CUDA_tensor_apply2<scalar_t, scalar_t>(
    input,
    result,
    [=] __device__ (  // GPU设备lambda
      const scalar_t& input_val,
      scalar_t& result_val) {
        // PReLU公式: output = input > 0 ? input : weight * input
        result_val = (input_val > 0) ? input_val : *weight_data * input_val;
  });
}

// 多权重的PReLU实现（每个通道有独立权重）
template <typename scalar_t>
__global__ void prelu_cuda_kernel_multi_weights(
  scalar_t* result_data,
  const scalar_t* input_data,
  const scalar_t* weight_data,
  int64_t input_stride0,  // 第0维步长（批次内元素数）
  int64_t input_stride1,  // 第1维步长（通道内元素数）
  int64_t input_numel) {  // 输入元素总数

  // 计算线性索引
  int64_t linearId = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearId >= input_numel) return;  // 边界检查

  // 计算通道索引: (线性索引 % 输入步长0) / 输入步长1
  int64_t channel = (linearId % input_stride0) / input_stride1;
  scalar_t input_data_val = input_data[linearId];
  
  // 应用PReLU公式
  result_data[linearId] = (input_data_val > 0) 
    ? input_data_val 
    : weight_data[channel] * input_data_val;
}

// PReLU主函数
Tensor prelu_cuda(const Tensor& self, const Tensor& weight_) {
  // 检查输入条件
  AT_CHECK(self.is_cuda(), "输入张量必须在CUDA设备上");
  AT_CHECK(weight_.is_cuda(), "权重张量必须在CUDA设备上");

  // 确保张量内存连续
  auto input = self.contiguous();
  auto weight = weight_.contiguous();

  AT_CHECK(input.is_contiguous(), "输入张量必须是连续的");
  AT_CHECK(weight.is_contiguous(), "权重张量必须是连续的");

  int64_t weight_num = weight.numel();  // 权重要素数量
  Tensor result = at::empty_like(input); // 创建输出张量
  auto strides = input.strides();        // 获取输入步长

  // 情况1: 所有通道共享同一个权重
  if (weight_num == 1) {
    // 分发浮点类型处理
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "prelu_cuda", [&] {
      prelu_cuda_kernel_share_weights<scalar_t>(
        input,
        result,
        weight.data<scalar_t>());  // 传入权重指针
    });
  }
  else { // 情况2: 每个通道有独立权重
    int64_t input_ndim = input.dim();
    AT_CHECK(input_ndim > 0, "不允许零维输入张量");

    int64_t channel_size = 1;     // 默认通道数为1
    int64_t input_stride0 = 1, input_stride1 = 1;

    // 对于维度大于1的张量（NCHW格式）
    if (input_ndim > 1) {
      channel_size = input.size(1); // 通道数是第2维度
      input_stride0 = strides[0];   // 批次维度步长
      input_stride1 = strides[1];   // 通道维度步长
    }
    
    // 检查通道数和权重要素数匹配
    AT_CHECK(channel_size == weight_num,
      "参数数量与输入通道大小不匹配。参数数量 = %d, 通道大小 = %d.",
      weight_num, channel_size);

    // 配置CUDA核函数参数
    int64_t input_numel = input.numel();
    const dim3 block = dim3(std::min(static_cast<int64_t>(cuda::getApplyBlock().x), input_numel));
    dim3 grid;
    int curDevice = -1;
    cudaGetDevice(&curDevice);  // 获取当前设备ID
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice); // 获取当前CUDA流
    
    // 计算网格维度
    AT_CHECK(cuda::getApplyGrid(input_numel, grid, curDevice), 
             "prelu: 输入太大或维度过多");

    // 分发浮点类型处理
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "prelu_cuda", [&] {
      // 启动多权重核函数
      prelu_cuda_kernel_multi_weights<scalar_t>
      <<<grid, block, 0, stream>>>(
        result.data<scalar_t>(),
        input.data<scalar_t>(),
        weight.data<scalar_t>(),
        input_stride0,
        input_stride1,
        input_numel);
    });
  }
  return result;
}

// ==============================================================
// PReLU 反向传播
// ==============================================================

// 共享权重的PReLU反向传播
template <typename scalar_t>
void prelu_cuda_backward_kernel_share_weights(
  const Tensor& input,
  const Tensor& grad_out,
  Tensor& input_grad,
  Tensor& weight_grad_collector,
  const scalar_t* weight_data) {

  // 四元组张量应用：输入、输出梯度、输入梯度、权重梯度收集器
  at::cuda::CUDA_tensor_apply4<scalar_t, scalar_t, scalar_t, scalar_t>(
    input,
    grad_out,
    input_grad,
    weight_grad_collector,
    [=] __device__ (
      const scalar_t& input_val,
      const scalar_t& grad_out_val,
      scalar_t& input_grad_val,
      scalar_t& weight_grad_collector_val) {
        // 输入梯度公式: dL/dx = dL/do * (x>0 ? 1 : w)
        input_grad_val = (input_val > 0) ? grad_out_val : *weight_data * grad_out_val;
        // 权重梯度临时值: dL/dw = dL/do * (x>0 ? 0 : x)
        weight_grad_collector_val = (input_val > 0) ? scalar_t(0) : input_val * grad_out_val;
  });
}

// 多权重的PReLU反向传播
template <typename scalar_t>
__global__ void prelu_cuda_backward_kernel_multi_weights(
  const scalar_t* input_data,
  const scalar_t* weight_data,
  const scalar_t* grad_out_data,
  scalar_t* input_grad_data,
  scalar_t* weight_grad_collector,
  int64_t input_stride0,
  int64_t input_stride1,
  int64_t input_numel) {

  int64_t linearId = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearId >= input_numel) return;
  
  // 计算通道索引
  int64_t channel = (linearId % input_stride0) / input_stride1;
  scalar_t input_data_val = input_data[linearId];
  scalar_t grad_out_data_val = grad_out_data[linearId];
  
  // 计算输入梯度
  input_grad_data[linearId] = (input_data_val > 0) 
    ? grad_out_data_val 
    : weight_data[channel] * grad_out_data_val;
  
  // 收集权重梯度分量
  weight_grad_collector[linearId] = (input_data_val > 0) 
    ? scalar_t(0) 
    : input_data_val * grad_out_data_val;
}

// PReLU反向传播主函数
std::tuple<Tensor, Tensor> prelu_backward_cuda(const Tensor& grad_out_, const Tensor& self, const Tensor& weight_) {
  // 输入检查
  AT_CHECK(grad_out_.is_cuda(), "梯度输出必须在CUDA设备上");
  AT_CHECK(self.is_cuda(), "输入必须在CUDA设备上");
  AT_CHECK(weight_.is_cuda(), "权重必须在CUDA设备上");

  // 确保内存连续
  auto input = self.contiguous();
  auto grad_out = grad_out_.contiguous();
  auto weight = weight_.contiguous();

  AT_CHECK(input.is_contiguous(), "输入必须是连续的");
  AT_CHECK(weight.is_contiguous(), "权重必须是连续的");
  AT_CHECK(grad_out.is_contiguous(), "梯度输出必须是连续的");

  int64_t weight_num = weight.numel();
  auto strides = input.strides();
  auto dims = input.dim();
  
  // 创建梯度张量
  Tensor input_grad = at::empty_like(input);          // 输入梯度
  Tensor weight_grad = at::empty_like(weight);        // 权重梯度
  Tensor weight_grad_collector = at::empty_like(input); // 权重梯度收集器

  // 情况1: 共享权重
  if (weight_num == 1) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "prelu_backward_cuda", [&] {
      prelu_cuda_backward_kernel_share_weights<scalar_t>(
        input,
        grad_out,
        input_grad,
        weight_grad_collector,
        weight.data<scalar_t>());
    });
    // 对收集的梯度求和得到最终权重梯度
    weight_grad.fill_(weight_grad_collector.sum());
  }
  else { // 情况2: 多权重
    int64_t input_ndim = input.dim();
    AT_CHECK(input_ndim > 0, "不允许零维输入张量");

    int64_t channel_size = 1;
    int64_t input_stride0 = 1, input_stride1 = 1;

    if (input_ndim > 1) {
      channel_size = input.size(1);
      input_stride0 = strides[0];
      input_stride1 = strides[1];
    }
    
    // 检查通道数匹配
    AT_CHECK(channel_size == weight_num,
      "参数数量与输入通道大小不匹配。参数数量 = %d, 通道大小 = %d.",
      weight_num, channel_size);

    // 配置CUDA核函数
    int64_t input_numel = input.numel();
    const dim3 block = dim3(std::min(static_cast<int64_t>(cuda::getApplyBlock().x), input_numel));
    dim3 grid;
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
    AT_CHECK(cuda::getApplyGrid(input_numel, grid, curDevice), 
             "prelu_backward_cuda: 输入太大或维度过多");

    // 分发处理
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "prelu_backward_cuda", [&] {
      // 启动多权重反向传播核函数
      prelu_cuda_backward_kernel_multi_weights<scalar_t>
      <<<grid, block, 0, stream>>>(
        input.data<scalar_t>(),
        weight.data<scalar_t>(),
        grad_out.data<scalar_t>(),
        input_grad.data<scalar_t>(),
        weight_grad_collector.data<scalar_t>(),
        input_stride0,
        input_stride1,
        input_numel);
    });
    
    // 计算权重梯度：沿批次和空间维度求和
    std::vector<int64_t> reduce_dims;
    reduce_dims.push_back(0);  // 批次维度
    if (dims > 2) {
      // 添加空间维度 (H, W, ...)
      for(int64_t i = 2; i < dims; i++) reduce_dims.push_back(i);
    }
    weight_grad = weight_grad_collector.sum(reduce_dims);
  }
  return std::tuple<Tensor, Tensor>{input_grad, weight_grad};
}

// ==============================================================
// Hardshrink 硬收缩函数
// ==============================================================

// Hardshrink前向传播
template <typename scalar_t>
void hardshrink_cuda_kernel(const Tensor& self, Tensor& out_tensor, scalar_t lambd) {
  // 二元张量应用
  at::cuda::CUDA_tensor_apply2<scalar_t, scalar_t>(
    self,
    out_tensor,
    [=] __device__ (scalar_t& self_val, scalar_t& out_tensor_val) {
        // Hardshrink公式: |x| <= λ ? 0 : x
        out_tensor_val = (self_val >= -lambd && self_val <= lambd) 
                         ? scalar_t(0) 
                         : self_val;
  });
}

// Hardshrink反向传播
template <typename scalar_t>
void hardshrink_backward_cuda_kernel(
  const Tensor& self, 
  Tensor& out_tensor, 
  scalar_t lambd, 
  const Tensor& grad) {
  
  // 三元张量应用
  at::cuda::CUDA_tensor_apply3<scalar_t, scalar_t, scalar_t>(
    self,
    grad,
    out_tensor,
    [=] __device__ (
      scalar_t& self_val,
      scalar_t& grad_val,
      scalar_t& out_tensor_val) {
        // 梯度公式: dL/dx = |x| <= λ ? 0 : dL/do
        out_tensor_val = (self_val >= -lambd && self_val <= lambd) 
                         ? scalar_t(0) 
                         : grad_val;
  });
}

// Hardshrink主函数
Tensor hardshrink_cuda(const Tensor & self, Scalar lambd) {
  auto out_tensor = at::empty_like(self); // 创建输出张量
  // 分发浮点类型处理
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "hardshrink_cuda", [&] {
    hardshrink_cuda_kernel<scalar_t>(self, out_tensor, lambd.to<scalar_t>());
  });
  return out_tensor;
}

// Hardshrink反向传播主函数
Tensor hardshrink_backward_cuda(const Tensor & grad, const Tensor & self, Scalar lambd) {
  auto out_tensor = at::empty_like(grad); // 创建输出梯度张量
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "hardshrink_backward_cuda", [&] {
    hardshrink_backward_cuda_kernel<scalar_t>(self, out_tensor, lambd.to<scalar_t>(), grad);
  });
  return out_tensor;
}

// ==============================================================
// Threshold 阈值函数
// ==============================================================

// 阈值函数核实现
template <typename scalar_t>
void threshold_kernel_impl(TensorIterator& iter, scalar_t threshold, scalar_t value) {
  // 二元GPU核函数
  gpu_binary_kernel(iter, [=]GPU_LAMBDA(scalar_t x, scalar_t other) -> scalar_t {
    // 阈值公式: x <= threshold ? value : other
    return x <= threshold ? value : other;
  });
}

// 阈值函数分发入口
static void threshold_kernel(TensorIterator& iter, Scalar threshold, Scalar value) {
  // 支持所有浮点类型和半精度
  AT_DISPATCH_ALL_TYPES_AND_HALF(iter.type(), "threshold", [&] {
    threshold_kernel_impl<scalar_t>(iter, threshold.to<scalar_t>(), value.to<scalar_t>());
  });
}

// 注册阈值函数分发器
REGISTER_DISPATCH(threshold_stub, &threshold_kernel);

}}  // namespace at::native