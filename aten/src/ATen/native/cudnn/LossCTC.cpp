#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>

#if AT_CUDNN_ENABLED()
  #include <ATen/cudnn/Descriptors.h>
#endif

#if !AT_CUDNN_ENABLED() || (CUDNN_VERSION < 7000)

namespace at { namespace native {

// 如果没有启用 cuDNN 或者 cuDNN 版本低于 7.0，则直接报错

// cuDNN CTC 损失函数
std::tuple<Tensor, Tensor> _cudnn_ctc_loss(const Tensor& log_probs, const Tensor& targets, IntList input_lengths, IntList target_lengths, int64_t BLANK, bool deterministic) {
  AT_ERROR("cudnn_ctc_loss: ATen not compiled with cuDNN >= 7 support");
}

}}

#else // AT_CUDNN_ENABLED

#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>

#include <ATen/TensorUtils.h>

namespace at { namespace native {

namespace {

}  // namespace

// cuDNN CTC 损失函数
std::tuple<Tensor, Tensor> _cudnn_ctc_loss(const Tensor& log_probs_t, const Tensor& targets_t, IntList input_lengths_, IntList target_lengths_, int64_t BLANK, bool deterministic) {
  CheckedFrom c = "cudnn_ctc_loss";
  TensorArg log_probs { log_probs_t, "log_probs", 1 };
  TensorArg targets { targets_t, "targets", 2 };
  // 检查 log_probs 的维度是否为 3 (T, N, C)
  checkDim(c, log_probs, 3);
  // 检查 log_probs 的数据类型是否为 kFloat
  checkScalarType(c, log_probs, kFloat);
  // 检查 targets 的维度是否为 1
  checkDim(c, targets, 1);
  // 检查 targets 的数据类型是否为 kInt
  checkScalarType(c, targets, kInt);
  // 检查 targets 是否连续
  checkContiguous(c, targets); 
  // 检查 log_probs 和 targets 是否在同一 GPU 上
  checkBackend(c, {*log_probs}, Backend::CUDA);
  checkBackend(c, {*targets}, Backend::CPU);
  // 获取批量大小
  int64_t batch_size = log_probs->size(1);
  // 检查 input_lengths 的大小是否与批量大小匹配
  AT_CHECK(input_lengths_.size() == batch_size, "input_lengths needs to have size to match batch_size");
  // 检查 target_lengths 的大小是否与批量大小匹配
  AT_CHECK(target_lengths_.size() == batch_size, "target_lengths needs to have size to match batch_size");

  // 将输入长度和目标长度转换为 std::vector<int>
  std::vector<int> input_lengths(input_lengths_.begin(), input_lengths_.end());
  std::vector<int> target_lengths(target_lengths_.begin(), target_lengths_.end());

  // 设置当前 CUDA 流
  setCuDNNStreamToCurrent();
  // 检查空白标签是否为 0
  AT_CHECK(BLANK == 0, "blank must be label 0 for cudnn_ctc_loss");
  // 检查其他条件（在 dispatch 中检查）：
  // 所有标签长度 <= 256
  // 所有输入长度 = logprob.size(0)

  // 获取 cuDNN 句柄
  auto handle = getCudnnHandle();

  // 选择 CTC 损失算法（确定性或非确定性）
  cudnnCTCLossAlgo_t algo = (deterministic ? CUDNN_CTC_LOSS_ALGO_DETERMINISTIC : CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC);

  // 计算 softmax
  Tensor probs = log_probs->softmax(2);
  // 创建概率张量描述符
  TensorDescriptor probs_desc{probs};
  // 创建梯度张量
  Tensor grad = at::empty_like(probs);
  // 创建梯度张量描述符
  TensorDescriptor grad_desc{grad};

  // 创建 CTC 损失描述符
  CTCLossDescriptor ctc_loss_desc;
  ctc_loss_desc.set(CUDNN_DATA_FLOAT);

  // 获取工作空间大小
  size_t workspace_size;
  AT_CUDNN_CHECK(cudnnGetCTCLossWorkspaceSize(handle, probs_desc.desc(), grad_desc.desc(),
					      targets->data<int>(), target_lengths.data(), input_lengths.data(),
					      algo, ctc_loss_desc.desc(), &workspace_size));

  // 创建工作空间
  Tensor workspace = at::empty(workspace_size, log_probs->options().dtype(kByte));
  // 创建损失张量
  Tensor costs = at::empty({log_probs->size(1)}, log_probs->options());

  // 调用 cuDNN CTC 损失函数
  AT_CUDNN_CHECK(cudnnCTCLoss(handle, probs_desc.desc(), probs.data_ptr(),
                              targets->data<int>(), target_lengths.data(), input_lengths.data(),
                              costs.data_ptr(), grad_desc.desc(), grad.data_ptr(), algo,
                              ctc_loss_desc.desc(), workspace.data_ptr(), workspace_size));

  // 返回损失和梯度
  return std::make_tuple(costs, grad);
}

}}  // namespace at::native

#endif