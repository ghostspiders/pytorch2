#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include <tuple>

namespace at {
namespace native {

// TBC格式卷积（Time-Batch-Channel格式）
// 输入张量维度: [时间步, 批次大小, 输入通道]
// 权重张量维度: [卷积核宽度, 输入通道, 输出通道]
// 偏置张量维度: [输出通道]
Tensor conv_tbc(const Tensor& self, const Tensor& weight, const Tensor& bias, int64_t pad) {
  // 检查输入维度必须为3维
  AT_CHECK(self.dim() == 3, "Input must have 3 dims: time, batch, in_channel");
  // 检查权重维度必须为3维
  AT_CHECK(weight.dim() == 3, "Weight tensor must have 3 dims: kernel_width, in_channels, out_channels.");
  // 检查偏置维度必须为1维
  AT_CHECK(bias.dim() == 1, "Bias must be 1-D");

  auto input_size = self.sizes();    // 获取输入张量尺寸
  auto weight_size = weight.sizes(); // 获取权重张量尺寸

  auto ilen = input_size[0];         // 输入时间步数
  auto batchSize = input_size[1];    // 批次大小
  auto inputPlanes = input_size[2];  // 输入通道数
  auto outputPlanes = weight_size[2];// 输出通道数
  auto kw = weight_size[0];          // 卷积核宽度
  auto olen = input_size[0] - kw + 1 + pad * 2; // 计算输出时间步数
  auto real_pad = (olen - ilen + kw - 1) / 2;   // 计算实际填充量

  // 检查输入通道数必须匹配权重的输入通道维度
  AT_CHECK(inputPlanes == weight_size[1], "Input dim 2 (input channels) is not == dim 1 in the weight tensor");
  // 检查权重输出通道数必须匹配偏置的通道数
  AT_CHECK(weight_size[2] == bias.sizes()[0], "Bias size must equal dim 2 in the weight tensor (output channels).");

  // 创建输出张量 [输出时间步, 批次大小, 输出通道]
  Tensor output = at::empty({
    olen,
    input_size[1],
    weight_size[2],
  }, self.options());
  
  // 使用偏置初始化输出张量（扩展偏置到输出形状）
  output.copy_(bias.expand(output.sizes()));
  
  // 遍历卷积核的每个位置（滑动窗口）
  for (int k = 0; k < kw; k++) {
    // 计算输入偏移（考虑填充）
    int iShift = std::max(0, static_cast<int>(k - real_pad));
    // 计算输出偏移（考虑填充）
    int oShift = std::max(0, static_cast<int>(real_pad - k));
    // 计算当前卷积核位置的有效时间步数
    int t = std::min(ilen + real_pad - k, olen) - oShift;
    
    // 如果存在有效时间步
    if (t > 0) {
      auto W = weight[k]; // 当前卷积核位置的权重切片 [输入通道, 输出通道]
      
      // 获取输入切片并重塑为 [t*batchSize, inputPlanes] 的矩阵
      auto I = self.narrow(0, iShift, t).view({t * batchSize, inputPlanes});
      
      // 获取输出切片并重塑为 [t*batchSize, outputPlanes] 的矩阵
      auto O = output.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      
      // 矩阵乘法累加: O = I * W + O
      O.addmm_(I, W);
    }
  }
  return output;
}

// TBC卷积的反向传播函数
std::tuple<Tensor, Tensor, Tensor> conv_tbc_backward(
    const Tensor& dOutput,   // 输出梯度
    const Tensor& input,     // 原始输入
    const Tensor& weight,    // 卷积权重
    const Tensor& bias,      // 偏置
    int64_t pad) {           // 填充值
  auto input_size = input.sizes();    // 输入尺寸
  auto weight_size = weight.sizes();  // 权重尺寸

  auto ilen = input_size[0];         // 输入时间步数
  auto batchSize = input_size[1];    // 批次大小
  auto inputPlanes = input_size[2];  // 输入通道数
  auto outputPlanes = weight_size[2];// 输出通道数
  auto kw = weight.sizes()[0];       // 卷积核宽度
  auto olen = input_size[0] - kw + 1 + pad * 2; // 输出时间步数
  int real_pad = (olen - ilen + kw - 1) / 2;    // 实际填充量

  // 初始化输入梯度张量（与输入同形）
  Tensor dInput = at::zeros_like(input);
  
  // 计算输入梯度: dInput = dOutput * weight^T
  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - real_pad);  // 输入偏移
    int oShift = std::max(0, real_pad - k);  // 输出偏移
    int t = std::min(ilen + real_pad - k, olen) - oShift; // 有效时间步
    
    if (t > 0) {
      // 获取输出梯度切片并重塑为矩阵 [t*batchSize, outputPlanes]
      auto dO = dOutput.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      
      // 获取输入梯度切片并重塑为矩阵 [t*batchSize, inputPlanes]
      auto dI = dInput.narrow(0, iShift, t).view({t * batchSize, inputPlanes});
      
      // 矩阵乘法: dI += dO * W[k]^T
      dI.addmm_(dO, weight[k].t());
    }
  }

  // 初始化权重梯度张量（与权重同形）
  Tensor dWeight = at::zeros_like(weight);
  
  // 计算权重梯度: dWeight = input^T * dOutput
  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - real_pad);  // 输入偏移
    int oShift = std::max(0, real_pad - k);  // 输出偏移
    int t = std::min(ilen + real_pad - k, olen) - oShift; // 有效时间步
    
    if (t > 0) {
      auto dW = dWeight[k]; // 当前卷积核位置的梯度切片
      
      // 获取输出梯度切片并重塑为矩阵 [t*batchSize, outputPlanes]
      auto dO = dOutput.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      
      // 获取输入切片，转置后为 [inputPlanes, t*batchSize]
      auto I = input.narrow(0, iShift, t).view({t * batchSize, inputPlanes}).t();
      
      // 矩阵乘法: dW += I * dO
      dW.addmm_(I, dO);
    }
  }

  // 计算偏置梯度: 对输出梯度在时间和批次维度求和
  Tensor dBias = at::zeros_like(bias);
  auto tmp = dOutput.sum(0, false); // 沿时间维度求和 [batch, outputPlanes]
  dBias.copy_(tmp.sum(0));          // 沿批次维度求和 [outputPlanes]

  // 返回三个梯度张量: dInput, dWeight, dBias
  return std::make_tuple(dInput, dWeight, dBias);
}

}
} // namespace at::native