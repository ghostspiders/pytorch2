#include "ATen/ATen.h"

namespace at { namespace native {

// 实现常数填充功能
Tensor constant_pad_nd(const Tensor& self, IntList pad, Scalar value) {
    // 检查填充列表长度必须为偶数（每个维度需要两个填充值：起始和结束）
    AT_CHECK(pad.size() % 2 == 0, "Length of pad must be even but instead it equals ",
             pad.size());

    auto input_sizes = self.sizes();  // 获取输入张量的维度大小
    auto l_inp = self.dim();          // 获取输入张量的维度数

    auto l_pad = pad.size() / 2;      // 需要填充的维度数（每个维度对应两个填充值）
    auto l_diff = l_inp - l_pad;      // 不需要填充的维度数（前导维度）
    // 检查填充维度不能超过输入维度
    AT_CHECK(l_inp >= l_pad, "Length of pad should be no more than twice the number of "
             "dimensions of the input. Pad length is ", pad.size(), "while the input has ",
             l_inp, "dimensions.");

    std::vector<int64_t> new_shape;  // 用于存储输出张量的新形状

    bool all_pads_non_positive = true;  // 标记所有填充值是否都非正数（负值表示裁剪）

    // 处理负填充（裁剪操作）
    auto c_input = self;  // 复制输入张量
    // 从最后一个维度开始处理（高维优先）
    for (int i = l_diff; i < l_inp; i++) {
        auto pad_idx = 2 * (l_inp - i - 1);  // 计算当前维度在pad列表中的起始索引
        
        // 处理起始位置的负填充（裁剪左侧）
        if (pad[pad_idx] < 0) {
            c_input = c_input.narrow(i, -pad[pad_idx], c_input.size(i) + pad[pad_idx]);
        } else if (pad[pad_idx] != 0) {
            all_pads_non_positive = false;  // 存在正填充
        }
        
        // 处理结束位置的负填充（裁剪右侧）
        if (pad[pad_idx + 1] < 0) {
            c_input = c_input.narrow(i, 0, c_input.size(i) + pad[pad_idx + 1]);
        } else if (pad[pad_idx + 1] != 0) {
            all_pads_non_positive = false;  // 存在正填充
        }
    }

    // 优化：如果所有填充都是非正数（只有裁剪），直接返回裁剪后的张量
    if (all_pads_non_positive) {
        return c_input;
    }

    // 构建输出形状
    // 1. 添加不需要填充的前导维度
    for (int i = 0; i < l_diff; i ++) {
        new_shape.emplace_back(input_sizes[i]);
    }

    // 2. 处理需要填充的维度
    for (int i = 0; i < l_pad; i++) {
        // 从pad列表末尾开始取每对填充值（从低维到高维）
        auto pad_idx = pad.size() - ((i + 1) * 2);
        // 计算新维度大小 = 原尺寸 + 左侧填充 + 右侧填充
        auto new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1];
        // 检查新维度是否有效（必须>0）
        AT_CHECK(new_dim > 0, "The input size ", input_sizes[l_diff + i], ", plus negative padding ",
                 pad[pad_idx], " and ", pad[pad_idx + 1], "resulted in a negative output size, "
                 "which is invalid. Check dimension ", l_diff + i, "of your input.");
        new_shape.emplace_back(new_dim);
    }

    // 创建填充了指定值的输出张量
    auto output = at::empty(new_shape, self.options());
    output.fill_(value);

    // 定位输出张量中需要复制原始数据的区域
    auto c_output = output;
    // 从高维到低维处理
    for (int i = l_diff; i < l_inp; i++) {
        auto pad_idx = 2 * (l_inp - i - 1);  // 当前维度的填充索引
        
        // 裁剪左侧填充区域（如果存在正填充）
        if (pad[pad_idx] > 0) {
            c_output = c_output.narrow(i, pad[pad_idx], c_output.size(i) - pad[pad_idx]);
        }
        // 裁剪右侧填充区域（如果存在正填充）
        if (pad[pad_idx + 1] > 0) {
            c_output = c_output.narrow(i, 0, c_output.size(i) - pad[pad_idx + 1]);
        }
    }
    
    // 将处理后的输入数据复制到输出张量的中心区域
    c_output.copy_(c_input);
    return output;
}

}}  // namespace at::native