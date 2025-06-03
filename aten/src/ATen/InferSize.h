#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <sstream>
#include <vector>

namespace at {

/**
 * 推断包含-1的维度大小，并检查新形状是否与元素总数兼容
 * @param shape 目标形状，可能包含-1表示待推断维度
 * @param numel 当前张量的·总元素数
 * @return 推断后的完整形状
 * @throws 当形状不兼容或存在多个-1时抛出异常
 */
static std::vector<int64_t> infer_size(IntList shape, int64_t numel) {
  auto res = shape.vec();  // 创建目标形状的副本用于修改
  
  int64_t newsize = 1;     // 计算非推断维度的总元素数
  auto infer_dim = c10::optional<int64_t>();  // 记录需要推断的维度位置

  // 第一遍遍历：检查形状有效性并定位待推断维度
  for (int64_t dim = 0, ndim = shape.size(); dim != ndim; dim++) {
    if (shape[dim] == -1) {  // 找到待推断维度
      if (infer_dim) {  // 如果已经存在待推断维度
        throw std::runtime_error("only one dimension can be inferred");
      }
      infer_dim = dim;  // 记录第一个-1出现的位置
    } else if (shape[dim] >= 0) {  // 正常维度
      newsize *= shape[dim];  // 累乘已知维度大小
    } else {  // 非法维度值
      AT_ERROR("invalid shape dimension ", shape[dim]);
    }
  }

  // 检查元素总数是否兼容
  if (numel == newsize ||  // 完全匹配情况
      (infer_dim && newsize > 0 && numel % newsize == 0)) {  // 可整除情况
    if (infer_dim) {  // 需要推断维度的情况
      // 遵循NumPy语义：0元素张量不能reshape到不确定形状
      AT_CHECK(newsize != 0, 
              "cannot reshape tensor of 0 elements into shape ", shape);
      // 计算推断维度的大小 = 总元素数 / 其他维度乘积
      res[*infer_dim] = numel / newsize;
    }
    return res;  // 返回推断后的完整形状
  }

  // 不兼容情况下的错误处理
  std::ostringstream ss;
  ss << "shape '" << shape << "' is invalid for input of size " << numel;
  throw std::runtime_error(ss.str());  // 抛出详细错误信息
}

} // namespace at