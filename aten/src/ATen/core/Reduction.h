#pragma once

namespace Reduction {

/**
 * @namespace Reduction
 * @brief 损失函数缩减策略的命名空间
 * @note 这些枚举值需要与Python端的torch/nn/modules/functional.py中的Reduction类保持同步
 *       理想情况下应该使用作用域枚举(scoped enum)，但JIT目前不支持该特性
 */

/**
 * @enum Reduction
 * @brief 定义损失函数的缩减方式
 * 
 * 该枚举控制损失函数的缩减行为，主要用于指定如何对多个损失值进行聚合
 */
enum Reduction {
  None,             ///< 不进行缩减，保留所有损失值的原始维度
  Mean,             ///< 计算损失的(可能加权)平均值
  Sum,              ///< 计算损失的总和
  END               ///< 标记枚举结束，用于内部检查
};

} // namespace Reduction