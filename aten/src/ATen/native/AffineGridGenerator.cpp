#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

namespace at { namespace native {

// 生成从-1到1的等间距序列，用于构建基础网格坐标
at::Tensor linspace_from_neg_one(const Tensor& grid, int64_t num_steps) {
  if (num_steps > 1) {
    // 当步数大于1时，生成[-1, 1]区间的等分点
    return at::linspace(-1, 1, num_steps, grid.options());
  } else {
    // 单点时直接返回-1
    return at::tensor(-1, grid.options());
  }
}

// 创建4D仿射变换的基础网格 (用于2D空间变换)
Tensor make_base_grid_4D(
    const Tensor& theta,    // 仿射变换矩阵 [N, 2, 3]
    int64_t N,              // batch大小
    int64_t C,              // 通道数 (未使用)
    int64_t H,              // 高度
    int64_t W) {            // 宽度
  
  // 创建基础网格张量 [N, H, W, 3]
  // 最后一个维度3表示齐次坐标 (x, y, 1)
  auto base_grid = at::empty({N, H, W, 3}, theta.options());

  // 设置x坐标：在宽度方向生成[-1,1]的线性序列
  base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W));
  
  // 设置y坐标：在高度方向生成[-1,1]的线性序列
  // unsqueeze(-1) 添加维度使其可广播
  base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H).unsqueeze_(-1));
  
  // 设置齐次坐标分量：全1
  base_grid.select(-1, 2).fill_(1);

  return base_grid;
}

// 创建5D仿射变换的基础网格 (用于3D体积变换)
Tensor make_base_grid_5D(
    const Tensor& theta,    // 仿射变换矩阵 [N, 3, 4]
    int64_t N,              // batch大小
    int64_t C,              // 通道数 (未使用)
    int64_t D,              // 深度
    int64_t H,              // 高度
    int64_t W) {            // 宽度
  
  // 创建基础网格张量 [N, D, H, W, 4]
  // 最后一个维度4表示齐次坐标 (x, y, z, 1)
  auto base_grid = at::empty({N, D, H, W, 4}, theta.options());

  // 设置x坐标：宽度方向[-1,1]线性序列
  base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W));
  
  // 设置y坐标：高度方向[-1,1]线性序列
  base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H).unsqueeze_(-1));
  
  // 设置z坐标：深度方向[-1,1]线性序列
  base_grid.select(-1, 2).copy_(linspace_from_neg_one(theta, D).unsqueeze_(-1).unsqueeze_(-1));
  
  // 设置齐次坐标分量：全1
  base_grid.select(-1, 3).fill_(1);

  return base_grid;
}

// 4D仿射网格生成器 (2D空间变换)
Tensor affine_grid_generator_4D(
    const Tensor& theta,    // 仿射变换矩阵 [N, 2, 3]
    int64_t N,              // batch大小
    int64_t C,              // 通道数
    int64_t H,              // 输出高度
    int64_t W) {            // 输出宽度
  
  // 1. 创建基础网格 [N, H, W, 3]
  Tensor base_grid = make_base_grid_4D(theta, N, C, H, W);
  
  // 2. 应用仿射变换：
  //    - 将网格重塑为 [N, H*W, 3]
  //    - 与转置后的theta矩阵 [N, 3, 2] 进行批矩阵乘法
  //    - 结果形状为 [N, H*W, 2]
  auto grid = base_grid.view({N, H * W, 3}).bmm(theta.transpose(1, 2));
  
  // 3. 将结果重塑为最终网格格式 [N, H, W, 2]
  return grid.view({N, H, W, 2});
}

// 5D仿射网格生成器 (3D体积变换)
Tensor affine_grid_generator_5D(
    const Tensor& theta,    // 仿射变换矩阵 [N, 3, 4]
    int64_t N,              // batch大小
    int64_t C,              // 通道数
    int64_t D,              // 输出深度
    int64_t H,              // 输出高度
    int64_t W) {            // 输出宽度
  
  // 1. 创建基础网格 [N, D, H, W, 4]
  Tensor base_grid = make_base_grid_5D(theta, N, C, D, H, W);
  
  // 2. 应用仿射变换：
  //    - 将网格重塑为 [N, D*H*W, 4]
  //    - 与转置后的theta矩阵 [N, 4, 3] 进行批矩阵乘法
  //    - 结果形状为 [N, D*H*W, 3]
  auto grid = base_grid.view({N, D * H * W, 4}).bmm(theta.transpose(1, 2));
  
  // 3. 将结果重塑为最终网格格式 [N, D, H, W, 3]
  return grid.view({N, D, H, W, 3});
}

// 仿射网格生成主入口函数
Tensor affine_grid_generator(const Tensor& theta, IntList size) {
  // 输入验证：只支持4D或5D
  AT_CHECK(
      size.size() == 4 || size.size() == 5,
      "AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.");
  
  if (size.size() == 4) {
    // 4D情况：2D空间变换 [N, C, H, W]
    return affine_grid_generator_4D(theta, size[0], size[1], size[2], size[3]);
  } else {
    // 5D情况：3D体积变换 [N, C, D, H, W]
    return affine_grid_generator_5D(
        theta, size[0], size[1], size[2], size[3], size[4]);
  }
}

// 4D仿射网格生成的反向传播
Tensor affine_grid_generator_4D_backward(
    const Tensor& grad_grid,  // 上游梯度 [N, H, W, 2]
    int64_t N,                // batch大小
    int64_t C,                // 通道数
    int64_t H,                // 高度
    int64_t W) {              // 宽度
  
  // 1. 创建基础网格 [N, H, W, 3]
  auto base_grid = make_base_grid_4D(grad_grid, N, C, H, W);
  
  // 验证梯度形状
  AT_ASSERT(grad_grid.sizes() == IntList({N, H, W, 2}));
  
  // 2. 计算theta梯度：
  //    a. 将基础网格重塑为 [N, H*W, 3] 并转置为 [N, 3, H*W]
  //    b. 将梯度重塑为 [N, H*W, 2]
  //    c. 执行批矩阵乘法: [N, 3, H*W] × [N, H*W, 2] = [N, 3, 2]
  auto grad_theta = base_grid.view({N, H * W, 3})
                        .transpose(1, 2)
                        .bmm(grad_grid.view({N, H * W, 2}));
  
  // 3. 转置回原始theta的形状 [N, 2, 3]
  return grad_theta.transpose(1, 2);
}

// 5D仿射网格生成的反向传播
Tensor affine_grid_generator_5D_backward(
    const Tensor& grad_grid,  // 上游梯度 [N, D, H, W, 3]
    int64_t N,                // batch大小
    int64_t C,                // 通道数
    int64_t D,                // 深度
    int64_t H,                // 高度
    int64_t W) {              // 宽度
  
  // 1. 创建基础网格 [N, D, H, W, 4]
  auto base_grid = make_base_grid_5D(grad_grid, N, C, D, H, W);
  
  // 验证梯度形状
  AT_ASSERT(grad_grid.sizes() == IntList({N, D, H, W, 3}));
  
  // 2. 计算theta梯度：
  //    a. 将基础网格重塑为 [N, D*H*W, 4] 并转置为 [N, 4, D*H*W]
  //    b. 将梯度重塑为 [N, D*H*W, 3]
  //    c. 执行批矩阵乘法: [N, 4, D*H*W] × [N, D*H*W, 3] = [N, 4, 3]
  auto grad_theta = base_grid.view({N, D * H * W, 4})
                        .transpose(1, 2)
                        .bmm(grad_grid.view({N, D * H * W, 3}));
  
  // 3. 转置回原始theta的形状 [N, 3, 4]
  return grad_theta.transpose(1, 2);
}

// 仿射网格生成反向传播主入口
Tensor affine_grid_generator_backward(const Tensor& grad, IntList size) {
  // 输入验证：只支持4D或5D
  AT_CHECK(
      size.size() == 4 || size.size() == 5,
      "AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.");
  
  if (size.size() == 4) {
    // 处理4D反向传播
    return affine_grid_generator_4D_backward(
        grad, size[0], size[1], size[2], size[3]);
  } else {
    // 处理5D反向传播
    return affine_grid_generator_5D_backward(
        grad, size[0], size[1], size[2], size[3], size[4]);
  }
}

}}  // namespace at::native