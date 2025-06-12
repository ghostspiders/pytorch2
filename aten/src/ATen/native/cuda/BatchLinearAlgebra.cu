#include "ATen/Context.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/cuda/PinnedMemoryAllocator.h"  // 固定内存分配器
#include "ATen/cuda/CUDAApplyUtils.cuh"        // CUDA应用工具

#include "ATen/native/LinearAlgebraUtils.h"    // 线性代数工具
#include "ATen/native/cuda/MiscUtils.h"        // CUDA杂项工具

#include "THC.h" // 用于USE_MAGMA宏定义

// 如果定义了USE_MAGMA，则包含MAGMA头文件
#ifdef USE_MAGMA
#include <magma.h>
#include <magma_types.h>
#endif

namespace at {
namespace native {

#ifdef USE_MAGMA
// MAGMA函数模板声明（float/double特化前的基础模板）
// 这些模板会在特化时实现具体的MAGMA函数调用

// 批量求解线性方程组 (AX = B)
template<class scalar_t>
void magmaGesvBatched(
    magma_int_t n, magma_int_t nrhs, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, scalar_t** dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array, magma_int_t batch_count, const MAGMAQueue& magma_queue) {
  AT_ERROR("gesv仅支持float或double类型张量");
}

// 批量LU分解
template<class scalar_t>
void magmaGetrfBatched(
    magma_int_t m, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  AT_ERROR("getrf仅支持float或double类型张量");
}

// 批量矩阵求逆（基于LU分解结果）
template<class scalar_t>
void magmaGetriBatched(
    magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, scalar_t** dinvA_array, magma_int_t lddia,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  AT_ERROR("getri仅支持float或double类型张量");
}

// 批量正定矩阵求解（基于Cholesky分解）
template<class scalar_t>
void magmaPotrsBatched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, scalar_t** dA_array, magma_int_t ldda,
    scalar_t** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  AT_ERROR("potrs仅支持float或double类型张量");
}

// 批量Cholesky分解
template<class scalar_t>
void magmaCholeskyBatched(
    magma_uplo_t uplo, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  AT_ERROR("cholesky仅支持float或double类型张量");
}

// =============== double类型的MAGMA函数特化实现 ===============

// gesv: 求解线性方程组 (double)
template<>
void magmaGesvBatched<double>(
    magma_int_t n, magma_int_t nrhs, double** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, double** dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array, magma_int_t batch_count, const MAGMAQueue& magma_queue) {
  magma_dgesv_batched(n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, 
                      dinfo_array, batch_count, magma_queue.get_queue());
}

// getrf: LU分解 (double)
template<>
void magmaGetrfBatched<double>(
    magma_int_t m, magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
    magma_dgetrf_batched(m, n, dA_array, ldda, ipiv_array, info_array, 
                         batchsize, magma_queue.get_queue());
}

// getri: 矩阵求逆 (double)
template<>
void magmaGetriBatched<double>(
    magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, double** dinvA_array, magma_int_t lddia,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
    magma_dgetri_outofplace_batched(n, dA_array, ldda, ipiv_array, 
                                    dinvA_array, lddia, info_array, 
                                    batchsize, magma_queue.get_queue());
}

// potrs: 正定矩阵求解 (double)
template<>
void magmaPotrsBatched<double>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, double** dA_array, magma_int_t ldda,
    double** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
    info = magma_dpotrs_batched(uplo, n, nrhs, dA_array, ldda, dB_array, 
                                lddb, batchsize, magma_queue.get_queue());
}

// cholesky: Cholesky分解 (double)
template<>
void magmaCholeskyBatched<double>(
    magma_uplo_t uplo, magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
    magma_dpotrf_batched(uplo, n, dA_array, ldda, info_array, 
                         batchsize, magma_queue.get_queue());
}

// =============== float类型的MAGMA函数特化实现 ===============

// gesv: 求解线性方程组 (float)
template<>
void magmaGesvBatched<float>(
    magma_int_t n, magma_int_t nrhs, float** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, float** dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array, magma_int_t batch_count, const MAGMAQueue& magma_queue) {
  magma_sgesv_batched(n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, 
                      dinfo_array, batch_count, magma_queue.get_queue());
}

// getrf: LU分解 (float)
template<>
void magmaGetrfBatched<float>(
    magma_int_t m, magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
    magma_sgetrf_batched(m, n, dA_array, ldda, ipiv_array, info_array, 
                         batchsize, magma_queue.get_queue());
}

// getri: 矩阵求逆 (float)
template<>
void magmaGetriBatched<float>(
    magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, float** dinvA_array, magma_int_t lddia,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
    magma_sgetri_outofplace_batched(n, dA_array, ldda, ipiv_array, 
                                    dinvA_array, lddia, info_array, 
                                    batchsize, magma_queue.get_queue());
}

// potrs: 正定矩阵求解 (float)
template<>
void magmaPotrsBatched<float>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, float** dA_array, magma_int_t ldda,
    float** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
    info = magma_spotrs_batched(uplo, n, nrhs, dA_array, ldda, dB_array, 
                                lddb, batchsize, magma_queue.get_queue());
}

// cholesky: Cholesky分解 (float)
template<>
void magmaCholeskyBatched<float>(
    magma_uplo_t uplo, magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
    magma_spotrf_batched(uplo, n, dA_array, ldda, info_array, 
                         batchsize, magma_queue.get_queue());
}
#endif  // USE_MAGMA结束

// 分配固定内存的辅助宏（避免多次重复代码）
#define ALLOCATE_ARRAY(name, type, size, dummy_tensor) \
  auto storage_##name = pin_memory<type>(size, dummy_tensor); \  // 分配固定内存
  name = static_cast<type*>(storage_##name.data());             // 获取指针

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ gesv (求解线性方程组) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_gesv(Tensor& b, Tensor& A, std::vector<int64_t>& infos) {
#ifndef USE_MAGMA
  // 如果没有MAGMA支持，抛出错误
  AT_ERROR("gesv: 编译时未找到MAGMA库。请重新编译并链接MAGMA。");
#else
  // 获取输入张量的数据指针和步长
  auto A_data = A.data<scalar_t>();
  auto b_data = b.data<scalar_t>();
  auto A_mat_stride = matrixStride(A);  // 每批矩阵的步长
  auto b_mat_stride = matrixStride(b);

  // 获取批次大小和矩阵维度
  magma_int_t batch_size = magma_int_cast(batchCount(A), "batchCount");
  magma_int_t n = magma_int_cast(A.size(-2), "A.size(-2)");      // 系数矩阵维度
  magma_int_t nrhs = magma_int_cast(b.size(-1), "b.size(-1)");   // 右侧向量数

  // 声明并分配所需数组
  magma_int_t* info_array;    // 存储每个批次的返回信息
  magma_int_t* ipiv_data;     // 主元交换信息
  magma_int_t** ipiv_array;   // 主元交换信息指针数组
  scalar_t** A_array;         // 系数矩阵指针数组
  scalar_t** b_array;         // 右侧矩阵指针数组

  // 使用宏分配固定内存
  ALLOCATE_ARRAY(info_array, magma_int_t, batch_size, b);
  ALLOCATE_ARRAY(ipiv_data, magma_int_t, batch_size * n, b);
  ALLOCATE_ARRAY(ipiv_array, magma_int_t*, batch_size, b);
  ALLOCATE_ARRAY(A_array, scalar_t*, batch_size, b);
  ALLOCATE_ARRAY(b_array, scalar_t*, batch_size, b);

  // 设置数组指针（每个批次指向对应矩阵）
  for (int64_t i = 0; i < batch_size; i++) {
    A_array[i] = &A_data[i * A_mat_stride];
    b_array[i] = &b_data[i * b_mat_stride];
    ipiv_array[i] = &ipiv_data[i * n];
  }

  // 获取当前设备的MAGMA队列
  MAGMAQueue magma_queue(b.get_device());
  
  // 调用MAGMA的批量求解函数
  magmaGesvBatched<scalar_t>(
      n, nrhs, A_array, n, ipiv_array, b_array, n,
      info_array, batch_size, magma_queue);

  // 将MAGMA返回信息复制到infos向量
  for (int64_t i = 0; i < batch_size; i++) {
    infos[i] = info_array[i];
  }
#endif
}

// gesv的公共接口函数
std::tuple<Tensor, Tensor> _gesv_helper_cuda(const Tensor& self, const Tensor& A) {
  // 初始化信息向量
  std::vector<int64_t> infos(batchCount(self), 0);
  
  // 创建列主序的副本（MAGMA要求列主序）
  auto self_working_copy = cloneBatchedColumnMajor(self);  // 右侧矩阵B
  auto A_working_copy = cloneBatchedColumnMajor(A);        // 系数矩阵A
  
  // 分发到具体类型处理
  AT_DISPATCH_FLOATING_TYPES(self.type(), "gesv", [&]{
    apply_gesv<scalar_t>(self_working_copy, A_working_copy, infos);
  });
  
  // 检查错误信息
  batchCheckErrors(infos, "gesv");
  
  // 返回解矩阵和LU分解后的系数矩阵
  return std::tuple<Tensor, Tensor>(self_working_copy, A_working_copy);
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ inverse (矩阵求逆) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_inverse(Tensor &self, Tensor &self_inv, std::vector<int64_t>& infos) {
#ifndef USE_MAGMA
  AT_ERROR("inverse: 编译时未找到MAGMA库。请重新编译并链接MAGMA。");
#else
  // 获取输入和输出张量数据
  auto self_data = self.data<scalar_t>();
  auto self_mat_stride = matrixStride(self);
  auto self_inv_data = self_inv.data<scalar_t>();
  auto self_inv_mat_stride = matrixStride(self_inv);

  magma_int_t batch_size = magma_int_cast(batchCount(self), "batchCount");
  magma_int_t n = magma_int_cast(self.size(-2), "self.size(-2)");

  // 分配所需数组
  magma_int_t* info_array;
  magma_int_t* ipiv_data;
  magma_int_t** ipiv_array;
  scalar_t** self_array;
  scalar_t** self_inv_array;

  ALLOCATE_ARRAY(info_array, magma_int_t, batch_size, self);
  ALLOCATE_ARRAY(ipiv_data, magma_int_t, batch_size * n, self);
  ALLOCATE_ARRAY(ipiv_array, magma_int_t*, batch_size, self);
  ALLOCATE_ARRAY(self_array, scalar_t*, batch_size, self);
  ALLOCATE_ARRAY(self_inv_array, scalar_t*, batch_size, self_inv);

  // 设置数组指针
  for (int64_t i = 0; i < batch_size; i++) {
    self_array[i] = &self_data[i * self_mat_stride];
    self_inv_array[i] = &self_inv_data[i * self_inv_mat_stride];
    ipiv_array[i] = &ipiv_data[i * n];
  }

  // 获取MAGMA队列
  MAGMAQueue magma_queue(self.get_device());
  
  // 步骤1: 进行批量LU分解
  magmaGetrfBatched<scalar_t>(
    n, n, self_array, n, ipiv_array, info_array,
    batch_size, magma_queue);

  // 步骤2: 基于LU分解结果进行批量矩阵求逆
  magmaGetriBatched<scalar_t>(
    n, self_array, n, ipiv_array, self_inv_array,
    n, info_array, batch_size, magma_queue);

  // 复制返回信息
  for (int64_t i = 0; i < batch_size; i++) {
    infos[i] = info_array[i];
  }
#endif
}

// 矩阵求逆的公共接口
Tensor _inverse_helper_cuda(const Tensor& self) {
  std::vector<int64_t> infos(batchCount(self), 0);
  
  // 创建输入和输出的列主序副本
  auto self_working_copy = cloneBatchedColumnMajor(self);        // 输入矩阵
  auto self_inv_working_copy = cloneBatchedColumnMajor(self);   // 输出矩阵
  
  // 分发到具体类型处理
  AT_DISPATCH_FLOATING_TYPES(self.type(), "inverse", [&]{
    apply_inverse<scalar_t>(
      self_working_copy, self_inv_working_copy, infos);
  });
  
  // 检查错误信息
  batchCheckErrors(infos, "inverse");
  
  // 返回求逆结果
  return self_inv_working_copy;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ potrs (正定矩阵求解) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_potrs(Tensor& b, Tensor& A, bool upper, int64_t& info) {
#ifndef USE_MAGMA
  AT_ERROR("potrs: 编译时未找到MAGMA库。请重新编译并链接MAGMA。");
#else
  // 设置上三角或下三角标志
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;

  auto A_data = A.data<scalar_t>();
  auto b_data = b.data<scalar_t>();
  auto A_mat_stride = matrixStride(A);
  auto b_mat_stride = matrixStride(b);

  magma_int_t batch_size = magma_int_cast(batchCount(A), "batchCount");
  magma_int_t n = magma_int_cast(A.size(-2), "A.size(-2)");
  magma_int_t nrhs = magma_int_cast(b.size(-1), "b.size(-1)");

  magma_int_t info_tmp;  // MAGMA返回信息
  magma_int_t* ipiv_data;
  magma_int_t** ipiv_array;
  scalar_t** A_array;
  scalar_t** b_array;

  // 分配所需数组（注意：potrs不需要主元信息，但为统一接口保留）
  ALLOCATE_ARRAY(ipiv_data, magma_int_t, batch_size * n, b);
  ALLOCATE_ARRAY(ipiv_array, magma_int_t*, batch_size, b);
  ALLOCATE_ARRAY(A_array, scalar_t*, batch_size, b);
  ALLOCATE_ARRAY(b_array, scalar_t*, batch_size, b);

  // 设置数组指针
  for (int64_t i = 0; i < batch_size; i++) {
    A_array[i] = &A_data[i * A_mat_stride];
    b_array[i] = &b_data[i * b_mat_stride];
    ipiv_array[i] = &ipiv_data[i * n];  // 实际未使用
  }

  // 获取MAGMA队列
  MAGMAQueue magma_queue(b.get_device());
  
  // 调用MAGMA的正定矩阵求解
  magmaPotrsBatched<scalar_t>(
      uplo, n, nrhs, A_array, n, b_array, n,
      info_tmp, batch_size, magma_queue);

  // 返回信息
  info = info_tmp;
#endif
}

// potrs的公共接口
Tensor _potrs_helper_cuda(const Tensor& self, const Tensor& A, bool upper) {
  int64_t info = 0;
  
  // 创建列主序副本
  auto self_working_copy = cloneBatchedColumnMajor(self);  // 右侧矩阵
  auto A_working_copy = cloneBatchedColumnMajor(A);        // Cholesky分解后的矩阵
  
  // 分发到具体类型处理
  AT_DISPATCH_FLOATING_TYPES(self.type(), "potrs", [&]{
    apply_potrs<scalar_t>(self_working_copy, A_working_copy, upper, info);
  });
  
  // 检查错误
  AT_CHECK(info == 0, "MAGMA potrs : 无效参数: ", -info);
  
  // 返回解矩阵
  return self_working_copy;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky (Cholesky分解) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_cholesky(Tensor& self, bool upper, std::vector<int64_t>& infos) {
#ifndef USE_MAGMA
  AT_ERROR("cholesky: 编译时未找到MAGMA库。请重新编译并链接MAGMA。");
#else
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;

  auto self_data = self.data<scalar_t>();
  auto self_mat_stride = matrixStride(self);

  magma_int_t batch_size = magma_int_cast(batchCount(self), "batchCount");
  magma_int_t n = magma_int_cast(self.size(-2), "self.size(-2)");

  magma_int_t* info_array;
  scalar_t** self_array;

  // 分配所需数组（Cholesky不需要主元信息）
  ALLOCATE_ARRAY(info_array, magma_int_t, batch_size, self);
  ALLOCATE_ARRAY(self_array, scalar_t*, batch_size, self);

  // 设置数组指针
  for (int64_t i = 0; i < batch_size; i++) {
    self_array[i] = &self_data[i * self_mat_stride];
  }

  // 获取MAGMA队列
  MAGMAQueue magma_queue(self.get_device());
  
  // 调用MAGMA的批量Cholesky分解
  magmaCholeskyBatched<scalar_t>(
    uplo, n, self_array, n, info_array,
    batch_size, magma_queue);

  // 复制返回信息
  for (int64_t i = 0; i < batch_size; i++) {
    infos[i] = info_array[i];
  }
#endif
}

// Cholesky分解的公共接口
Tensor _cholesky_helper_cuda(const Tensor& self, bool upper) {
  std::vector<int64_t> infos(batchCount(self), 0);
  Tensor self_working_copy;
  
  // MAGMA使用列主序且只处理下三角，因此需要转换
  if (upper) {
    // 上三角存储：转置为下三角（列主序等价于行主序的上三角）
    self_working_copy = cloneBatchedColumnMajor(self.transpose(-1, -2));
  } else {
    // 下三角存储：直接使用列主序
    self_working_copy = cloneBatchedColumnMajor(self);
  }

  // 分发处理（总是以下三角形式处理）
  AT_DISPATCH_FLOATING_TYPES(self.type(), "cholesky", [&]{
    apply_cholesky<scalar_t>(self_working_copy, false, infos);
  });
  
  // 检查错误
  batchCheckErrors(infos, "cholesky");
  
  // 转换回原始布局
  if (upper) {
    return self_working_copy.transpose(-1, -2);
  } else {
    return self_working_copy;
  }
}

}}  // namespace at::native

// 取消分配宏定义
#undef ALLOCATE_ARRAY