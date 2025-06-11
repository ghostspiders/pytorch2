#pragma once

#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/Exceptions.h"

#include "cudnn-wrapper.h"
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include "ATen/cuda/ATenCUDAGeneral.h"
#include <cuda.h>

#if CUDNN_VERSION < 7000

#include <curand_kernel.h>

/*  
注释 [cuDNN dropout描述符初始化]  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

多数情况下，cuDNN中的描述符设置是低开销的（例如cudnnSetTensorNdDescriptor）。  
但cudnnSetDropoutDescriptor在cuDNN 6/7中会执行昂贵的预计算来初始化随机数生成器状态。  
cuDNN 6中这是初始化dropout描述符的唯一官方方式，意味着规范用法需要缓存描述符。  
然而ATen接口(1)无状态（无法缓存描述符）且(2)不接受用户自定义类型（无法传递描述符），导致两难境地。  

cuDNN 7新增了cudnnRestoreDropoutDescriptor函数，可通过预初始化CUDA张量跳过昂贵初始化过程。  
遗憾的是该函数在cuDNN 6中不可用。  

解决方案是通过逆向工程获取底层dropout描述符结构布局，自主实现cudnnRestoreDropoutDescriptor功能。  
*/  

// 基于cuDNN 6逆向工程实现，参见注释[cuDNN dropout描述符初始化]  

struct cudnnDropoutStruct {
  float dropout;
  int nstates;
  void * states;
};

#endif

namespace at { namespace native {

// TODO: Add constructors for all of the descriptors

inline int dataSize(cudnnDataType_t dataType)
{
  switch (dataType) {
    case CUDNN_DATA_HALF: return 2;
    case CUDNN_DATA_FLOAT: return 4;
    default: return 8;
  }
}

// 对于大小为1的维度，其步长并非唯一确定——实际上可以设为任意值，
// 因为该维度大小为1意味着永远不会真正通过此步长移动指针。

// 但cuDNN对步长有更严格的要求：
// 若传递的是连续内存输入，则必须保证第i维的步长等于
// 第i+1维到末尾所有维度大小的乘积。此步长是唯一确定的。
// 本函数会原地修改'stride'数组以确保该不变式成立。

static inline void fixSizeOneDimStride(int dim, const int *size, int *stride) {
  int64_t z = 1;
  for(int d = dim-1; d >= 0; d--)
  {
    if (size[d] == 1) {
      stride[d] = z;
    } else {
      z *= size[d];
    }
  }
}

template <typename T, cudnnStatus_t (*dtor)(T*)>
struct DescriptorDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      AT_CUDNN_CHECK(dtor(x));
    }
  }
};

// A generic class for wrapping cuDNN descriptor types.  All you need
// is to give the underlying type the Descriptor_t points to (usually,
// if it's cudnnTensorDescriptor_t it points to cudnnTensorStruct),
// the constructor and the destructor.  Subclasses are responsible
// for defining a set() function to actually set the descriptor.
//
// Descriptors default construct to a nullptr, and have a descriptor
// initialized the first time you call set() or any other initializing
// function.
template <typename T, cudnnStatus_t (*ctor)(T**), cudnnStatus_t (*dtor)(T*)>
class AT_CUDA_API Descriptor
{
public:
  // TODO: Figure out why const-correctness doesn't work here

  // Use desc() to access the underlying descriptor pointer in
  // a read-only fashion.  Most client code should use this.
  // If the descriptor was never initialized, this will return
  // nullptr.
  T* desc() const { return desc_.get(); }
  T* desc() { return desc_.get(); }

  // Use mut_desc() to access the underlying desciptor pointer
  // if you intend to modify what it points to (e.g., using
  // cudnnSetFooDescriptor).  This will ensure that the descriptor
  // is initialized.  Code in this file will use this function.
  T* mut_desc() { init(); return desc_.get(); }
protected:
  void init() {
    if (desc_ == nullptr) {
      T* raw_desc;
      AT_CUDNN_CHECK(ctor(&raw_desc));
      desc_.reset(raw_desc);
    }
  }
private:
  std::unique_ptr<T, DescriptorDeleter<T, dtor>> desc_;
};

class AT_CUDA_API TensorDescriptor
  : public Descriptor<cudnnTensorStruct,
                      &cudnnCreateTensorDescriptor,
                      &cudnnDestroyTensorDescriptor>
{
public:
  TensorDescriptor() {}
  explicit TensorDescriptor(const at::Tensor &t, size_t pad = 0) {
    set(t, pad);
  }

  // Note [CuDNN broadcast padding]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // pad specifies the minimum dimensionality of the tensor descriptor
  // we produce (it doesn't have anything to do with, e.g., convolution
  // padding).  If 't' is lower-dimensional than 'pad', the remaining
  // dimensions (on the right) are padded with ones.  This doesn't
  // affect the underlying data layout.  This is particularly useful for
  // dealing with a pecularity of the CuDNN API, which is that broadcasting in CuDNN is
  // done in two steps: first, the client code is expected to pad out
  // (the dimensions) input tensors to be the same dimension as the
  // target broadcast, and then second, CuDNN takes of actually
  // broadcasting size 1 dimensions.

  void set(const at::Tensor &t, size_t pad = 0);
  void set(cudnnDataType_t dataType, IntList sizes, IntList strides, size_t pad = 0);

  void print();

private:
  void set(cudnnDataType_t dataType, int dim, int* size, int* stride) {
    fixSizeOneDimStride(dim, size, stride);
    AT_CUDNN_CHECK(cudnnSetTensorNdDescriptor(mut_desc(), dataType, dim, size, stride));
  }
};

std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d);

class FilterDescriptor
  : public Descriptor<cudnnFilterStruct,
                      &cudnnCreateFilterDescriptor,
                      &cudnnDestroyFilterDescriptor>
{
public:
  void set(const at::Tensor &t, int64_t pad = 0);

private:
  void set(cudnnDataType_t dataType, int dim, int* size) {
    AT_CUDNN_CHECK(cudnnSetFilterNdDescriptor(mut_desc(), dataType, CUDNN_TENSOR_NCHW, dim, size));
  }
};

struct AT_CUDA_API ConvolutionDescriptor
  : public Descriptor<cudnnConvolutionStruct,
                      &cudnnCreateConvolutionDescriptor,
                      &cudnnDestroyConvolutionDescriptor>
{
  void set(cudnnDataType_t dataType, int dim, int* pad, int* stride, int * upscale /* aka dilation */, int groups) {
    cudnnDataType_t mathType = dataType;
    if (dataType == CUDNN_DATA_HALF) mathType = CUDNN_DATA_FLOAT;
    AT_CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(mut_desc(), dim, pad, stride, upscale,
                                          CUDNN_CROSS_CORRELATION, mathType));
#if CUDNN_VERSION >= 7000
    AT_CUDNN_CHECK(cudnnSetConvolutionGroupCount(mut_desc(), groups));
    AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(), CUDNN_DEFAULT_MATH));
    if(dataType == CUDNN_DATA_HALF)
      AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(), CUDNN_TENSOR_OP_MATH));
#endif
  }
};

struct AT_CUDA_API SpatialTransformerDescriptor
  : public Descriptor<cudnnSpatialTransformerStruct,
                      &cudnnCreateSpatialTransformerDescriptor,
                      &cudnnDestroySpatialTransformerDescriptor>
{
  void set(cudnnDataType_t dataType, int dim, int* size) {
    AT_CUDNN_CHECK(cudnnSetSpatialTransformerNdDescriptor(mut_desc(), CUDNN_SAMPLER_BILINEAR, dataType, dim, size));
  }
};

#if CUDNN_VERSION < 7000

// See Note [cuDNN dropout descriptor initialization]
inline cudnnStatus_t cudnnRestoreDropoutDescriptor(
    cudnnDropoutDescriptor_t dropoutDesc,
    cudnnHandle_t handle,
    float dropout,
    void *states,
    size_t stateSizeInBytes,
    unsigned long long seed) {
  // Try to accurately simulate cuDNN's behavior, for our cuDNN 6 friends.
  // This is not entirely accurate but is good enough to catch some API
  // uses which would not be compatible in cuDNN 7.  Feel free to fix
  // this if you notice something is wrong.
  if (states == nullptr) return CUDNN_STATUS_INVALID_VALUE;
  if (stateSizeInBytes == 0) return CUDNN_STATUS_INVALID_VALUE;
  size_t expectedStateSizeInBytes;
  // State size will differ depending on size of GPU
  auto ret = cudnnDropoutGetStatesSize(handle, &expectedStateSizeInBytes);
  if (ret != CUDNN_STATUS_SUCCESS) return ret;
  if (expectedStateSizeInBytes != stateSizeInBytes) return CUDNN_STATUS_INVALID_VALUE;
  dropoutDesc->dropout = dropout;
  dropoutDesc->nstates = (int)stateSizeInBytes/sizeof(curandState_t);
  dropoutDesc->states = states;
  return CUDNN_STATUS_SUCCESS;
}

#endif // CUDNN_VERSION

struct AT_CUDA_API DropoutDescriptor
  : public Descriptor<cudnnDropoutStruct,
                      &cudnnCreateDropoutDescriptor,
                      &cudnnDestroyDropoutDescriptor>
{
  at::Tensor state;

  // Initialize a dropout descriptor's RNG state.
  // WARNING: This function is very expensive, avoid calling this function!
  // NB: it takes a Type so that we can generate a Variable if necessary.
  void initialize_rng(cudnnHandle_t handle, float dropout, long long int seed, const TensorOptions& options) {
    AT_ASSERTM(dropout > 0, "dropout must be nonzero; otherwise call set_no_dropout");
    size_t state_size;
    AT_CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &state_size));
    AT_ASSERT(options.device().type() == kCUDA);
    AT_ASSERT(options.dtype() == kByte);
    state = at::empty({static_cast<int64_t>(state_size)}, options);
    AT_CUDNN_CHECK(cudnnSetDropoutDescriptor(mut_desc(), handle, dropout, state.data_ptr(), state_size, seed));
  }

  // Restore a dropout descriptor given a dropout probability and existing RNG state.
  // See Note [cuDNN dropout descriptor initialization]
  void set(cudnnHandle_t handle, float dropout, at::Tensor state_) {
    AT_ASSERTM(dropout > 0, "dropout must be nonzero; otherwise call set_no_dropout");
    state = state_;
    void *state_ptr = state.data_ptr();
    size_t state_size = state.size(0);
    // NB: The seed doesn't actually matter, so we give a dummy value
    AT_CUDNN_CHECK(cudnnRestoreDropoutDescriptor(mut_desc(), handle, dropout, state_ptr, state_size, 0 /* seed */));
  }

  // Restore a dropout descriptor corresponding to no dropout
  // See Note [cuDNN dropout descriptor initialization]
  void set_no_dropout(cudnnHandle_t handle) {
    // NB: seed doesn't matter when dropout = 0, because no random number
    // initialization actually takes place when there is no dropout.
    // NB: Empirically, cudnnSetDropoutDescriptor is cheap when
    // dropoot == 0
    AT_CUDNN_CHECK(cudnnSetDropoutDescriptor(mut_desc(), handle, 0 /* dropout */, nullptr, 0 /* state_size */, 0 /* seed */));
  }
};

struct AT_CUDA_API RNNDescriptor
  : public Descriptor<cudnnRNNStruct,
                      &cudnnCreateRNNDescriptor,
                      &cudnnDestroyRNNDescriptor>
{
  DropoutDescriptor dropout_desc_;
  void set(cudnnHandle_t handle, int hidden_size, int num_layers, DropoutDescriptor&& dropout_desc,
           cudnnRNNInputMode_t input_mode, cudnnDirectionMode_t bidirectional,
           cudnnRNNMode_t mode, cudnnDataType_t datatype, cudnnRNNAlgo_t algo) {
    dropout_desc_ = std::move(dropout_desc);
    AT_CUDNN_CHECK(cudnnSetRNNDescriptor_v6(
          handle,
          mut_desc(),
          hidden_size,
          num_layers,
          dropout_desc_.desc(),
          input_mode,
          bidirectional,
          mode,
          algo,
          datatype));
#if CUDNN_VERSION >= 7000 && CUDA_VERSION >= 9000
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    if (prop->major >= 7) {
      if (datatype == CUDNN_DATA_HALF) {
        cudnnSetRNNMatrixMathType(mut_desc(), CUDNN_TENSOR_OP_MATH);
      } else {
        // Technically, as the default it's not necessary to explicitly
        // set this.
        cudnnSetRNNMatrixMathType(mut_desc(), CUDNN_DEFAULT_MATH);
      }
    }
#endif
  }
};

#if CUDNN_VERSION >= 7000

struct AT_CUDA_API CTCLossDescriptor
  : public Descriptor<cudnnCTCLossStruct,
                      &cudnnCreateCTCLossDescriptor,
                      &cudnnDestroyCTCLossDescriptor>
{
  void set(cudnnDataType_t datatype) {
    AT_CUDNN_CHECK(cudnnSetCTCLossDescriptor(mut_desc(), datatype));
  }
};

#endif

union Constant
{
  float f;
  double d;
  Constant(cudnnDataType_t dataType, double value) {
    if (dataType == CUDNN_DATA_HALF || dataType == CUDNN_DATA_FLOAT) {
      f = static_cast<float>(value);
    } else {
      d = value;
    }
  }
};

}}  // namespace
