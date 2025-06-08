#pragma once

#include <c10/macros/Macros.h>
#include <stdint.h>
#include <cstddef>

namespace at {

// PtrTraits模板参数用于TensorAccessor/PackedTensorAccessor
// 在CUDA中支持对数据指针使用__restrict__关键字/修饰符
template <typename T>
struct DefaultPtrTraits {
  typedef T* PtrType;  // 默认指针类型
};

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict__ PtrType;  // 带restrict限定的指针类型，用于CUDA/HIP
};
#endif

// TensorAccessorBase和TensorAccessor用于CPU和CUDA张量
// 对于CUDA张量，它仅用于设备代码中
// PtrTraits参数仅与CUDA相关，用于支持`__restrict__`指针

// 张量访问器基类模板
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class TensorAccessorBase {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;  // 定义指针类型

  // 构造函数(主机和设备)
  C10_HOST_DEVICE TensorAccessorBase(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : data_(data_), sizes_(sizes_), strides_(strides_) {}
  
  // 获取尺寸列表(主机)
  C10_HOST IntList sizes() const {
    return IntList(sizes_,N);
  }
  
  // 获取步长列表(主机)
  C10_HOST IntList strides() const {
    return IntList(strides_,N);
  }
  
  // 获取指定维度的步长(主机和设备)
  C10_HOST_DEVICE index_t stride(index_t i) const {
    return strides_[i];
  }
  
  // 获取指定维度的大小(主机和设备)
  C10_HOST_DEVICE index_t size(index_t i) const {
    return sizes_[i];
  }
  
  // 获取数据指针(主机和设备)
  C10_HOST_DEVICE PtrType data() {
    return data_;
  }
  
  // 获取常量数据指针(主机和设备)
  C10_HOST_DEVICE const PtrType data() const {
    return data_;
  }
  
protected:
  PtrType data_;          // 数据指针
  const index_t* sizes_;  // 各维度大小数组
  const index_t* strides_;// 各维度步长数组
};

// 张量访问器模板(用于CPU张量的Tensor.accessor<T, N>())
// 对于CUDA张量，主机端使用PackedTensorAccessor，设备端索引使用TensorAccessor
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class TensorAccessor : public TensorAccessorBase<T,N,PtrTraits,index_t> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  // 构造函数(主机和设备)
  C10_HOST_DEVICE TensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : TensorAccessorBase<T, N, PtrTraits, index_t>(data_,sizes_,strides_) {}

  // 重载[]操作符，返回降维后的访问器(主机和设备)
  C10_HOST_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) {
    return TensorAccessor<T,N-1,PtrTraits,index_t>(
      this->data_ + this->strides_[0]*i,
      this->sizes_+1,
      this->strides_+1);
  }

  // 常量版本的[]操作符
  C10_HOST_DEVICE const TensorAccessor<T, N-1, PtrTraits, index_t> operator[](index_t i) const {
    return TensorAccessor<T,N-1,PtrTraits,index_t>(
      this->data_ + this->strides_[0]*i,
      this->sizes_+1,
      this->strides_+1);
  }
};

// 一维张量访问器特化版本
template<typename T, template <typename U> class PtrTraits, typename index_t>
class TensorAccessor<T,1,PtrTraits,index_t> : public TensorAccessorBase<T,1,PtrTraits,index_t> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  // 构造函数(主机和设备)
  C10_HOST_DEVICE TensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : TensorAccessorBase<T, 1, PtrTraits, index_t>(data_,sizes_,strides_) {}
  
  // 重载[]操作符，返回元素引用(主机和设备)
  C10_HOST_DEVICE T & operator[](index_t i) {
    return this->data_[this->strides_[0]*i];
  }
  
  // 常量版本的[]操作符
  C10_HOST_DEVICE const T & operator[](index_t i) const {
    return this->data_[this->strides_[0]*i];
  }
};

// PackedTensorAccessorBase和PackedTensorAccessor用于CUDA张量的主机端
// 与TensorAccessor不同，它们在实例化时复制步长和大小(在主机上)
// 以便在调用内核时将它们传输到设备上
// 在设备上，多维张量的索引会转换为TensorAccessor
// 如果要标记张量数据指针为__restrict__，请使用RestrictPtrTraits作为PtrTraits
// 从数据、大小、步长实例化仅在主机上需要，设备上没有std::copy

// 打包张量访问器基类模板
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class PackedTensorAccessorBase {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  
  // 构造函数(主机)
  C10_HOST PackedTensorAccessorBase(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : data_(data_) {
    std::copy(sizes_, sizes_ + N, std::begin(this->sizes_));
    std::copy(strides_, strides_ + N, std::begin(this->strides_));
  }

  // 如果index_t不是int64_t，我们还需要一个int64_t的构造函数
  template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
  C10_HOST PackedTensorAccessorBase(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_)
      : data_(data_) {
    for (int i = 0; i < N; i++) {
      this->sizes_[i] = sizes_[i];
      this->strides_[i] = strides_[i];
    }
  }

  // 获取指定维度的步长(主机和设备)
  C10_HOST_DEVICE index_t stride(index_t i) const {
    return strides_[i];
  }
  
  // 获取指定维度的大小(主机和设备)
  C10_HOST_DEVICE index_t size(index_t i) const {
    return sizes_[i];
  }
  
  // 获取数据指针(主机和设备)
  C10_HOST_DEVICE PtrType data() {
    return data_;
  }
  
  // 获取常量数据指针(主机和设备)
  C10_HOST_DEVICE const PtrType data() const {
    return data_;
  }
  
protected:
  PtrType data_;    // 数据指针
  index_t sizes_[N];   // 存储各维度大小的数组
  index_t strides_[N]; // 存储各维度步长的数组
};

// 打包张量访问器模板
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class PackedTensorAccessor : public PackedTensorAccessorBase<T,N,PtrTraits,index_t> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  // 构造函数(主机)
  C10_HOST PackedTensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : PackedTensorAccessorBase<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

  // int64_t版本的构造函数(主机)
  template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
  C10_HOST PackedTensorAccessor(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_)
      : PackedTensorAccessorBase<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

  // 重载[]操作符，返回降维后的访问器(设备)
  C10_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) {
    index_t* new_sizes = this->sizes_ + 1;
    index_t* new_strides = this->strides_ + 1;
    return TensorAccessor<T,N-1,PtrTraits,index_t>(
      this->data_ + this->strides_[0]*i, 
      new_sizes, 
      new_strides);
  }

  // 常量版本的[]操作符
  C10_DEVICE const TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) const {
    const index_t* new_sizes = this->sizes_ + 1;
    const index_t* new_strides = this->strides_ + 1;
    return TensorAccessor<T,N-1,PtrTraits,index_t>(
      this->data_ + this->strides_[0]*i, 
      new_sizes, 
      new_strides);
  }
};

// 一维打包张量访问器特化版本
template<typename T, template <typename U> class PtrTraits, typename index_t>
class PackedTensorAccessor<T,1,PtrTraits,index_t> : public PackedTensorAccessorBase<T,1,PtrTraits,index_t> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  
  // 构造函数(主机)
  C10_HOST PackedTensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : PackedTensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}

  // int64_t版本的构造函数(主机)
  template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
  C10_HOST PackedTensorAccessor(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_)
      : PackedTensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}

  // 重载[]操作符，返回元素引用(设备)
  C10_DEVICE T & operator[](index_t i) {
    return this->data_[this->strides_[0] * i];
  }
  
  // 常量版本的[]操作符
  C10_DEVICE const T& operator[](index_t i) const {
    return this->data_[this->strides_[0]*i];
  }
};

} // namespace at