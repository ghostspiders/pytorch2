#include "ATen/native/BinaryOps.h"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

// 定义二元操作的分发函数
DEFINE_DISPATCH(add_stub);
DEFINE_DISPATCH(sub_stub);
DEFINE_DISPATCH(mul_stub);
DEFINE_DISPATCH(div_stub);

// add_out 函数：将两个张量相加，结果存储在 result 中
Tensor& add_out(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha) {
    // 如果 other 是稀疏张量
    if (other.is_sparse()) {
        // 如果 result 未定义，则初始化为一个空张量
        if (!result.defined()) {
            result = at::empty({0}, self.options());
        }
        // 如果 self 也是稀疏张量，调用 _sparse_add_out 函数
        if (self.is_sparse()) {
            at::_sparse_add_out(result, self, other, alpha);
        } 
        // 如果 self 是密集张量，调用 _sparse_dense_add_out 函数
        else {
            at::_sparse_dense_add_out(result, self, SparseTensorRef(other), alpha);
        }
        return result;
    } 
    // 如果 self 是稀疏张量，报错
    else if (self.is_sparse()) {
        AT_ERROR("add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
    }
    // 如果 self 和 other 都是密集张量
    auto iter = TensorIterator::binary_op(result, self, other);
    // 调用分发函数 add_stub
    add_stub(iter->device_type(), *iter, alpha);
    result = iter->output();
    return result;
}

// add 函数：将两个张量相加，返回结果张量
Tensor add(const Tensor& self, const Tensor& other, Scalar alpha) {
    Tensor result;
    return native::add_out(result, self, other, alpha);
}

// add_ 函数：将 self 和 other 相加，结果存储在 self 中
Tensor& add_(Tensor& self, const Tensor& other, Scalar alpha) {
    return native::add_out(self, self, other, alpha);
}

// div_out 函数：将 self 除以 other，结果存储在 result 中
Tensor& div_out(Tensor& result, const Tensor& self, const Tensor& other) {
    // 如果 self 是稀疏张量
    if (self.is_sparse()) {
        // 如果 result 未定义，则初始化为一个空张量
        if (!result.defined()) {
            result = at::empty({0}, self.options());
        }
        // 如果 other 不是标量，报错
        if (other.dim() != 0) {
            AT_ERROR("div(): sparse division only supports division by a scalar ",
                "(got shape ", other.sizes(), " for argument 'other')");
        }
        // 调用 _sparse_div_zerodim_out 函数
        return at::_sparse_div_zerodim_out(result, self, other);
    }
    // 如果 self 是密集张量
    auto iter = TensorIterator::binary_op(result, self, other);
    // 调用分发函数 div_stub
    div_stub(iter->device_type(), *iter);
    result = iter->output();
    return result;
}

// div 函数：将 self 除以 other，返回结果张量
Tensor div(const Tensor& self, const Tensor& other) {
    Tensor result;
    return native::div_out(result, self, other);
}

// div_ 函数：将 self 除以 other，结果存储在 self 中
Tensor& div_(Tensor& self, const Tensor& other) {
    return native::div_out(self, self, other);
}

// mul_out 函数：将 self 和 other 相乘，结果存储在 result 中
Tensor& mul_out(Tensor& result, const Tensor& self, const Tensor& other) {
    // 如果 self 或 other 是稀疏张量
    if (self.is_sparse() || other.is_sparse()) {
        // 如果 result 未定义，则初始化为一个空张量
        if (!result.defined()) {
            result = at::empty({0}, self.options());
        }
        // 调用 _sparse_mul_out 函数
        return at::_sparse_mul_out(result, self, other);
    }
    // 如果 self 和 other 都是密集张量
    auto iter = TensorIterator::binary_op(result, self, other);
    // 调用分发函数 mul_stub
    mul_stub(iter->device_type(), *iter);
    result = iter->output();
    return result;
}

// mul 函数：将 self 和 other 相乘，返回结果张量
Tensor mul(const Tensor& self, const Tensor& other) {
    Tensor result;
    return native::mul_out(result, self, other);
}

// mul_ 函数：将 self 和 other 相乘，结果存储在 self 中
Tensor& mul_(Tensor& self, const Tensor& other) {
    return native::mul_out(self, self, other);
}

// sub_out 函数：将 self 减去 other，结果存储在 result 中
Tensor& sub_out(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha) {
    // 如果 other 是稀疏张量
    if (other.is_sparse()) {
        // 如果 result 未定义，则初始化为一个空张量
        if (!result.defined()) {
            result = at::empty({0}, self.options());
        }
        // 如果 self 和 other 的尺寸不匹配，报错
        if (!self.sizes().equals(other.sizes())) {
            AT_ERROR("sizes do not match");
        }
        // 如果 self 是稀疏张量，调用 _sparse_add_out 函数
        if (self.is_sparse()) {
            at::_sparse_add_out(result, self, other, -alpha);
        } 
        // 如果 self 是密集张量，调用 _sparse_dense_add_out 函数
        else {
            at::_sparse_dense_add_out(result, self, SparseTensorRef(other), -alpha);
        }
        return result;
    } 
    // 如果 self 是稀疏张量，报错
    else if (self.is_sparse()) {
        AT_ERROR("sub(sparse, dense) is not supported. Use sub(dense, sparse) instead.");
    }
    // 如果 self 和 other 都是密集张量
    auto iter = TensorIterator::binary_op(result, self, other);
    // 调用分发函数 sub_stub
    sub_stub(iter->device_type(), *iter, alpha);
    result = iter->output();
    return result;
}

// sub 函数：将 self 减去 other，返回结果张量
Tensor sub(const Tensor& self, const Tensor& other, Scalar alpha) {
    Tensor result;
    return native::sub_out(result, self, other, alpha);
}

// sub_ 函数：将 self 减去 other，结果存储在 self 中
Tensor& sub_(Tensor& self, const Tensor& other, Scalar alpha) {
    return native::sub_out(self, self, other, alpha);
}

// rsub 函数：将 other 减去 self，返回结果张量
Tensor rsub(const Tensor& self, const Tensor& other, Scalar alpha) {
    return native::sub(other, self, alpha);
}

// scalar_tensor 函数：将标量转换为张量
static Tensor scalar_tensor(Scalar scalar) {
    auto tensor = scalar_to_tensor(scalar);
    tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
    return tensor;
}

// 以下函数用于处理标量和张量的运算
Tensor add(const Tensor& self, Scalar other, Scalar alpha) {
    return native::add(self, scalar_tensor(other), alpha);
}

Tensor& add_(Tensor& self, Scalar other, Scalar alpha) {
    return native::add_(self, scalar_tensor(other), alpha);
}

Tensor div(const Tensor& self, Scalar other) {
    return native::div(self, scalar_tensor(other));
}

Tensor& div_(Tensor& self, Scalar other) {
    return native::div_(self, scalar_tensor(other));
}

Tensor mul(const Tensor& self, Scalar other) {
    return native::mul(self, scalar_tensor(other));
}

Tensor& mul_(Tensor& self, Scalar other) {
    return native::mul_(self, scalar_tensor(other));
}

Tensor sub(const Tensor& self, Scalar other, Scalar alpha) {
    return native::sub(self, scalar_tensor(other), alpha);
}

Tensor& sub_(Tensor& self, Scalar other, Scalar alpha) {
    return native::sub_(self, scalar_tensor(other), alpha);
}

Tensor rsub(const Tensor& self, Scalar other, Scalar alpha) {
    return native::rsub(self, scalar_tensor(other), alpha);
}

}
}  // namespace at