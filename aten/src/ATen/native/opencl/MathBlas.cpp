#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/Exception.h>
#include <ATen/Backend.h>
#include <ATen/Utils.h>
#include <ATen/NamedTensorUtils.h>

namespace at {
namespace native {

Tensor & _opencl_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
    Backend backend = Backend::CPU;
    if (detail::getCUDAHooks().hasCUDA()) {
        backend = Backend::CUDA;
    }
    result.copy_(self.toBackend(backend).addmm(mat1.toBackend(backend), mat2.toBackend(backend), beta, alpha));
    return result;
}

Tensor _opencl_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
    Backend backend = Backend::CPU;
    if (detail::getCUDAHooks().hasCUDA()) {
        backend = Backend::CUDA;
    }
    return self.toBackend(backend).addmm(mat1.toBackend(backend), mat2.toBackend(backend), beta, alpha).toBackend(self.options().backend());
}

Tensor & _opencl_addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
    Backend backend = Backend::CPU;
    if (detail::getCUDAHooks().hasCUDA()) {
        backend = Backend::CUDA;
    }
    auto self_ = self.toBackend(backend);
    self.copy_(self_.addmm_(mat1.toBackend(backend), mat2.toBackend(backend), beta, alpha));
    return self;
}

Tensor _opencl_unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) {
    Backend backend = Backend::CPU;
    if (detail::getCUDAHooks().hasCUDA()) {
        backend = Backend::CUDA;
    }
    return self.toBackend(backend).unfold(dimension, size, step).toBackend(self.options().backend());
}

}} // namespace at::native
