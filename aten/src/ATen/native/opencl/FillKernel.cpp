#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/opencl/OpenCLContext.h>
#include <ATen/opencl/OpenCLEvent.h>
#include <c10/opencl/OpenCLStream.h>
#include <c10/opencl/OpenCLGuard.h>
#include <ATen/opencl/PinnedMemoryAllocator.h>
#include <ATen/native/opencl/Utils.h>
#include <ATen/native/Fill.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

void fill_kernel_opencl(TensorIterator& iter, Scalar value) {
  if (iter.numel() == 0) {
    return;
  }
  Tensor self = iter.tensor(0);
  TensorImpl* self_ = checked_tensor_unwrap(self, "self", 2, "fill_kernel_opencl", false, c10::Backend::OpenCL, iter.dtype());
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "fill_opencl", [&]() {
    auto value_converted = value.to<scalar_t>();
    auto kernel_name = "cast_" + getOpenCLKernelTypeSuffix(typeMetaToScalarType(self_->dtype())) + "_" + getOpenCLKernelTypeSuffix(iter.dtype()) + "_s";
    auto kernel_opt = c10::opencl::opencl_kernel(kernel_name);
    TORCH_INTERNAL_ASSERT(kernel_opt.has_value(), "Kernel not found \"", kernel_name, "\"");
    auto stream = at::opencl::getCurrentOpenCLStream(self_->device().index());
    auto kernel = kernel_opt.value();
    AT_OPENCL_CHECK(kernel.setArg<scalar_t>(0, value_converted));
    AT_OPENCL_CHECK(kernel.setArg<cl_mem>(1, (*toBuffer(self_->data()))()));
    AT_OPENCL_CHECK(stream.stream()->enqueueNDRangeKernel(kernel, /*offset=*/0, self_->numel(), 1));
    C10_OPENCL_CHECK(syncOpenCLPointer(self_->data()));
  });
}

REGISTER_DISPATCH(fill_stub, &fill_kernel_opencl);

} // namespace native
} // namespace at
