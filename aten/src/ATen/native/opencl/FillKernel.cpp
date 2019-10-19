#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/opencl/OpenCLContext.h>
#include <ATen/opencl/OpenCLEvent.h>
#include <c10/opencl/OpenCLStream.h>
#include <c10/opencl/OpenCLGuard.h>
#include <c10/opencl/OpenCLFunctions.h>
#include <ATen/opencl/PinnedMemoryAllocator.h>
#include <ATen/native/opencl/Utils.h>
#include <ATen/native/Fill.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/NativeFunctions.h>

namespace at { namespace native {

struct OpenCLCastFunctor : std::function<cl_int(cl::Buffer, cl::Buffer, at::native::opencl::OpenCLCastType, at::native::opencl::OpenCLCastType)> {};

void fill_kernel_opencl(TensorIterator& iter, Scalar value) {
  if (iter.numel() == 0) {
    return;
  }
  Tensor self = iter.tensor(0);
  Tensor scalar_tensor = at::native::empty_opencl({0}, self.options(), self.suggest_memory_format());
  scalar_tensor.fill_(value);
  TensorImpl* self_ = checked_tensor_unwrap(self, "self", 2, "fill_kernel_opencl", false, c10::Backend::OpenCL, iter.dtype());
  TensorImpl* scalar_tensor_ = checked_tensor_unwrap(scalar_tensor, "value", 2, "fill_kernel_opencl", false, c10::Backend::OpenCL, iter.dtype());
  auto type = getOpenCLKernelCastType(iter.dtype());
  auto stream = at::opencl::getCurrentOpenCLStream(self_->device().index());
  auto kernel_name = "cast_s";
  c10::optional<OpenCLCastFunctor> kernel_opt = c10::opencl::opencl_kernel_func<OpenCLCastFunctor>(kernel_name, {*stream.stream(), self_->numel(), 1});
  TORCH_INTERNAL_ASSERT(kernel_opt.has_value(), "Kernel not found \"", kernel_name, "\"");
  auto kernel = kernel_opt.value();
  AT_OPENCL_CHECK(kernel(*toBuffer(scalar_tensor_->data()), *toBuffer(self_->data()), type, type));
  C10_OPENCL_CHECK(syncOpenCLPointer(self_->data()));
}

REGISTER_DISPATCH(fill_stub, &fill_kernel_opencl);

} // namespace native
} // namespace at
