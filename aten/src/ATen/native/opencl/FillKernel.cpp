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
#include <ATen/native/opencl/Resize.h>

namespace at { namespace native {

void fill_kernel_opencl(TensorIterator& iter, Scalar value) {
  if (iter.numel() == 0) {
    return;
  }
  Tensor self = iter.tensor(0);
  TensorImpl* self_ = checked_tensor_unwrap(self, "self", 2, "fill_kernel_opencl", false, c10::Backend::OpenCL, iter.dtype());
  auto type = getOpenCLKernelCastType(iter.dtype());
  auto stream = at::opencl::getCurrentOpenCLStream(self_->device().index());

  Tensor scalar_tensor = at::native::scalar_tensor_opencl(value, self.options());
  auto scalar_tensor_ = scalar_tensor.storage().unsafeGetStorageImpl();

  static const std::string kernel_name = "cast_s";
  auto kernel = c10::opencl::opencl_kernel_func<OpenCLCastFunctor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange((size_t)self_->numel()), 1});
  AT_OPENCL_CHECK(kernel(*toBuffer(scalar_tensor_->data()), *toBuffer(self_->data()), type, type));
  AT_OPENCL_CHECK(syncOpenCLPointer(self_->data(), stream));
  stream.synchronize();
}

REGISTER_DISPATCH(fill_stub, &fill_kernel_opencl);

} // namespace native
} // namespace at
