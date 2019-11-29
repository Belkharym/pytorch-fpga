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
#include <ATen/Utils.h>

namespace at { namespace native {

void fill_kernel_opencl(TensorIterator& iter, Scalar value) {
  if (iter.numel() == 0) {
    return;
  }

  Tensor self = iter.tensor(0);
  auto type = getOpenCLKernelCastType(iter.common_dtype());
  auto stream = at::opencl::getCurrentOpenCLStream(self.device().index());

  at::Tensor scalar_buffer;
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, iter.common_dtype(), "fill_opencl", [&]() {
    scalar_buffer = at::native::scalar_buffer_opencl<scalar_t>(value, self.device().index());
  });

  static const std::string kernel_name = "cast_s";
  auto kernel = c10::opencl::opencl_kernel_func<OpenCLCastFunctor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange{(size_t)self.storage_offset()}, cl::NDRange((size_t)self.numel()), 1});
  AT_OPENCL_CHECK(kernel(*toBuffer(scalar_buffer.data_ptr()), *toBuffer(self.data_ptr()), type, type));
  AT_OPENCL_CHECK(syncOpenCLPointer(self.data_ptr(), stream));
  stream.synchronize();
}

REGISTER_DISPATCH(fill_stub, &fill_kernel_opencl);

} // namespace native
} // namespace at
