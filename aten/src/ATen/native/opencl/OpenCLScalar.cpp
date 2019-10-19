#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <ATen/opencl/OpenCLContext.h>
#include <c10/opencl/OpenCLStream.h>
#include <ATen/native/opencl/Utils.h>

namespace at {
namespace native {

Scalar _local_scalar_dense_opencl(const Tensor& self) {
  Scalar r;
  AT_DISPATCH_ALL_TYPES_AND3(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, self.scalar_type(), "_local_scalar_dense_opencl", [&] {
      scalar_t value;
      at::opencl::OpenCLStream stream = at::opencl::getCurrentOpenCLStream(self.device().index());
      AT_OPENCL_CHECK(syncOpenCLPointer(self.storage().data()));
      stream.synchronize(); // Pointless syncronization, since we read blocking-ly.
      value = ((scalar_t*)self.storage().data())[0];
      r = Scalar(value);
    });
  return r;
}

}} // at::native
