#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <ATen/opencl/OpenCLContext.h>
#include <c10/opencl/OpenCLStream.h>

namespace at {
namespace native {

Scalar _local_scalar_dense_opencl(const Tensor& self) {
  Scalar r;
  if (self.scalar_type() == ScalarType::Int) {
    using scalar_t = int32_t;
    scalar_t value;
    opencl::OpenCLStream stream = at::opencl::getCurrentOpenCLStream(self.device().index());
    AT_OPENCL_CHECK(stream.stream()->enqueueReadBuffer(*(cl::Buffer*)self.data_ptr(), CL_TRUE, 0, sizeof(scalar_t), &value));
    r = Scalar(value);
  }
  else {
    AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, self.scalar_type(), "_local_scalar_dense_opencl", [&] {
        scalar_t value;
        opencl::OpenCLStream stream = at::opencl::getCurrentOpenCLStream(self.device().index());
        AT_OPENCL_CHECK(stream.stream()->enqueueReadBuffer(*(cl::Buffer*)self.data_ptr(), CL_TRUE, 0, sizeof(scalar_t), &value));
        stream.synchronize(); // Pointless syncronization, since we read blocking-ly.
        r = Scalar(value);
      });
  }
  return r;
}

}} // at::native
