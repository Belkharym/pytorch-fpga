#include <ATen/opencl/OpenCLContext.h>
#include <c10/opencl/OpenCLFunctions.h>
#include <c10/opencl/OpenCLCachingAllocator.h>

namespace at { namespace opencl {

Allocator* getOpenCLDeviceAllocator() {
  return at::GetAllocator(DeviceType::OPENCL);
}

} // namespace opencl

} // namespace at
