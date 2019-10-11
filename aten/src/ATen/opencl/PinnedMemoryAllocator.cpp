#include <ATen/opencl/PinnedMemoryAllocator.h>
#include <ATen/Context.h>
#include <ATen/Config.h>

#include <stdexcept>

namespace at { namespace opencl {

at::Allocator* getPinnedMemoryAllocator() {
  return GetAllocator(at::kOPENCL);
}

}} // namespace at::opencl
