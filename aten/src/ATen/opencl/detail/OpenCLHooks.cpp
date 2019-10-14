#include <ATen/opencl/detail/OpenCLHooks.h>

#include <ATen/OpenCLGenerator.h>
#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/DynamicLibrary.h>
#include <ATen/opencl/OpenCLDevice.h>
#include <ATen/opencl/Exceptions.h>
#include <ATen/opencl/PinnedMemoryAllocator.h>
#include <ATen/detail/OpenCLHooksInterface.h>
#include <c10/util/Exception.h>
#include <ATen/opencl/OpenCLContext.h>


#include <sstream>
#include <cstddef>
#include <functional>
#include <memory>

namespace at {
namespace opencl {
namespace detail {

// NB: deleter is dynamic, because we need it to live in a separate
// compilation unit (alt is to have another method in hooks, but
// let's not if we don't need to!)
std::unique_ptr<THOState, void (*)(THOState*)> OpenCLHooks::initOpenCL() const {
  C10_LOG_API_USAGE_ONCE("aten.init.opencl");
  // Calling device_count() launchs the caffe2 initialization of OpenCL.
  c10::opencl::device_count();
  return std::unique_ptr<THOState, void (*)(THOState*)>(
      nullptr, [](THOState* p) {
        // no-op
      });
}

Generator* OpenCLHooks::getDefaultOpenCLGenerator(DeviceIndex device_index) const {
  return at::opencl::detail::getDefaultOpenCLGenerator(device_index);
}

Device OpenCLHooks::getDeviceFromPtr(void* data) const {
  return at::opencl::getDeviceFromPtr(data);
}

bool OpenCLHooks::isPinnedPtr(void* data) const {
  // First check if driver is broken/missing, in which case PyTorch CPU
  // functionalities should still work, we should report `false` here.
  if (!OpenCLHooks::hasOpenCL()) {
    return false;
  }

  return OpenCLCachingHostAllocator_isPinnedPtr(data);
}

bool OpenCLHooks::hasOpenCL() const {
  return at::opencl::is_available();
}

int64_t OpenCLHooks::current_device() const {
  return c10::opencl::current_device();
}

Allocator* OpenCLHooks::getPinnedMemoryAllocator() const {
  return at::opencl::getPinnedMemoryAllocator();
}

std::string OpenCLHooks::showConfig() const {
  std::ostringstream oss;

  cl::Platform cl_platform = c10::opencl::opencl_platform();
  cl_int err;
  std::string version = cl_platform.getInfo<CL_PLATFORM_VERSION>(&err);
  AT_OPENCL_CHECK(err);
  oss << version;
  oss << "\n";

  return oss.str();
}

int OpenCLHooks::getNumDevices() const {
  return c10::opencl::device_count();
}

// Sigh, the registry doesn't support namespaces :(
using at::OpenCLHooksRegistry;
using at::RegistererOpenCLHooksRegistry;

REGISTER_OPENCL_HOOKS(OpenCLHooks);

} // namespace detail
} // namespace opencl
} // namespace at
