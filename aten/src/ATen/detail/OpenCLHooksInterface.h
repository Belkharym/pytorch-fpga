#pragma once

#include <c10/core/Allocator.h>
#include <ATen/core/Generator.h>
#include <c10/util/Exception.h>

#include <c10/util/Registry.h>

#include <cstddef>
#include <functional>
#include <memory>

// Forward-declares THOState
struct THOState;

namespace at {
class Context;
}

// NB: Class must live in `at` due to limitations of Registry.h.
namespace at {

// The OpenCLHooksInterface is an omnibus interface for any OpenCL functionality
// which we may want to call into from CPU code (and thus must be dynamically
// dispatched, to allow for separate compilation of OpenCL code).  See
// CUDAHooksInterface for more detailed motivation.
struct CAFFE2_API OpenCLHooksInterface {
  // This should never actually be implemented, but it is used to
  // squelch -Werror=non-virtual-dtor
  virtual ~OpenCLHooksInterface() {}

  // Initialize THHState and, transitively, the HIP state
  virtual std::unique_ptr<THOState, void (*)(THOState*)> initOpenCL() const {
    AT_ERROR("Cannot initialize OpenCL without ATen_opencl library.");
  }

  virtual std::unique_ptr<Generator> initOpenCLGenerator(Context*) const {
    AT_ERROR("Cannot initialize OpenCL generator without ATen_opencl library.");
  }

  virtual Generator* getDefaultOpenCLGenerator(DeviceIndex device_index = -1) const {
    AT_ERROR("Cannot get default OpenCL generator without ATen_opencl library.");
  }

  virtual Device getDeviceFromPtr(void* data) const {
    AT_ERROR("Cannot get device of pointer on OpenCL without ATen_opencl library.");
  }

  virtual bool isPinnedPtr(void* data) const {
    return false;
  }

  virtual bool hasOpenCL() const {
    return false;
  }

  virtual int64_t current_device() const {
    return -1;
  }

  virtual Allocator* getPinnedMemoryAllocator() const {
    AT_ERROR("Pinned memory requires OpenCL.");
  }

  virtual void registerOpenCLTypes(Context*) const {
    AT_ERROR("Cannot registerOpenCLTypes() without ATen_opencl library.");
  }
  
  virtual std::string showConfig() const {
    AT_ERROR("Cannot query detailed OpenCL version without ATen_opencl library.");
  }

  virtual int getNumDevices() const {
    return 0;
  }
};

// NB: dummy argument to suppress "ISO C++11 requires at least one argument
// for the "..." in a variadic macro"
struct CAFFE2_API OpenCLHooksArgs {};

C10_DECLARE_REGISTRY(OpenCLHooksRegistry, OpenCLHooksInterface, OpenCLHooksArgs);
#define REGISTER_OPENCL_HOOKS(clsname) \
  C10_REGISTER_CLASS(OpenCLHooksRegistry, clsname, clsname)

namespace detail {
CAFFE2_API const OpenCLHooksInterface& getOpenCLHooks();

} // namespace detail
} // namespace at
