#include <ATen/detail/OpenCLHooksInterface.h>

#include <ATen/Generator.h>
#include <c10/util/Optional.h>

// TODO: No need to have this whole header, we can just put it all in
// the cpp file

namespace at { namespace opencl { namespace detail {

// The real implementation of OpenCLHooksInterface
struct OpenCLHooks : public at::OpenCLHooksInterface {
  OpenCLHooks(at::OpenCLHooksArgs) {}
  std::unique_ptr<THOState, void(*)(THOState*)> initOpenCL() const override;
  Device getDeviceFromPtr(void* data) const override;
  bool isPinnedPtr(void* data) const override;
  Generator* getDefaultOpenCLGenerator(DeviceIndex device_index = -1) const override;
  bool hasOpenCL() const override;
  int64_t current_device() const override;
  Allocator* getPinnedMemoryAllocator() const override;
  std::string showConfig() const override;
  int getNumDevices() const override;
};

}}} // at::cuda::detail
