#include <torch/opencl.h>

#include <ATen/Context.h>

#include <cstddef>

namespace torch {
namespace opencl {
size_t device_count() {
  return at::detail::getOpenCLHooks().getNumDevices();
}

bool is_available() {
  // NB: the semantics of this are different from at::globalContext().hasOpenCL();
  // ATen's function tells you if you have a working driver and OpenCL build,
  // whereas this function also tells you if you actually have any Devices.
  // This function matches the semantics of at::opencl::is_available()
  return opencl::device_count() > 0;
}
} // namespace opencl
} // namespace torch
