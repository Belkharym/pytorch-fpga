#pragma once

#include "OpenCLMacros.h"

#include <c10/core/Device.h>
#include <c10/util/Optional.h>

namespace c10 {
namespace opencl {

DeviceIndex device_count() noexcept;
// Returns the current device.
DeviceIndex current_device();
// Sets the current device.
void set_device(DeviceIndex device_id);

// Returns the global OpenCL context.
cl::Context opencl_context();
cl::Device opencl_device(DeviceIndex device_id = -1);
c10::optional<cl::Kernel> opencl_kernel(const std::string& kernel_func_name, cl_int *err = NULL);

}} // namespace c10::opencl
