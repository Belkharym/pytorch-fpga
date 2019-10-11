#pragma once

#include <ATen/opencl/Exceptions.h>
#include <c10/opencl/OpenCLFunctions.h>

namespace at {
namespace opencl {

inline Device getDeviceFromPtr(void* ptr) {
  // OpenCL Buffers are not assigned to a specific device.
  return {DeviceType::OPENCL, static_cast<int16_t>(c10::opencl::current_device())};
}

}} // namespace at::cuda
