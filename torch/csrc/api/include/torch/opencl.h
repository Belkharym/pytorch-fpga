#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>

namespace torch {
namespace opencl {
/// Returns the number of CUDA devices available.
size_t TORCH_API device_count();

/// Returns true if at least one CUDA device is available.
bool TORCH_API is_available();
} // namespace opencl
} // namespace torch
