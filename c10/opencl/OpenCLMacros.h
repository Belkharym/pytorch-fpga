#pragma once

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#define C10_COMPILE_TIME_MAX_OPENCL_DEVICES 16

namespace c10 {
namespace opencl {

using CommandQueue_t = cl::CommandQueue*;

} // namespace opencl
} // namespace c10
