#pragma once

#include <c10/opencl/impl/opencl_cmake_macros.h>
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

// See c10/macros/Export.h for a detailed explanation of what the function
// of these macros are.  We need one set of macros for every separate library
// we build.

#ifdef _WIN32
#if defined(C10_OPENCL_BUILD_SHARED_LIBS)
#define C10_OPENCL_EXPORT __declspec(dllexport)
#define C10_OPENCL_IMPORT __declspec(dllimport)
#else
#define C10_OPENCL_EXPORT
#define C10_OPENCL_IMPORT
#endif
#else // _WIN32
#if defined(__GNUC__)
#define C10_OPENCL_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define C10_OPENCL_EXPORT
#endif // defined(__GNUC__)
#define C10_OPENCL_IMPORT C10_OPENCL_EXPORT
#endif // _WIN32

// This one is being used by libc10_opencl.so
#ifdef C10_OPENCL_BUILD_MAIN_LIB
#define C10_OPENCL_API C10_OPENCL_EXPORT
#else
#define C10_OPENCL_API C10_OPENCL_IMPORT
#endif

#define C10_COMPILE_TIME_MAX_OPENCL_DEVICES 16

namespace c10 {
namespace opencl {

using CommandQueue_t = cl::CommandQueue*;

} // namespace opencl
} // namespace c10
