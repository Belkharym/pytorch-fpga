#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS 1
#define CL_HPP_CL_1_2_DEFAULT_BUILD 1
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
//#include "libopencl.h"
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <c10/macros/Export.h>
#include <c10/opencl/OpenCLMacros.h>

#define AT_OPENCL_API CAFFE2_API

namespace at {
namespace opencl {

using namespace c10::opencl;

}} // at::opencl
