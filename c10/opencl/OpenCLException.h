#pragma once

#include <c10/util/Exception.h>
#include <c10/macros/Macros.h>
#include <ATen/opencl/ATenOpenCLGeneral.h>

// Note [CHECK macro]
// ~~~~~~~~~~~~~~~~~~
// This is a macro so that AT_ERROR can get accurate __LINE__
// and __FILE__ information.  We could split this into a short
// macro and a function implementation if we pass along __LINE__
// and __FILE__, but no one has found this worth doing.

const char* clErrorString(cl_int error);
const char* clDeviceTypeString(cl_device_type device_type);

// For OpenCL Runtime API
#define C10_OPENCL_CHECK(EXPR, ...)                                  \
  do {                                                               \
    cl_int __err = EXPR;                                             \
    if (__err != CL_SUCCESS) {                                       \
      TORCH_CHECK(false, __FILE__, ":", __LINE__, " : OpenCL error : ", clErrorString(__err), " ", ##__VA_ARGS__); \
    }                                                                \
  } while (0)

#define C10_OPENCL_CHECK_WARN(EXPR, ...)                       \
  do {                                                         \
    cl_int __err = EXPR;                                       \
    if (__err != CL_SUCCESS) {                                 \
      TORCH_WARN(__FILE__, ":", __LINE__, " : OpenCL warning: ", clErrorString(__err), " ", ##__VA_ARGS__);\
    }                                                          \
  } while (0)
