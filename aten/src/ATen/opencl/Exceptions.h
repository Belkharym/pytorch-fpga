#pragma once

#include <ATen/Context.h>
#include <c10/util/Exception.h>
#include <c10/opencl/OpenCLException.h>

// See Note [CHECK macro]
#define AT_OPENCL_CHECK(EXPR) C10_OPENCL_CHECK(EXPR)

/* TODO If we need to check for the OpenCL drivers, implement it here */

// // For CUDA Driver API
// //
// // This is here instead of in c10 because NVRTC is loaded dynamically via a stub
// // in ATen, and we need to use its nvrtcGetErrorString.
// // See NOTE [ USE OF NVRTC AND DRIVER API ].
// #ifndef __HIP_PLATFORM_HCC__

// #define AT_CUDA_DRIVER_CHECK(EXPR)                                                                               \
//   do {                                                                                                           \
//     CUresult __err = EXPR;                                                                                       \
//     if (__err != CUDA_SUCCESS) {                                                                                 \
//       const char* err_str;                                                                                       \
//       CUresult get_error_str_err C10_UNUSED = at::globalContext().getNVRTC().cuGetErrorString(__err, &err_str);  \
//       if (get_error_str_err != CUDA_SUCCESS) {                                                                   \
//         AT_ERROR("CUDA driver error: unknown error");                                                            \
//       } else {                                                                                                   \
//         AT_ERROR("CUDA driver error: ", err_str);                                                                \
//       }                                                                                                          \
//     }                                                                                                            \
//   } while (0)

// #else

// #define AT_CUDA_DRIVER_CHECK(EXPR)                                                \
//   do {                                                                            \
//     CUresult __err = EXPR;                                                        \
//     if (__err != CUDA_SUCCESS) {                                                  \
//       AT_ERROR("CUDA driver error: ", static_cast<int>(__err));                   \
//     }                                                                             \
//   } while (0)

// #endif
