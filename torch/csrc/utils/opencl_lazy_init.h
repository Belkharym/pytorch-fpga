#pragma once

#include <c10/core/TensorOptions.h>

// opencl_lazy_init() is always compiled, even for CPU-only builds.
// Thus, it does not live in the cuda/ folder.

namespace torch {
namespace utils {

// The INVARIANT is that this function MUST be called before you attempt
// to get a OpenCL Type object from ATen, in any way.  Here are some common
// ways that a Type object may be retrieved:
//
//    - You call getNonVariableType or getNonVariableTypeOpt
//    - You call toBackend() on a Type
//
// It's important to do this correctly, because if you forget to add it
// you'll get an oblique error message about "Cannot initialize OpenCL without
// ATen_opencl library" if you try to use OpenCL functionality from a CPU-only
// build, which is not good UX.
//
void opencl_lazy_init();
void opencl_set_run_yet_variable_to_false();

static void maybe_initialize_opencl(const at::TensorOptions& options) {
  if (options.device().is_opencl()) {
    torch::utils::opencl_lazy_init();
  }
}

}
}
