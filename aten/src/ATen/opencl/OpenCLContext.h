#pragma once

#include <ATen/core/ATenGeneral.h>
#include <ATen/Context.h>
#include <c10/opencl/OpenCLStream.h>
#include <c10/opencl/OpenCLFunctions.h>
#include <ATen/opencl/Exceptions.h>

#include <cstdint>

namespace at {
namespace opencl {

/*
A common OpenCL interface for ATen.

This interface is distinct from OpenCLHooks, which defines an interface that links
to both CPU-only and OpenCL builds. That interface is intended for runtime
dispatch and should be used from files that are included in both CPU-only and
OpenCL builds.

OpenCLContext, on the other hand, should be preferred by files only included in
OpenCL builds. It is intended to expose OpenCL functionality in a consistent
manner.

This means there is some overlap between the OpenCLContext and OpenCLHooks, but
the choice of which to use is simple: use OpenCLContext when in an OpenCL-only file,
use OpenCLHooks otherwise.

Note that OpenCLContext simply defines an interface with no associated class.
It is expected that the modules whose functions compose this interface will
manage their own state. There is only a single OpenCL context/state.
*/

/**
 * OpenCL is available if we compiled with OpenCL, and there are one or more
 * devices.  If we compiled with OpenCL but there is a driver problem, etc.,
 * this function will report OpenCL is not available (rather than raise an error.)
 */
inline bool is_available() {
    return c10::opencl::device_count() > 0;
}

CAFFE2_API Allocator* getOpenCLDeviceAllocator();

} // namespace opencl
} // namespace at
