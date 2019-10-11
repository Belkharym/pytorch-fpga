#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/opencl/OpenCLGuard.h>
#include <c10/opencl/OpenCLStream.h>
#include <ATen/opencl/OpenCLContext.h>

#include <vector>

namespace at { namespace opencl {

// TODO: Implement this generically in c10.  You'll need some way to get
// the number of Devices from the GuardImpl, in that case.
class OpenCLMultiStreamGuard final {
public:
  /// Calls `set_stream` on each of the streams in the list.
  /// This may be useful if you need to set different streams
  /// for different devices.
  explicit OpenCLMultiStreamGuard(ArrayRef<OpenCLStream> streams) : OpenCLMultiStreamGuard() {
    for (const auto& s : streams) {
      setCurrentOpenCLStream(s);
    }
  }

  OpenCLMultiStreamGuard() {
    const size_t device_count = opencl::device_count();
    original_streams_.reserve(device_count);
    for (size_t device = 0; device < device_count; ++device) {
      original_streams_.push_back(getCurrentOpenCLStream(device));
    }
  }

  OpenCLMultiStreamGuard(const OpenCLGuard&) = delete;
  OpenCLMultiStreamGuard& operator=(const OpenCLGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OpenCLMultiStreamGuard(OpenCLGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OpenCLMultiStreamGuard& operator=(OpenCLGuard&& other) = delete;

  ArrayRef<OpenCLStream> original_streams() const {
    return original_streams_;
  }

  /// Resets the OpenCL stream on each device to the one that was active upon
  /// construction.
  ~OpenCLMultiStreamGuard() {
    for (const auto& s : original_streams_) {
      setCurrentOpenCLStream(s);
    }
  }

private:
  /// The original streams that were active on all devices.
  std::vector<OpenCLStream> original_streams_;
};

}} // namespace at::opencl
