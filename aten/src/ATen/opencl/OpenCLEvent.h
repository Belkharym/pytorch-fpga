#pragma once

#include <ATen/opencl/ATenOpenCLGeneral.h>
#include <ATen/opencl/OpenCLContext.h>
#include <c10/opencl/OpenCLStream.h>
#include <c10/opencl/OpenCLGuard.h>
#include <ATen/opencl/Exceptions.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <utility>

namespace at { namespace opencl {

/*
* OpenCLEvents are movable not copyable wrappers around OpenCL's events.
*
* OpenCLEvents are constructed lazily when first recorded unless it is
* reconstructed from a cl_event. The event has a device, and this
* device is acquired from the first recording stream. However, if reconstructed
* from a handle, the device should be explicitly specified; or if ipc_handle() is
* called before the event is ever recorded, it will use the current device.
* Later streams that record the event must match this device.
*/
struct AT_OPENCL_API OpenCLEvent {
  // Constructors
  // Default value for `flags` is specified below - it's cudaEventDisableTiming
  OpenCLEvent() {}
  OpenCLEvent(unsigned int flags) : flags_{flags} {}

  // Note: event destruction done on creating device to avoid creating a
  // OpenCL context on other devices.
  ~OpenCLEvent() {}

  OpenCLEvent(const OpenCLEvent&) = delete;
  OpenCLEvent& operator=(const OpenCLEvent&) = delete;

  OpenCLEvent(OpenCLEvent&& other) { moveHelper(std::move(other)); }
  OpenCLEvent& operator=(OpenCLEvent&& other) {
    moveHelper(std::move(other));
    return *this;
  }

  operator cl::Event() { return *event(); }

  // Less than operator (to allow use in sets)
  friend bool operator<(const OpenCLEvent& left, const OpenCLEvent& right) {
    return &left.event_ < &right.event_;
  }

  optional<at::Device> device() const {
    if (is_created_) {
      return at::Device(at::kOPENCL, device_index_);
    } else {
      return {};
    }
  }

  bool isCreated() const { return is_created_; }
  DeviceIndex device_index() const {return device_index_;}
  cl::Event* event() { return &event_; }

  // Note: openclEventQuery can be safely called from any device
  bool query() const {
    if (!is_created_) {
      return true;
    }

    cl_int err;
    cl_int status = event_.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>(&err);
    if ((err == CL_SUCCESS && status == CL_COMPLETE) || err != CL_INVALID_EVENT) {
      return true;
    } else if (err != CL_INVALID_EVENT) {
      C10_OPENCL_CHECK(err);
    }

    return false;
  }

  void record() { record(c10::opencl::getCurrentOpenCLStream()); }

  void recordOnce(const OpenCLStream& stream) {
    if (!was_recorded_) record(stream);
  }

  // Note: cudaEventRecord must be called on the same device as the event.
  void record(const OpenCLStream& stream) {
    device_index_ = stream.device_index();
    OpenCLGuard guard(device_index_);
    AT_OPENCL_CHECK(stream.stream()->enqueueMarkerWithWaitList(NULL, &event_));
    is_created_ = true;
    was_recorded_ = true;
  }

  // Note: cudaStreamWaitEvent must be called on the same device as the stream.
  // The event has no actual GPU resources associated with it.
  void block(const OpenCLStream& stream) {
    if (is_created_) {
      OpenCLGuard guard(stream.device_index());
      std::vector<cl::Event> waitList;
      waitList.push_back(this->event_);
      AT_OPENCL_CHECK(stream.stream()->enqueueMarkerWithWaitList(&waitList));
    }
  }

  // Note: cudaEventElapsedTime can be safely called from any device
  float elapsed_time(const OpenCLEvent& other) const {
    TORCH_CHECK(is_created_ && other.isCreated(),
      "Both events must be recorded before calculating elapsed time.");
    float time_ms = 0;
    // TODO Find a way to enable CL_QUEUE_PROFILING_ENABLE on the command queue before hand.
    AT_OPENCL_CHECK(-64); // Informal CL_UNKNOWN_ERROR
    return time_ms;
  }

  // Note: cudaEventSynchronize can be safely called from any device
  void synchronize() const {
    if (is_created_) {
      AT_OPENCL_CHECK(event_.wait());
    }
  }

private:
  unsigned int flags_ = CL_QUEUE_PROFILING_ENABLE;
  bool is_created_ = false;
  bool was_recorded_ = false;
  DeviceIndex device_index_ = -1;
  cl::Event event_;

  void moveHelper(OpenCLEvent&& other) {
    std::swap(flags_, other.flags_);
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }
};

} // namespace cuda
} // namespace at
