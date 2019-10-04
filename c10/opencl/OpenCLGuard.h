#pragma once

#include <c10/opencl/impl/OpenCLGuardImpl.h>
#include <c10/opencl/OpenCLMacros.h>
#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include <cstddef>

namespace c10 { namespace opencl {

// This code is kind of boilerplatey.  See Note [Whither the DeviceGuard boilerplate]

/// A variant of DeviceGuard that is specialized for OpenCL.  It accepts
/// integer indices (interpreting them as OpenCL devices) and is a little
/// more efficient than DeviceGuard.
struct OpenCLGuard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit OpenCLGuard() = delete;

  /// Set the current OpenCL device to the passed device index.
  explicit OpenCLGuard(DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current OpenCL device to the passed device.  Errors if the passed
  /// device is not a OpenCL device.
  explicit OpenCLGuard(Device device) : guard_(device) {}

  // Copy is not allowed
  OpenCLGuard(const OpenCLGuard&) = delete;
  OpenCLGuard& operator=(const OpenCLGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  OpenCLGuard(OpenCLGuard&& other) = delete;
  OpenCLGuard& operator=(OpenCLGuard&& other) = delete;

  /// Sets the OpenCL device to the given device.  Errors if the given device
  /// is not a OpenCL device.
  void set_device(Device device) { guard_.set_device(device); }

  /// Sets the OpenCL device to the given device.  Errors if the given device
  /// is not a OpenCL device.  (This method is provided for uniformity with
  /// DeviceGuard).
  void reset_device(Device device) { guard_.reset_device(device); }

  /// Sets the OpenCL device to the given device index.
  void set_index(DeviceIndex device_index) { guard_.set_index(device_index); }

  /// Returns the device that was set upon construction of the guard
  Device original_device() const { return guard_.original_device(); }

  /// Returns the last device that was set via `set_device`, if any, otherwise the
  /// device passed during construction.
  Device current_device() const { return guard_.current_device(); }

 private:
  /// The guard for the current device.
  c10::impl::InlineDeviceGuard<impl::OpenCLGuardImpl> guard_;
};

/// A variant of OptionalDeviceGuard that is specialized for OpenCL.  See
/// OpenCLGuard for when you can use this.
struct OptionalOpenCLGuard {
  /// Create an uninitialized OptionalOpenCLGuard.
  explicit OptionalOpenCLGuard() : guard_() {}

  /// Set the current OpenCL device to the passed Device, if it is not nullopt.
  explicit OptionalOpenCLGuard(optional<Device> device_opt) : guard_(device_opt) {}

  /// Set the current OpenCL device to the passed device index, if it is not
  /// nullopt
  explicit OptionalOpenCLGuard(optional<DeviceIndex> device_index_opt) : guard_(device_index_opt) {}

  // Copy is not allowed
  OptionalOpenCLGuard(const OptionalOpenCLGuard&) = delete;
  OptionalOpenCLGuard& operator=(const OptionalOpenCLGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalOpenCLGuard(OptionalOpenCLGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalOpenCLGuard& operator=(OptionalOpenCLGuard&& other) = delete;

  /// Sets the OpenCL device to the given device, initializing the guard if it
  /// is not already initialized.  Errors if the given device is not a OpenCL device.
  void set_device(Device device) { guard_.set_device(device); }

  /// Sets the OpenCL device to the given device, initializing the guard if it is
  /// not already initialized.  Errors if the given device is not a OpenCL device.
  /// (This method is provided for uniformity with OptionalDeviceGuard).
  void reset_device(Device device) { guard_.reset_device(device); }

  /// Sets the OpenCL device to the given device index, initializing the guard if
  /// it is not already initialized.
  void set_index(DeviceIndex device_index) { guard_.set_index(device_index); }

  /// Returns the device that was set immediately prior to initialization of the
  /// guard, or nullopt if the guard is uninitialized.
  optional<Device> original_device() const { return guard_.original_device(); }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  optional<Device> current_device() const { return guard_.current_device(); }

  /// Restore the original OpenCL device, resetting this guard to uninitialized state.
  void reset() { guard_.reset(); }

private:
  c10::impl::InlineOptionalDeviceGuard<impl::OpenCLGuardImpl> guard_;
};

/// A variant of StreamGuard that is specialized for OpenCL.  See OpenCLGuard
/// for when you can use this.
struct OpenCLStreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit OpenCLStreamGuard() = delete;

  /// Set the current OpenCL device to the device associated with the passed stream,
  /// and set the current OpenCL stream on that device to the passed stream.
  /// Errors if the Stream is not a OpenCL stream.
  explicit OpenCLStreamGuard(Stream stream) : guard_(stream) {}

  /// Copy is disallowed
  OpenCLStreamGuard(const OpenCLStreamGuard&) = delete;
  OpenCLStreamGuard& operator=(const OpenCLStreamGuard&) = delete;

  /// Move is disallowed, as OpenCLStreamGuard does not have an uninitialized state,
  /// which is required for moves on types with nontrivial destructors.
  OpenCLStreamGuard(OpenCLStreamGuard&& other) = delete;
  OpenCLStreamGuard& operator=(OpenCLStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Errors if the stream passed is not a OpenCL stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.
  void reset_stream(Stream stream) { guard_.reset_stream(stream); }

  /// Returns the OpenCL stream that was set at the time the guard was constructed.
  OpenCLStream original_stream() const {
    return OpenCLStream(OpenCLStream::UNCHECKED, guard_.original_stream());
  }

  /// Returns the most recent OpenCL stream that was set using this device guard,
  /// either from construction, or via set_stream.
  OpenCLStream current_stream() const {
    return OpenCLStream(OpenCLStream::UNCHECKED, guard_.current_stream());
  }

  /// Returns the most recent OpenCL device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  Device current_device() const { return guard_.current_device(); }

  /// Returns the OpenCL device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  Device original_device() const { return guard_.original_device(); }

private:
  c10::impl::InlineStreamGuard<impl::OpenCLGuardImpl> guard_;
};

/// A variant of OptionalStreamGuard that is specialized for OpenCL.  See OpenCLGuard
/// for when you can use this.
struct OptionalOpenCLStreamGuard {
  /// Create an uninitialized guard.
  explicit OptionalOpenCLStreamGuard() : guard_() {}

  /// Set the current OpenCL device to the device associated with the passed stream,
  /// and set the current OpenCL stream on that device to the passed stream.
  /// Errors if the Stream is not a OpenCL stream.
  explicit OptionalOpenCLStreamGuard(Stream stream) : guard_(stream) {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream,
  /// if the passed stream is not nullopt.
  explicit OptionalOpenCLStreamGuard(optional<Stream> stream_opt) : guard_(stream_opt) {}

  /// Copy is disallowed
  OptionalOpenCLStreamGuard(const OptionalOpenCLStreamGuard&) = delete;
  OptionalOpenCLStreamGuard& operator=(const OptionalOpenCLStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalOpenCLStreamGuard(OptionalOpenCLStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalOpenCLStreamGuard& operator=(OptionalOpenCLStreamGuard&& other) = delete;

  /// Resets the currently set OpenCL stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Initializes the guard if it was not previously initialized.
  void reset_stream(Stream stream) { guard_.reset_stream(stream); }

  /// Returns the OpenCL stream that was set at the time the guard was most recently
  /// initialized, or nullopt if the guard is uninitialized.
  optional<OpenCLStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return make_optional(OpenCLStream(OpenCLStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  /// Returns the most recent OpenCL stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  optional<OpenCLStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return make_optional(OpenCLStream(OpenCLStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  /// Restore the original OpenCL device and stream, resetting this guard to uninitialized state.
  void reset() { guard_.reset(); }

private:
  c10::impl::InlineOptionalStreamGuard<impl::OpenCLGuardImpl> guard_;
};

} // namespace opencl
} // namespace c10
