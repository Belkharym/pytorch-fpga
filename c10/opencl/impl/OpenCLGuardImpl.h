#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/core/Stream.h>
#include <c10/opencl/OpenCLMacros.h>

namespace c10 {
namespace opencl {
namespace impl {

struct C10_OPENCL_API OpenCLGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::OPENCL;

  OpenCLGuardImpl() {}
  explicit OpenCLGuardImpl(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == DeviceType::OPENCL);
  }
  DeviceType type() const override {
    return DeviceType::OPENCL;
  }
  Device exchangeDevice(Device d) const override;
  Device getDevice() const override;
  void setDevice(Device d) const override;
  void uncheckedSetDevice(Device d) const noexcept override;
  Stream getStream(Device d) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, c10::Device(DeviceType::OPENCL, -1));
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, c10::Device(DeviceType::OPENCL, -1));
  }
  DeviceIndex deviceCount() const noexcept override;

  void destroyEvent(
    void* event,
    const DeviceIndex device_index) const noexcept override;

  void record(
    void** event,
    const Stream& stream,
    const DeviceIndex device_index,
    const EventFlag flag) const override;

  void block(
    void* event,
    const Stream& stream) const override;

  // May be called from any device
  bool queryEvent(void* event) const override;
};

}}} // namespace c10::cuda::impl
