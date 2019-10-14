#include <c10/opencl/impl/OpenCLGuardImpl.h>
#include <c10/opencl/OpenCLFunctions.h>
#include <c10/opencl/OpenCLStream.h>

namespace c10 {
namespace opencl {
namespace impl {

constexpr DeviceType OpenCLGuardImpl::static_type;


Device OpenCLGuardImpl::exchangeDevice(Device d) const {
    TORCH_INTERNAL_ASSERT(d.type() == DeviceType::OPENCL);
    Device old_device = getDevice();
    if (old_device.index() != d.index()) {
        c10::opencl::set_device(d.index());
    }
    return old_device;
}
Device OpenCLGuardImpl::getDevice() const {
    int device;
    device = c10::opencl::current_device();
    TORCH_CHECK(device < c10::opencl::device_count(), "OpenCL Error: Device not found ", device, " (device count:", c10::opencl::device_count(), ")");
    return Device(DeviceType::OPENCL, device);
}
void OpenCLGuardImpl::setDevice(Device d) const {
    TORCH_INTERNAL_ASSERT(d.type() == DeviceType::OPENCL);
    c10::opencl::set_device(d.index());
}
void OpenCLGuardImpl::uncheckedSetDevice(Device d) const noexcept {
    c10::opencl::set_device(d.index());
}

DeviceIndex OpenCLGuardImpl::deviceCount() const noexcept {
    return c10::opencl::device_count();
}

void OpenCLGuardImpl::destroyEvent(
    void* event,
    const DeviceIndex device_index) const noexcept {
    if (!event) return;
    auto opencl_event = static_cast<cl::Event*>(event);

    const auto& orig_device = c10::opencl::current_device();
    c10::opencl::set_device(device_index);
    clReleaseEvent((*opencl_event)());
    c10::opencl::set_device(orig_device);
}

void OpenCLGuardImpl::record(
    void** event,
    const Stream& stream,
    const DeviceIndex device_index,
    const EventFlag flag) const {
    TORCH_CHECK(device_index == -1 || device_index == stream.device_index(),
    "Event device index ",
    device_index,
    " does not match recording stream's device index ",
    stream.device_index(),
    ".");

    cl::Event* opencl_event = static_cast<cl::Event*>(*event);
    OpenCLStream opencl_stream{stream};

    // Moves to stream's device to record
    const auto orig_device = getDevice();
    setDevice(stream.device());

    C10_OPENCL_CHECK(opencl_stream.stream()->enqueueMarkerWithWaitList(NULL, opencl_event));
    // Makes the void* point to the (possibly just allocated) CUDA event
    *event = opencl_event;

    // Resets device
    setDevice(orig_device);
}

void OpenCLGuardImpl::block(
    void* event,
    const Stream& stream) const {
    if (!event) return;
    cl::Event* opencl_event = static_cast<cl::Event*>(event);
    const auto orig_device = getDevice();
    setDevice(stream.device());
    opencl_event->wait();
    setDevice(orig_device);
}

// May be called from any device
bool OpenCLGuardImpl::queryEvent(void* event) const {
    if (!event) return true;
    cl::Event* opencl_event = static_cast<cl::Event*>(event);
    cl_int opencl_status;
    const cl_int err = clGetEventInfo((*opencl_event)(),
        CL_EVENT_COMMAND_EXECUTION_STATUS,
        sizeof(opencl_status),
        &opencl_status,
        NULL);
    if (err != CL_SUCCESS) {
        TORCH_CHECK(false, "OpenCL Error : Cannot get event status (", err, ")");
    }
    return (opencl_status == CL_COMPLETE);
}
C10_REGISTER_GUARD_IMPL(OPENCL, OpenCLGuardImpl);

}}} // namespace c10::cuda::detail
