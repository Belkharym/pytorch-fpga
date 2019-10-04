#include "OpenCLFunctions.h"

#include <vector>
#include <mutex>


namespace c10 {
namespace opencl {

namespace {
static cl::Platform platform;
static cl::Context context;
static std::vector<cl::Device> devices;
static DeviceIndex current_device_ = 0;

static std::once_flag init_flag;

static void initOpenCLContext(cl_int* cl_err) {
    const auto platform_id = 0;
    const auto device_id = 0;

    auto platforms = std::vector<cl::Platform>();
    *cl_err = cl::Platform::get(&platforms);
    if (*cl_err == CL_SUCCESS && (platforms.size() == 0 || platform_id >= platforms.size())) {
        *cl_err = CL_INVALID_PLATFORM;
    }
    if (*cl_err != CL_SUCCESS) {
        TORCH_WARN_ONCE("Cannot find platform for OpenCL.");
        return;
    }
    platform = platforms[platform_id];

    devices = std::vector<cl::Device>();
    *cl_err = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (*cl_err == CL_SUCCESS && (devices.size() == 0 || device_id >= devices.size())) {
        *cl_err = CL_DEVICE_NOT_FOUND;
    }
    if (*cl_err != CL_SUCCESS) {
        TORCH_WARN_ONCE("Cannot find OpenCL compatible device.");
        return;
    }
    current_device_ = device_id;

    context = cl::Context(devices, NULL, NULL, NULL, cl_err);
}

} // namespace ::<unnamed>

DeviceIndex device_count() noexcept {
    int count;
    cl_int cl_err;
    // Lazy initialization of the global OpenCL context.
    std::call_once(init_flag, initOpenCLContext, &cl_err);
    if (cl_err != CL_SUCCESS) {
        return static_cast<DeviceIndex>(0);
    }

    count = devices.size();

    return static_cast<DeviceIndex>(count);
}

DeviceIndex current_device() {
    return current_device_;
}

void set_device(DeviceIndex device_id) {
    // TODO Check if the device id is valid
    current_device_ = device_id;
}

cl::Context opencl_context() {
    return context;
}

cl::Device opencl_device(DeviceIndex device_id) {
    if (device_id == -1) {
        device_id = current_device_;
    }
    return devices[device_id];
}

}} // namespace c10::opencl
