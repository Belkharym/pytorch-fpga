#pragma once

#include "OpenCLMacros.h"

#include <c10/core/Device.h>
#include <c10/util/Optional.h>

namespace c10 {
namespace opencl {

DeviceIndex device_count() noexcept;
// Returns the current device.
DeviceIndex current_device();
// Sets the current device.
void set_device(DeviceIndex device_id);

// Returns the global OpenCL context.
cl::Platform opencl_platform();
cl::Context opencl_context();
cl::Device opencl_device(DeviceIndex device_id = -1);
c10::optional<cl::Kernel> opencl_kernel(const std::string& kernel_func_name, cl_int *err = NULL);

C10_API std::string clRemoveNullChars(const std::string &str);

template<size_t idx, typename Arg>
cl_int setArgs(cl::Kernel &kernel, Arg&& arg) {
    return kernel.setArg<Arg>(idx, arg);
}

template<size_t idx, typename Arg, typename... Args>
cl_int setArgs(cl::Kernel &kernel, Arg&& arg, Args&& ...args) {
    cl_int err = kernel.setArg<Arg>(idx, arg);
    if (err == CL_SUCCESS) {
        return setArgs<idx + 1>(kernel, std::forward<Args>(args)...);
    }
    return err;
}

struct KernelExecConfig {
    const cl::NDRange global;
    const cl::NDRange offset;
    const cl::NDRange local;
    const std::vector<cl::Event> *events;
    cl::Event* event;
    KernelExecConfig(const cl::NDRange global,
                     const cl::NDRange offset = 0,
                     const cl::NDRange local = 1,
                     const std::vector<cl::Event> *events = NULL,
                     cl::Event* event = NULL)
                     : global(global)
                     , offset(offset)
                     , local(local)
                     , events(events)
                     , event(event) {}
};

template<typename... Args>
cl_int runKernel(cl::Kernel &kernel, const cl::EnqueueArgs &config, Args&& ...args) {
    cl::KernelFunctor<Args...> functor{kernel};
    cl_int err;
    functor(config, std::forward<Args>(args)..., err);
    return err;
}

}} // namespace c10::opencl
