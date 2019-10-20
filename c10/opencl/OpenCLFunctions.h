#pragma once

#include <string>
#include <type_traits>

#include "OpenCLMacros.h"

#include <c10/util/C++17.h>
#include <c10/util/Optional.h>
#include <c10/core/Device.h>
#include <c10/util/function_traits>


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

template<typename Func, typename... Args, typename std::enable_if<std::integral_constant<bool, (::c10::function_traits<Func>::arity == sizeof...(Args))>::value, int>::type = 0>
c10::optional<Func> opencl_kernel_func(const std::string& kernel_func_name, cl::EnqueueArgs config, cl_int *err) {
    auto kernel = opencl_kernel(kernel_func_name, err);
    if (kernel.has_value()) {
        return static_cast<Func>([kernel_func_name, config](Args&&... args) -> cl_int {
            auto kern = opencl_kernel(kernel_func_name);
            TORCH_INTERNAL_ASSERT(kern.has_value(), "An opencl kernel went missing.");
            auto functor = cl::KernelFunctor<Args...>{kern.value()};
            cl_int cl_err;
            functor(config, std::forward<Args&&>(args)..., cl_err);
            return cl_err;
        });
    }
    return {};
}

template<typename Func, typename... Args, size_t N = c10::function_traits<Func>::arity, size_t M = sizeof...(Args), typename std::enable_if<(N > M), int>::type = 0>
c10::optional<Func> opencl_kernel_func(const std::string &kernel_func_name, cl::EnqueueArgs config, cl_int* err) {
    typedef typename c10::function_traits<Func>::template argument<(N - M) - 1>::type NewArg;
    return opencl_kernel_func<Func, NewArg, Args...>(std::forward<const std::string&>(kernel_func_name), std::forward<cl::EnqueueArgs>(config), std::forward<cl_int*>(err));
}

template<typename Func, size_t N = c10::function_traits<Func>::arity, typename std::enable_if<(N > 0), int>::type = 0>
c10::optional<Func> opencl_kernel_func(const std::string &kernel_func_name, cl::EnqueueArgs config, cl_int* err = NULL) {
    typedef typename c10::function_traits<Func>::template argument<N - 1>::type NewArg;
    return opencl_kernel_func<Func, NewArg>(std::forward<const std::string&>(kernel_func_name), std::forward<cl::EnqueueArgs>(config), std::forward<cl_int*>(err));
}

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
