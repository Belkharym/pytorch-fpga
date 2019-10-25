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

template<typename Func>
struct is_std_function : std::false_type {};

template<typename Func>
struct is_std_function<std::function<Func>> : std::is_function<Func> {};

template<typename Func>
struct unwrap_function {
    typedef typename std::enable_if<std::is_function<Func>::value, Func>::type type;
};

template<typename Func>
struct unwrap_function<std::function<Func>> {
    typedef typename std::enable_if<std::is_function<Func>::value, Func>::type type;
};

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

namespace {

/**
 * @brief   Returns a functor which calls an OpenCL kernel with the given
 *          parameters.
 * 
 * This function is similar to {@code opencl_kernel} in the sense where it
 * fetches a kernel with the given name {@code kernel_func_name}, but instead
 * of returning a {@code cl::Kernel} instance, it returns a functor which can
 * be called to run the kernel with with the given function signature.
 */
template<typename Func, typename... Args, typename std::enable_if<std::integral_constant<bool, (::c10::function_traits<Func>::arity == sizeof...(Args))>::value && std::is_function<Func>::value, int>::type = 0>
std::function<typename unwrap_function<Func>::type> opencl_kernel_func(const std::string& kernel_func_name, cl::EnqueueArgs config, cl::Event *event) {
    return static_cast<std::function<typename unwrap_function<Func>::type>>(
        [kernel_func_name, config, event](Args&&... args) -> cl_int {
            cl_int cl_err;
            auto kern = opencl_kernel(kernel_func_name, &cl_err);
            if (kern.has_value()) {
                auto functor = cl::KernelFunctor<Args...>{kern.value()};
                cl::Event ev= functor(config, std::forward<Args&&>(args)..., cl_err);
                if (event) {
                    *event = ev;
                }
            }
            return cl_err;
        }
    );
}

template<typename Func, typename... Args, size_t N = sizeof...(Args), typename std::enable_if<(c10::function_traits<Func>::arity > N) && std::is_function<Func>::value, int>::type = 0>
std::function<typename unwrap_function<Func>::type> opencl_kernel_func(const std::string &kernel_func_name, cl::EnqueueArgs config, cl::Event* err) {
    typedef typename c10::function_traits<Func>::template argument<N>::type NewArg;
    return opencl_kernel_func<typename unwrap_function<Func>::type, Args..., NewArg>(std::forward<const std::string&>(kernel_func_name), std::forward<cl::EnqueueArgs>(config), std::forward<cl::Event*>(err));
}

} // namespace

template<typename Func, typename std::enable_if<(c10::function_traits<Func>::arity > 0) && std::is_function<Func>::value, int>::type = 0>
std::function<typename unwrap_function<Func>::type> opencl_kernel_func(const std::string &kernel_func_name, cl::EnqueueArgs config, cl::Event* err = NULL) {
    typedef typename c10::function_traits<Func>::template argument<0>::type NewArg;
    return opencl_kernel_func<typename unwrap_function<Func>::type, NewArg>(std::forward<const std::string&>(kernel_func_name), std::forward<cl::EnqueueArgs>(config), std::forward<cl::Event*>(err));
}

template<typename Func, typename std::enable_if<(c10::function_traits<Func>::arity == 0) && std::is_function<Func>::value, int>::type = 0>
std::function<typename unwrap_function<Func>::type> opencl_kernel_func(const std::string &kernel_func_name, cl::EnqueueArgs config, cl::Event* err = NULL) {
    return opencl_kernel_func<typename unwrap_function<Func>::type>(std::forward<const std::string&>(kernel_func_name), std::forward<cl::EnqueueArgs>(config), std::forward<cl::Event*>(err));
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

template<typename... Args>
C10_DEPRECATED_MESSAGE("runKernel is deprecated in favor to using the opencl_kernel_func.")
cl_int runKernel(cl::Kernel &kernel, const cl::EnqueueArgs &config, Args&& ...args) {
    cl::KernelFunctor<Args...> functor{kernel};
    cl_int err;
    functor(config, std::forward<Args>(args)..., err);
    return err;
}

}} // namespace c10::opencl
