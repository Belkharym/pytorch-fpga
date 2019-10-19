#pragma once

#include "OpenCLMacros.h"

#include <c10/core/Device.h>
#include <c10/util/Optional.h>

#include <tuple>
#include <type_traits>

namespace c10 {
namespace opencl {

namespace {

template <size_t _i, typename _Arg, typename... _Args>
struct _arg {
    typedef typename std::conditional<_i == 0, _Arg, typename _arg<_i - 1, _Args...>::type>::type type;
};

} // namespace

/// https://stackoverflow.com/questions/7943525/is-it-possible-to-figure-out-the-parameter-type-and-return-type-of-a-lambda
template <typename T>
struct function_traits
    : public function_traits<decltype(&T::operator())>
{};
// For generic types, directly use the result of the signature of its 'operator()'

template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const>
// we specialize for pointers to member function
{
    enum { arity = sizeof...(Args) };
    // arity is the number of arguments.

    typedef ReturnType result_type;

    template <size_t i>
    struct arg
    {
        typedef typename _arg<i, Args...>::type type;
        // the i-th argument is equivalent to the i-th tuple element of a tuple
        // composed of those arguments.
    };
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
struct _final_recursion {};
}

template<typename Func, typename... Args, typename = _final_recursion>
c10::optional<Func> opencl_kernel_func(const std::string& kernel_func_name, cl::EnqueueArgs config, cl_int *err = NULL) {
    auto kernel = opencl_kernel(kernel_func_name, err);
    if (kernel.has_value()) {
        auto functor = cl::KernelFunctor<Args...>{kernel.value()};
        return {[&](Args&&... args) {
            cl_int err;
            functor(config, std::forward<Args&&>(args)..., err);
            return err;
        }};
    }
    return {};
}

template<typename Func, typename... Args, typename std::enable_if<std::is_function<Func>::value && !(c10::opencl::function_traits<Func>::arity > sizeof...(Args)), void>::type>
c10::optional<Func> opencl_kernel_func(const std::string &&kernel_func_name, cl::EnqueueArgs  &&config, cl_int* &&err = NULL) {
    opencl_kernel_func<Func, Args..., _final_recursion>(std::forward<const std::string&>(kernel_func_name), std::forward<cl::EnqueueArgs>(config), std::forward<cl_int*>(err));
}

template<typename Func, typename... Args, size_t N = c10::opencl::function_traits<Func>::arity, size_t M = sizeof...(Args), typename std::enable_if<std::is_function<Func>::value && (N > M), void>::type>
c10::optional<Func> opencl_kernel_func(const std::string &&kernel_func_name, cl::EnqueueArgs  &&config, cl_int* &&err = NULL) {
    typedef typename c10::opencl::function_traits<Func>::arg<(N - M) - 1>::type NewArg;
    opencl_kernel_func<Func, NewArg, Args...>(std::forward<const std::string&>(kernel_func_name), std::forward<cl::EnqueueArgs>(config), std::forward<cl_int*>(err));
}

template<typename Func, size_t N = c10::opencl::function_traits<Func>::arity, size_t M = 0, typename std::enable_if<std::is_function<Func>::value && (N > M), void>::type>
c10::optional<Func> opencl_kernel_func(const std::string &&kernel_func_name, cl::EnqueueArgs  &&config, cl_int* &&err = NULL) {
    typedef typename c10::opencl::function_traits<Func>::arg<(N - M) - 1>::type NewArg;
    opencl_kernel_func<Func, NewArg>(std::forward<const std::string&>(kernel_func_name), std::forward<cl::EnqueueArgs>(config), std::forward<cl_int*>(err));
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
