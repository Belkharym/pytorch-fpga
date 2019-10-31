#pragma once

#include <string>
#include <type_traits>
#include <stdlib.h>

#include "OpenCLMacros.h"

#include <c10/util/C++17.h>
#include <c10/util/Optional.h>
#include <c10/core/Device.h>
#include <c10/util/function_traits>


namespace c10 {
namespace opencl {

struct openclDeviceProp {
public:
    cl_uint addressBits;
    cl_bool available;
    std::vector<std::string> builtInKernels;
    cl_bool compilerAvailable;
    cl_device_fp_config doubleFpConfig;
    cl_bool endianLittle;
    cl_bool errorCorrectionSupport;
    cl_device_exec_capabilities executionCapabilities;
    std::vector<std::string> extensions;
    cl_ulong globalMemCacheSize;
    cl_device_mem_cache_type globalMemCacheType;
    cl_uint globalMemCachelineSize;
    cl_ulong globalMemSize;
    cl_device_fp_config halfFpConfig;
    cl_bool hostUnifiedMemory;
    cl_bool imageSupport;
    size_t image2dMaxHeight;
    size_t image2dMaxWidth;
    size_t image3dMaxDepth;
    size_t image3dMaxHeight;
    size_t image3dMaxWidth;
    size_t imageMaxBufferSize;
    size_t imageMaxArraySize;
    cl_bool linkerAvailable;
    cl_ulong localMemSize;
    cl_device_local_mem_type localMemType;
    cl_uint maxClockFrequency;
    cl_uint maxComputeUnits;
    cl_uint maxConstantArgs;
    cl_ulong maxConstantBufferSize;
    cl_ulong maxMemAllocSize;
    size_t maxParameterSize;
    cl_uint maxReadImageArgs;
    cl_uint maxSamplers;
    size_t maxWorkGroupSize;
    cl_uint maxWorkItemDimensions;
    std::vector<size_t> maxWorkItemSizes;
    cl_uint maxWriteImageArgs;
    cl_uint memBaseAddrAlign;
    cl_uint minDataTypeAlignSize;
    std::string name;
    // Returns the native ISA vector width.
    // The vector width is defined as the number of
    // scalar elements that can be stored in the vector.
    cl_uint nativeVectorWidthChar;
    cl_uint nativeVectorWidthDouble;
    cl_uint nativeVectorWidthFloat;
    cl_uint nativeVectorWidthHalf;
    cl_uint nativeVectorWidthInt;
    cl_uint nativeVectorWidthLong;
    cl_uint nativeVectorWidthShort;
    std::string openclCVersion;
    cl_device_id parentDevice;
    cl_uint partitionMaxSubDevices;
    std::vector<cl_device_partition_property> partitionProperties;
    cl_device_affinity_domain partitionAffinityDomain;
    std::vector<cl_device_partition_property> partitionType;
    cl_platform_id platform;
    cl_uint preferredVectorWidthChar;
    cl_uint preferredVectorWidthDouble;
    cl_uint preferredVectorWidthFloat;
    cl_uint preferredVectorWidthHalf;
    cl_uint preferredVectorWidthInt;
    cl_uint preferredVectorWidthLong;
    cl_uint preferredVectorWidthShort;
    size_t printfBufferSize;
    cl_bool preferredInteropUserSync;
    std::string profile;
    size_t profilingTimerResolution;
    cl_command_queue_properties queueProperties;
    cl_uint referenceCount;
    cl_device_fp_config singleFpConfig;
    cl_device_type type;
    std::string vendor;
    cl_uint vendorId;
    std::string version; // Device version
    std::string driverVersion; // Driver version
};

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

//! \brief Never throws. On errors, returns 0. Errors can be retrieved from {\code err}.
C10_OPENCL_API DeviceIndex device_count(cl_int* err = NULL) noexcept;
// Returns the current device.
C10_OPENCL_API DeviceIndex current_device() noexcept;
// Sets the current device.
C10_OPENCL_API void set_device(DeviceIndex device_id);

C10_OPENCL_API openclDeviceProp* getCurrentDeviceProperties(cl_int* err = NULL);
C10_OPENCL_API openclDeviceProp* getDeviceProperties(int64_t device, cl_int* err = NULL);

// Returns the global OpenCL context.
C10_OPENCL_API cl::Platform opencl_platform();
C10_OPENCL_API cl::Context opencl_context();
C10_OPENCL_API cl::Device opencl_device(DeviceIndex device_id = -1);
C10_OPENCL_API c10::optional<cl::Kernel> opencl_kernel(const std::string& kernel_func_name, cl_int *err = NULL);

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

C10_OPENCL_API std::string clRemoveNullChars(const std::string &str);

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

C10_OPENCL_API bool checkSystemHasOpenCL() noexcept;

}} // namespace c10::opencl
