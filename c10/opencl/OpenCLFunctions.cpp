#include "OpenCLFunctions.h"

#include <vector>
#include <map>
#include <algorithm>
#include <mutex>
#include <string>
#include <sstream>
#include <fstream>
#include <experimental/filesystem>


#ifdef __cpp_lib_experimental_filesystem
namespace fs {
using namespace std::experimental::filesystem::v1;
} // namespace fs
#endif // __cpp_lib_experimental_filesystem

#ifndef _STRINGIFY
// Defined in GNU header file cdefs.h
#ifndef __STRING
#define __STRING(str) #str
#endif
#define _STRINGIFY(str) __STRING(str)
#endif

#ifndef _OPENCL_KERNEL_DIR
#define _OPENCL_KERNEL_DIR .
#endif // !_OPENCL_KERNEL_DIR

namespace c10 {
namespace opencl {

namespace {
static cl::Platform platform;
static cl::Context context;
static std::vector<cl::Device> devices;
static DeviceIndex current_device_ = 0;
static cl::Program program;
static std::map<std::string, cl::Kernel> kernels;

static std::once_flag init_flag;

static void initOpenCLKernels(cl_int* cl_err) {
    static const std::string kernels_dir{_STRINGIFY(_OPENCL_KERNEL_DIR)};
    
    // TODO Fetch files and put them all in a cl::Program::Sources object
    // to then build them (if on FPGA, fetch the binaries and put them
    // in a cl::Program::Binaries) and fetch all the kernel names using
    // cl::Program::createKernels . Then use getInfo on the cl::Kernel to
    // get the name:
    //   cl_int cl_err = kernel.getInfo<char*>(
    //                       CL_KERNEL_FUNCTION_NAME, &kernel_name);
    fs::path kernel_dir_path{kernels_dir};
    if (!fs::exists(kernel_dir_path)) {
        TORCH_WARN_ONCE("OpenCL Error : the kernel directory path \"", kernels_dir,"\" is not a valide path.");
        if (cl_err) {
            *cl_err = CL_INVALID_KERNEL_NAME;
        }
        return;
    }
    std::vector<fs::path> files;
    fs::directory_iterator start(kernel_dir_path);
    fs::directory_iterator end;
    std::transform(start, end, std::back_inserter(files), [](const fs::directory_entry& entry) {return entry.path();});
    cl::Program::Sources sources;
    std::transform(files.cbegin(), files.cend(), std::back_inserter(sources), [] (const fs::path& path) {
        std::ifstream stream{path};
        std::string content{std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
        char *c_str = (char*)malloc(content.size() + 1);
        strncpy(c_str, content.c_str(), content.size() + 1);
        return std::make_pair(c_str, content.size());
    });

    program = cl::Program{context, sources, cl_err};
    if (*cl_err != CL_SUCCESS) {
        TORCH_WARN_ONCE("OpenCL Error : cannot create OpenCL Program (code=", *cl_err, ")");
        return;
    }

    *cl_err = program.build(devices);
    if (*cl_err != CL_SUCCESS) {
        TORCH_WARN_ONCE("OpenCL Error : cannot build OpenCL Program (code=", *cl_err, ")");
        if (*cl_err == CL_BUILD_PROGRAM_FAILURE) {
            for (auto& device : devices) {
                auto build_status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
                if (build_status != CL_BUILD_SUCCESS) {
                    auto device_name = device.getInfo<CL_DEVICE_NAME>();
                    auto build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                    TORCH_WARN("- Device ", device_name);
                    TORCH_WARN("Build logs: \n", build_log, "\n");
                }
            }
        }
        return;
    }

    std::vector<cl::Kernel> kernels_;
    *cl_err = program.createKernels(&kernels_);
    if (*cl_err != CL_SUCCESS) {
        TORCH_WARN_ONCE("OpenCL Error : cannot fetch OpenCL kernels (code=", *cl_err, ")");
        return;
    }

    std::transform(kernels_.cbegin(), kernels_.cend(), std::inserter(kernels, kernels.end()),
        [](const cl::Kernel& kernel) -> std::pair<std::string,cl::Kernel> {
            auto name = kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
            // There can be a null character left at the end of the string, which mess with the comparison in the map.
            name.resize(name.size() - 1);
            return std::make_pair(name, kernel);
        }
    );
}

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
    if (*cl_err != CL_SUCCESS) {
        TORCH_WARN_ONCE("Cannot initialize OpenCL context.");
        return;
    }

    initOpenCLKernels(cl_err);
    if (*cl_err != CL_SUCCESS) {
        TORCH_WARN_ONCE("Cannot initialize OpenCL kernels.");
        return;
    }
}

} // namespace ::<unnamed>

DeviceIndex device_count() noexcept {
    int count;
    cl_int cl_err = CL_SUCCESS;
    // Lazy initialization of the global OpenCL context.
    std::call_once(init_flag, initOpenCLContext, &cl_err);
    if (cl_err != CL_SUCCESS) {
        TORCH_WARN("OpenCL Error : Could not init the OpenCL Context (code=", cl_err, ")");
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

c10::optional<cl::Kernel> opencl_kernel(const std::string& kernel_func_name, cl_int *err) {
    const auto& it = kernels.find(kernel_func_name);
    if (err) {
        *err = CL_SUCCESS;
    }
    if (it == kernels.cend()) {
        if (err) {
            *err = CL_INVALID_KERNEL_NAME;
        }
        return {};
    }
    return {it->second};
}

}} // namespace c10::opencl
