#include "OpenCLFunctions.h"
#include "OpenCLException.h"

#include <vector>
#include <map>
#include <algorithm>
#include <mutex>
#include <string>
#include <sstream>
#include <fstream>
#include <dirent.h>

#include <c10/util/typeid.h>

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

static constexpr char kPathSeparator =
#ifdef _WIN32
        '\\';
#else
        '/';
#endif

namespace caffe2 {
// This is required to allow Storage class to handle cl::Buffers in its data pointer.
CAFFE_KNOWN_TYPE(cl::Buffer);
} // namespace caffe2

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

static std::vector<std::string> listDirectory(const std::string& dirPath) {
    std::vector<std::string> entries;
    DIR *dir;
    struct dirent *ent;
    // Open directory
    if ((dir = opendir(dirPath.c_str())) != NULL) {
        // Fetch every entry name.
        while ((ent = readdir (dir)) != NULL) {
            if (ent->d_type != DT_DIR) {
                entries.emplace_back(dirPath + kPathSeparator + ent->d_name);
            }
        }
        closedir (dir);
    }

    return entries;
}

static void initOpenCLKernels(cl_int* cl_err) {
    static const std::string kernels_dir{_STRINGIFY(_OPENCL_KERNEL_DIR)};
    
    // TODO Fetch files and put them all in a cl::Program::Sources object
    // to then build them (if on FPGA, fetch the binaries and put them
    // in a cl::Program::Binaries) and fetch all the kernel names using
    // cl::Program::createKernels . Then use getInfo on the cl::Kernel to
    // get the name:
    //   cl_int cl_err = kernel.getInfo<char*>(
    //                       CL_KERNEL_FUNCTION_NAME, &kernel_name);
    std::string kernel_dir_path{kernels_dir};
    std::vector<std::string> files = listDirectory(kernel_dir_path);
    if (files.size() == 0) {
        TORCH_WARN_ONCE("OpenCL Error : the kernel directory path \"", kernels_dir, "\" is not a valide path (no files found).");
        if (cl_err) {
            *cl_err = CL_INVALID_KERNEL_NAME;
        }
        return;
    }
#ifdef C10_USE_FPGA
    using contentVector_t = cl::Program::Binaries;
#else
    using contentVector_t = cl::Program::Sources;
#endif // C10_USE_FPGA
    using fileContent_t = contentVector_t::value_type;
    contentVector_t contents;
    std::transform(files.cbegin(), files.cend(), std::back_inserter(contents), [] (const std::string& path) {
        try {
            std::ifstream stream{path};
            if (!stream.fail()) {
                std::string content{std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
                fileContent_t fileContent{content.begin(), content.end()};
                return fileContent;
            }
        } catch(std::exception& ptr) {
            TORCH_WARN(ptr.what());
        }
        return fileContent_t{};
    });
    contentVector_t tmp;
    std::copy_if(contents.cbegin(), contents.cend(), std::back_inserter(tmp), [&](const contentVector_t::value_type& p) {
        return p.size() > 0;
    });
    std::swap(contents, tmp);

#ifdef C10_USE_FPGA
    // Only get the first binary, and apply it to all devices
    cl::Program::Binaries binaries(devices.size(), contents[0]);
    std::vector<cl_int> binaryStatus;
    program = cl::Program{context, devices, binaries, &binaryStatus, cl_err};
#else
    program = cl::Program{context, contents, cl_err};
#endif // C10_USE_FPGA
    if (*cl_err != CL_SUCCESS) {
        TORCH_WARN_ONCE("OpenCL Error : cannot create OpenCL Program (", clErrorString(*cl_err), ")");
#ifdef C10_USE_FPGA
        TORCH_WARN("Device status:");
        for (size_t i = 0; i < binaryStatus.size(); ++i) {
            TORCH_WARN("  Device #", i, " [", devices[i].getInfo<CL_DEVICE_NAME>(), "]: ", clErrorString(binaryStatus[i]));
        }
#endif // C10_USE_FPGA
        return;
    }

#ifndef C10_USE_FPGA
    *cl_err = program.build(devices);
    if (*cl_err != CL_SUCCESS) {
        TORCH_WARN_ONCE("OpenCL Error : cannot build OpenCL Program (", clErrorString(*cl_err), ")");
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
#endif // !C10_USE_FPGA

    std::vector<cl::Kernel> kernels_;
    *cl_err = program.createKernels(&kernels_);
    if (*cl_err != CL_SUCCESS) {
        TORCH_WARN_ONCE("OpenCL Error : cannot fetch OpenCL kernels (", clErrorString(*cl_err), ")");
        return;
    }

    std::transform(kernels_.cbegin(), kernels_.cend(), std::inserter(kernels, kernels.end()),
        [](const cl::Kernel& kernel) -> std::pair<std::string,cl::Kernel> {
            auto name = kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
            // There can be a null character left at the end of the string, which mess with the comparison in the map.
            name = clRemoveNullChars(name);
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
        TORCH_WARN_ONCE("Cannot find platform for OpenCL. (", clErrorString(*cl_err), ")");
        return;
    }
    platform = platforms[platform_id];

    devices = std::vector<cl::Device>();
    *cl_err = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (*cl_err == CL_SUCCESS && (devices.size() == 0 || device_id >= devices.size())) {
        *cl_err = CL_DEVICE_NOT_FOUND;
    }
    if (*cl_err != CL_SUCCESS) {
        TORCH_WARN_ONCE("Cannot find OpenCL compatible device. (", clErrorString(*cl_err),")");
        return;
    }
    current_device_ = device_id;

    context = cl::Context(devices, NULL, NULL, NULL, cl_err);
    if (*cl_err != CL_SUCCESS) {
        C10_OPENCL_CHECK_WARN(*cl_err);
        return;
    }

    initOpenCLKernels(cl_err);
    C10_OPENCL_CHECK_WARN(*cl_err);
}

} // namespace ::<unnamed>

DeviceIndex device_count() noexcept {
    int count;
    cl_int cl_err = CL_SUCCESS;
    // Lazy initialization of the global OpenCL context.
    std::call_once(init_flag, initOpenCLContext, &cl_err);
    if (cl_err != CL_SUCCESS) {
        TORCH_WARN("OpenCL Error : Could not init the OpenCL Context (", clErrorString(cl_err), ")");
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

cl::Platform opencl_platform() {
    return platform;
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

std::string clRemoveNullChars(const std::string &str) {
  std::string ret;
  std::copy_if(str.begin(), str.end(), std::back_inserter(ret), [](const char& c) {return !!c;});
  return ret;
}

}} // namespace c10::opencl
