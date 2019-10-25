#include "OpenCLFunctions.h"
#include "OpenCLException.h"

#include <vector>
#include <map>
#include <algorithm>
#include <mutex>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#ifdef _WIN32
#include <Windows.h>
#else
#include <dirent.h>
#endif // _WIN32

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

using namespace std::placeholders;

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

static bool endsWith(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
      0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static std::vector<std::string> listDirectory(std::string dirPath) {
    std::vector<std::string> entries;
#ifdef C10_USE_FPGA
            static const std::vector<std::string> ext = {".xclbin",".awsxclbin"};
#else
            static const std::vector<std::string> ext = {".cl"};
#endif // C10_USE_FPGA

#ifndef _WIN32
    DIR *dir;
    struct dirent *ent;
    // Open directory
    if ((dir = opendir(dirPath.c_str())) != NULL) {
        // Fetch every entry name.
        while ((ent = readdir (dir)) != NULL) {
            if (ent->d_type != DT_DIR && std::any_of(ext.begin(), ext.end(), std::bind(endsWith, ent->d_name, _1))) {
                entries.emplace_back(dirPath + kPathSeparator + ent->d_name);
            }
        }
        closedir (dir);
    }
#else
    //open a directory the WIN32 way
    HANDLE hFind = INVALID_HANDLE_VALUE;
    WIN32_FIND_DATA fdata;
 
    if(dirPath[dirPath.size()-1] == '\\' || dirPath[dirPath.size()-1] == '/') {
        dirPath = dirPath.substr(0,dirPath.size()-1);
    }
 
    hFind = FindFirstFile(std::string(dirPath).append("\\*").c_str(), &fdata); 
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (strncmp(fdata.cFileName, ".", sizeof(fdata.cFileName)) != 0 &&
                strncmp(fdata.cFileName, "..", sizeof(fdata.cFileName)) != 0) {
                if (fdata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY || !std::any_of(ext.begin(), ext.end(), std::bind(endsWith, fdata.cFileName, _1))) {
                    continue; // a diretory
                }
                else {
                    entries.push_back(fdata.cFileName);
                }
            }
        }
        while (FindNextFile(hFind, &fdata) != 0);
    } else {
        // Can't open directory
        return entries;
    }
 
    if (GetLastError() != ERROR_NO_MORE_FILES) {
        FindClose(hFind);
        TORCH_WARN("some error with opening directory: ", GetLastError());
        return entries;
    }
 
    FindClose(hFind);
    hFind = INVALID_HANDLE_VALUE;
#endif // _WIN32

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
        TORCH_WARN("OpenCL Error : the kernel directory path \"", kernels_dir, "\" is not a valide path (no files found).");
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
        TORCH_WARN("OpenCL Error : cannot create OpenCL Program (", clErrorString(*cl_err), ")");
#ifdef C10_USE_FPGA
        TORCH_WARN("Device status:");
        for (size_t i = 0; i < binaryStatus.size(); ++i) {
            TORCH_WARN("  Device #", i, " [", devices[i].getInfo<CL_DEVICE_NAME>(), "]: ", clErrorString(binaryStatus[i]));
        }
#endif // C10_USE_FPGA
        return;
    }

    std::stringstream build_params;
    cl_bitfield min_fp_config;
    cl_bitfield device_fp_config;

#define CHECK_FP_CONFIG(FP, MIN_FP_CONFIG) \
    min_fp_config = MIN_FP_CONFIG; \
    device_fp_config = devices[0].getInfo<CL_DEVICE_##FP##_FP_CONFIG>(cl_err); \
    if (*cl_err != CL_SUCCESS) { \
        TORCH_WARN("OpenCL Error : cannot get device property of device #0"); \
        return; \
    } \
    if ((device_fp_config & min_fp_config) == min_fp_config) { \
        build_params << " -D" #FP "_PRECISION"; \
    }

    CHECK_FP_CONFIG(DOUBLE, CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM);
    CHECK_FP_CONFIG(HALF, CL_FP_ROUND_TO_INF | CL_FP_INF_NAN);
    if ((device_fp_config & min_fp_config) != min_fp_config) {
        CHECK_FP_CONFIG(HALF, CL_FP_ROUND_TO_ZERO | CL_FP_INF_NAN);
    }
    CHECK_FP_CONFIG(SINGLE, CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN);
#undef CHECK_FP_CONFIG

#ifndef C10_USE_FPGA
    *cl_err = program.build(devices, std::string{"-I" + kernels_dir + build_params.str()}.c_str());
    if (*cl_err != CL_SUCCESS) {
        TORCH_WARN("OpenCL Error : cannot build OpenCL Program (", clErrorString(*cl_err), ")");
        if (*cl_err == CL_BUILD_PROGRAM_FAILURE) {
            for (auto& device : devices) {
                auto build_status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
                if (build_status != CL_BUILD_SUCCESS) {
                    auto device_name = device.getInfo<CL_DEVICE_NAME>();
                    auto device_type = device.getInfo<CL_DEVICE_TYPE>();
                    auto build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                    TORCH_WARN("- Device [", clDeviceTypeString(device_type), "] ", device_name);
                    TORCH_WARN("  Build logs: \n", build_log, "\n");
                }
            }
        }
        return;
    }
#endif // !C10_USE_FPGA

    std::vector<cl::Kernel> kernels_;
    *cl_err = program.createKernels(&kernels_);
    if (*cl_err != CL_SUCCESS) {
        TORCH_WARN("OpenCL Error : cannot fetch OpenCL kernels (", clErrorString(*cl_err), ")");
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
        TORCH_WARN("Cannot find platform for OpenCL. (", clErrorString(*cl_err), ")");
        return;
    }
    platform = platforms[platform_id];

    devices = std::vector<cl::Device>();
    *cl_err = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (*cl_err == CL_SUCCESS && (devices.size() == 0 || device_id >= devices.size())) {
        *cl_err = CL_DEVICE_NOT_FOUND;
    }
    if (*cl_err != CL_SUCCESS) {
        TORCH_WARN("Cannot find OpenCL compatible device. (", clErrorString(*cl_err),")");
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
