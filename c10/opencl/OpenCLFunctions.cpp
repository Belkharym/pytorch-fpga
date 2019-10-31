#include "OpenCLFunctions.h"
#include "OpenCLException.h"

#include <vector>
#include <deque>
#include <map>
#include <mutex>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <regex>
#ifdef _WIN32
#include <Windows.h>
#else
#include <dirent.h>
#endif // _WIN32

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

// This is taken directly from the CLEW library, but since we only
// need to check for the availability of the OpenCL drivers, it is
// the bare minimum we need
// https://github.com/martijnberger/clew/blob/master/src/clew.c
#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #define VC_EXTRALEAN
    #include <windows.h>

    typedef HMODULE             OPENCL_DYNLIB_HANDLE;

    #define OPENCL_DYNLIB_OPEN    LoadLibrary
    #define OPENCL_DYNLIB_CLOSE   FreeLibrary
    #define OPENCL_DYNLIB_IMPORT  GetProcAddress
#else // _WIN32
    #include <dlfcn.h>
    
    typedef void*                   OPENCL_DYNLIB_HANDLE;

    #define OPENCL_DYNLIB_OPEN(path)  dlopen(path, RTLD_NOW | RTLD_GLOBAL)
    #define OPENCL_DYNLIB_CLOSE       dlclose
    #define OPENCL_DYNLIB_IMPORT      dlsym
#endif // _WIN32

using namespace std::placeholders;

static constexpr char kPathSeparator =
#ifdef _WIN32
        '\\';
#else
        '/';
#endif

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

    try {
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
                        entries.push_back(dirPath + kPathSeparator + fdata.cFileName);
                    }
                }
            }
            while (FindNextFile(hFind, &fdata) != 0);
        } else {
            // Can't open directory
            TORCH_WARN("Can't open directory ", dirPath);
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
    }
    catch (...) { // no throw
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
        } catch(std::exception& ex) {
            TORCH_WARN(ex.what());
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
        TORCH_WARN("OpenCL Error : cannot create OpenCL Program (", clErrorString(*cl_err), ") {dir:\"", kernel_dir_path,"\"; content.size:", contents.size(), "}");
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
        TORCH_WARN("OpenCL Error : cannot get device property of device #0 (", clErrorString(*cl_err), ")"); \
        return; \
    } \
    if ((device_fp_config & min_fp_config) == min_fp_config) { \
        build_params << " -D" #FP "_PRECISION"; \
    }

    CHECK_FP_CONFIG(DOUBLE, CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM);
    std::string extensions = devices[0].getInfo<CL_DEVICE_EXTENSIONS>(cl_err);
    if (std::regex_match(extensions, std::regex{"cl_khr_fp16"})) {
        CHECK_FP_CONFIG(HALF, CL_FP_ROUND_TO_INF | CL_FP_INF_NAN);
        if ((device_fp_config & min_fp_config) != min_fp_config) {
            CHECK_FP_CONFIG(HALF, CL_FP_ROUND_TO_ZERO | CL_FP_INF_NAN);
        }
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

    if (!checkSystemHasOpenCL()) {
        *cl_err = CL_INVALID_PLATFORM;
        TORCH_WARN("Missing OpenCL drivers. (", clErrorString(*cl_err), ")");
        return;
    }

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
    set_device(device_id);

    context = cl::Context(devices, NULL, NULL, NULL, cl_err);
    if (*cl_err != CL_SUCCESS) {
        TORCH_WARN("Cannot create OpenCL context. (", clErrorString(*cl_err),")");
        return;
    }

    initOpenCLKernels(cl_err);
    if (*cl_err != CL_SUCCESS) {
      TORCH_WARN("Cannot initialize OpenCL kernels. (", clErrorString(*cl_err), ")");
    }
}

} // namespace ::<unnamed>

DeviceIndex device_count(cl_int* err) noexcept {
    int count;
    cl_int cl_err = CL_SUCCESS;
    // Lazy initialization of the global OpenCL context.
    std::call_once(init_flag, initOpenCLContext, &cl_err);
    if (err) *err = cl_err;
    if (cl_err != CL_SUCCESS) {
        TORCH_WARN("OpenCL Error : Could not init the OpenCL Context (", clErrorString(cl_err), ")");
        return static_cast<DeviceIndex>(0);
    }

    count = devices.size();

    return static_cast<DeviceIndex>(count);
}

DeviceIndex current_device() noexcept {
    return current_device_;
}

void set_device(DeviceIndex device_id) {
    if (device_id >= devices.size() || device_id < 0) {
        throw std::range_error("device_id is out of range (given " + c10::to_string(device_id) + ", device count is " + c10::to_string(device_count()));
    }
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
        device_id = current_device();
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

namespace {

DeviceIndex num_devices = -1;
std::once_flag device_init_flag;
std::deque<std::once_flag> device_flags;
std::vector<openclDeviceProp> device_properties;

void initOpenCLContextVectors(cl_int* err = NULL) {
  num_devices = c10::opencl::device_count(err);
  if (num_devices <= 0) {
    return;
  }
  device_flags.resize(num_devices);
  device_properties.resize(num_devices);
}

static std::vector<std::string> stringSplit(std::string strToSplit, char delimeter)
{
    std::stringstream ss(strToSplit);
    std::string item;
    std::vector<std::string> splittedStrings;
    while (std::getline(ss, item, delimeter))
    {
       splittedStrings.push_back(item);
    }
    return splittedStrings;
}

void initDeviceProperty(DeviceIndex device_index) {
  cl_int cl_err;
  openclDeviceProp device_prop;
  auto device = c10::opencl::opencl_device(device_index);

  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_ADDRESS_BITS, &device_prop.addressBits));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_AVAILABLE, &device_prop.available));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_AVAILABLE, &device_prop.available));
  auto kernels = clRemoveNullChars(device.getInfo<CL_DEVICE_BUILT_IN_KERNELS>(&cl_err));
  C10_OPENCL_CHECK(cl_err);
  device_prop.builtInKernels = stringSplit(kernels, ';');
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_COMPILER_AVAILABLE, &device_prop.compilerAvailable));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_DOUBLE_FP_CONFIG, &device_prop.doubleFpConfig));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_ENDIAN_LITTLE, &device_prop.endianLittle));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_ERROR_CORRECTION_SUPPORT, &device_prop.errorCorrectionSupport));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_EXECUTION_CAPABILITIES, &device_prop.executionCapabilities));
  auto extensions = clRemoveNullChars(device.getInfo<CL_DEVICE_EXTENSIONS>(&cl_err));
  C10_OPENCL_CHECK(cl_err);
  device_prop.extensions = stringSplit(extensions, ' ');
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &device_prop.globalMemCacheSize));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, &device_prop.globalMemCacheType));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, &device_prop.globalMemCachelineSize));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &device_prop.globalMemSize));
  device_prop.halfFpConfig = 0;
  if (std::any_of(device_prop.extensions.cbegin(), device_prop.extensions.cend(), [](const std::string& s) {return s.compare("cl_khr_fp16") == 0;})) {
    C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_HALF_FP_CONFIG, &device_prop.halfFpConfig));
  }
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_HOST_UNIFIED_MEMORY, &device_prop.hostUnifiedMemory));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_IMAGE_SUPPORT, &device_prop.imageSupport));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT, &device_prop.image2dMaxHeight));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_IMAGE2D_MAX_WIDTH, &device_prop.image2dMaxWidth));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_IMAGE3D_MAX_DEPTH, &device_prop.image3dMaxDepth));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_IMAGE3D_MAX_HEIGHT, &device_prop.image3dMaxHeight));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_IMAGE3D_MAX_WIDTH, &device_prop.image3dMaxWidth));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, &device_prop.imageMaxBufferSize));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, &device_prop.imageMaxArraySize));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_LINKER_AVAILABLE, &device_prop.linkerAvailable));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &device_prop.localMemSize));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_LOCAL_MEM_TYPE, &device_prop.localMemType));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &device_prop.maxClockFrequency));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &device_prop.maxComputeUnits));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_CONSTANT_ARGS, &device_prop.maxConstantArgs));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, &device_prop.maxConstantBufferSize));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &device_prop.maxMemAllocSize));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_PARAMETER_SIZE, &device_prop.maxParameterSize));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_READ_IMAGE_ARGS, &device_prop.maxReadImageArgs));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_SAMPLERS, &device_prop.maxSamplers));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &device_prop.maxWorkGroupSize));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &device_prop.maxWorkItemDimensions));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &device_prop.maxWorkItemSizes));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_WRITE_IMAGE_ARGS, &device_prop.maxWriteImageArgs));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_MEM_BASE_ADDR_ALIGN, &device_prop.memBaseAddrAlign));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, &device_prop.minDataTypeAlignSize));
  device_prop.name = clRemoveNullChars(device.getInfo<CL_DEVICE_NAME>(&cl_err));
  C10_OPENCL_CHECK(cl_err);
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, &device_prop.nativeVectorWidthChar));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, &device_prop.nativeVectorWidthShort));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, &device_prop.nativeVectorWidthInt));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, &device_prop.nativeVectorWidthLong));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, &device_prop.nativeVectorWidthFloat));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, &device_prop.nativeVectorWidthDouble));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, &device_prop.nativeVectorWidthHalf));
  device_prop.openclCVersion = clRemoveNullChars(device.getInfo<CL_DEVICE_OPENCL_C_VERSION>(&cl_err));
  C10_OPENCL_CHECK(cl_err);
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_PARENT_DEVICE, &device_prop.parentDevice));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_PARTITION_MAX_SUB_DEVICES, &device_prop.partitionMaxSubDevices));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_PARTITION_PROPERTIES, &device_prop.partitionProperties));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_PARTITION_AFFINITY_DOMAIN, &device_prop.partitionAffinityDomain));
  size_t nTypes;
  C10_OPENCL_CHECK(clGetDeviceInfo(device(), CL_DEVICE_PARTITION_TYPE, 0, NULL, &nTypes));
  device_prop.partitionType.resize(nTypes);
  C10_OPENCL_CHECK(clGetDeviceInfo(device(), CL_DEVICE_PARTITION_TYPE, device_prop.partitionType.size(), device_prop.partitionType.data(), NULL));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_PLATFORM, &device_prop.platform));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, &device_prop.preferredVectorWidthChar));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, &device_prop.preferredVectorWidthShort));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, &device_prop.preferredVectorWidthInt));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, &device_prop.preferredVectorWidthLong));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, &device_prop.preferredVectorWidthFloat));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, &device_prop.preferredVectorWidthDouble));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, &device_prop.preferredVectorWidthHalf));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_PRINTF_BUFFER_SIZE, &device_prop.printfBufferSize));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, &device_prop.preferredInteropUserSync));
  device_prop.profile = clRemoveNullChars(device.getInfo<CL_DEVICE_PROFILE>(&cl_err));
  C10_OPENCL_CHECK(cl_err);
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_PROFILING_TIMER_RESOLUTION, &device_prop.profilingTimerResolution));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_QUEUE_PROPERTIES, &device_prop.queueProperties));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_REFERENCE_COUNT, &device_prop.referenceCount));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_SINGLE_FP_CONFIG, &device_prop.singleFpConfig));
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_TYPE, &device_prop.type));
  device_prop.vendor = clRemoveNullChars(device.getInfo<CL_DEVICE_VENDOR>(&cl_err));
  C10_OPENCL_CHECK(cl_err);
  C10_OPENCL_CHECK(device.getInfo(CL_DEVICE_VENDOR_ID, &device_prop.vendorId));
  device_prop.version = clRemoveNullChars(device.getInfo<CL_DEVICE_VERSION>(&cl_err));
  C10_OPENCL_CHECK(cl_err);
  device_prop.driverVersion = clRemoveNullChars(device.getInfo<CL_DRIVER_VERSION>(&cl_err));
  C10_OPENCL_CHECK(cl_err);

  device_properties[device_index] = device_prop;
}

} // anonymous namespace

openclDeviceProp* getCurrentDeviceProperties(cl_int* err) {
  auto device = c10::opencl::current_device();
  return getDeviceProperties(device, err);
}

openclDeviceProp* getDeviceProperties(int64_t device, cl_int* err) {
  if (err) *err = CL_SUCCESS;
  std::call_once(device_init_flag, initOpenCLContextVectors, err);
  if (err != CL_SUCCESS) {
    return nullptr;
  }
  if (device == -1) device = c10::opencl::current_device();
  if (device < 0 || device >= num_devices) {
    if (err) *err = CL_INVALID_DEVICE;
    return nullptr;
  }
  std::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

// This is taken directly from the CLEW library, but since we only
// need to check for the availability of the OpenCL drivers, it is
// the bare minimum we need
// https://github.com/martijnberger/clew/blob/master/src/clew.c
static OPENCL_DYNLIB_HANDLE dynamic_library_open_find(const char **paths) {
  return NULL;
}

// This is taken directly from the CLEW library, but since we only
// need to check for the availability of the OpenCL drivers, it is
// the bare minimum we need.
// https://github.com/martijnberger/clew/blob/master/src/clew.c
// Based on the clewInit() function.
bool checkSystemHasOpenCL() noexcept {
#ifdef _WIN32
    static constexpr const char *paths[] = {"OpenCL.dll", NULL};
#elif defined(__APPLE__)
    static constexpr const char *paths[] = {"/Library/Frameworks/OpenCL.framework/OpenCL", NULL};
#else
    static constexpr const char *paths[] = {"libOpenCL.so",
                           "libOpenCL.so.0",
                           "libOpenCL.so.1",
                           "libOpenCL.so.2",
                           NULL};
#endif

    try {
        // Try to load library
        int i = 0;
        while (paths[i] != NULL) {
            OPENCL_DYNLIB_HANDLE lib = OPENCL_DYNLIB_OPEN(paths[i]);
            if (lib != NULL) {
                OPENCL_DYNLIB_CLOSE(lib);
                return true;
            }
            ++i;
        }
    }
    catch(...) { // no throw
        // Ignore errors.
    }
    return false;
}

}} // namespace c10::opencl
