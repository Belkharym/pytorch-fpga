#pragma once

#include <ATen/core/ATenGeneral.h>
#include <ATen/Context.h>
#include <c10/opencl/OpenCLStream.h>
#include <c10/opencl/OpenCLFunctions.h>
#include <ATen/opencl/Exceptions.h>

#include <cstdint>

namespace at {
namespace opencl {

/*
A common OpenCL interface for ATen.

This interface is distinct from OpenCLHooks, which defines an interface that links
to both CPU-only and OpenCL builds. That interface is intended for runtime
dispatch and should be used from files that are included in both CPU-only and
OpenCL builds.

OpenCLContext, on the other hand, should be preferred by files only included in
OpenCL builds. It is intended to expose OpenCL functionality in a consistent
manner.

This means there is some overlap between the OpenCLContext and OpenCLHooks, but
the choice of which to use is simple: use OpenCLContext when in an OpenCL-only file,
use OpenCLHooks otherwise.

Note that OpenCLContext simply defines an interface with no associated class.
It is expected that the modules whose functions compose this interface will
manage their own state. There is only a single OpenCL context/state.
*/

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

/**
 * OpenCL is available if we compiled with OpenCL, and there are one or more
 * devices.  If we compiled with OpenCL but there is a driver problem, etc.,
 * this function will report OpenCL is not available (rather than raise an error.)
 */
inline bool is_available() {
    return c10::opencl::device_count() > 0;
}

CAFFE2_API std::string clRemoveNullChars(const std::string &str);

CAFFE2_API openclDeviceProp* getCurrentDeviceProperties();

CAFFE2_API openclDeviceProp* getDeviceProperties(int64_t device);

CAFFE2_API Allocator* getOpenCLDeviceAllocator();

} // namespace opencl
} // namespace at
