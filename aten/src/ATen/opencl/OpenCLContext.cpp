#include <ATen/opencl/OpenCLContext.h>
#include <c10/opencl/OpenCLCachingAllocator.h>

#include <mutex>
#include <deque>
#include <vector>
#include <algorithm>

namespace at { namespace opencl {

namespace {

DeviceIndex num_devices = -1;
std::once_flag init_flag;
std::deque<std::once_flag> device_flags;
std::vector<openclDeviceProp> device_properties;

void initOpenCLContextVectors() {
  num_devices = c10::opencl::device_count();
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

  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_ADDRESS_BITS, &device_prop.addressBits));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_AVAILABLE, &device_prop.available));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_AVAILABLE, &device_prop.available));
  auto kernels = clRemoveNullChars(device.getInfo<CL_DEVICE_BUILT_IN_KERNELS>(&cl_err));
  AT_OPENCL_CHECK(cl_err);
  device_prop.builtInKernels = stringSplit(kernels, ';');
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_COMPILER_AVAILABLE, &device_prop.compilerAvailable));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_DOUBLE_FP_CONFIG, &device_prop.doubleFpConfig));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_ENDIAN_LITTLE, &device_prop.endianLittle));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_ERROR_CORRECTION_SUPPORT, &device_prop.errorCorrectionSupport));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_EXECUTION_CAPABILITIES, &device_prop.executionCapabilities));
  auto extensions = clRemoveNullChars(device.getInfo<CL_DEVICE_EXTENSIONS>(&cl_err));
  AT_OPENCL_CHECK(cl_err);
  device_prop.extensions = stringSplit(extensions, ' ');
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &device_prop.globalMemCacheSize));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, &device_prop.globalMemCacheType));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, &device_prop.globalMemCachelineSize));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &device_prop.globalMemSize));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_HALF_FP_CONFIG, &device_prop.halfFpConfig));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_HOST_UNIFIED_MEMORY, &device_prop.hostUnifiedMemory));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_IMAGE_SUPPORT, &device_prop.imageSupport));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT, &device_prop.image2dMaxHeight));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_IMAGE2D_MAX_WIDTH, &device_prop.image2dMaxWidth));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_IMAGE3D_MAX_DEPTH, &device_prop.image3dMaxDepth));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_IMAGE3D_MAX_HEIGHT, &device_prop.image3dMaxHeight));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_IMAGE3D_MAX_WIDTH, &device_prop.image3dMaxWidth));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, &device_prop.imageMaxBufferSize));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, &device_prop.imageMaxArraySize));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_LINKER_AVAILABLE, &device_prop.linkerAvailable));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &device_prop.localMemSize));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_LOCAL_MEM_TYPE, &device_prop.localMemType));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &device_prop.maxClockFrequency));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &device_prop.maxComputeUnits));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_CONSTANT_ARGS, &device_prop.maxConstantArgs));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, &device_prop.maxConstantBufferSize));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &device_prop.maxMemAllocSize));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_PARAMETER_SIZE, &device_prop.maxParameterSize));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_READ_IMAGE_ARGS, &device_prop.maxReadImageArgs));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_SAMPLERS, &device_prop.maxSamplers));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &device_prop.maxWorkGroupSize));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &device_prop.maxWorkItemDimensions));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &device_prop.maxWorkItemSizes));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_MAX_WRITE_IMAGE_ARGS, &device_prop.maxWriteImageArgs));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_MEM_BASE_ADDR_ALIGN, &device_prop.memBaseAddrAlign));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, &device_prop.minDataTypeAlignSize));
  device_prop.name = clRemoveNullChars(device.getInfo<CL_DEVICE_NAME>(&cl_err));
  AT_OPENCL_CHECK(cl_err);
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, &device_prop.nativeVectorWidthChar));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, &device_prop.nativeVectorWidthShort));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, &device_prop.nativeVectorWidthInt));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, &device_prop.nativeVectorWidthLong));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, &device_prop.nativeVectorWidthFloat));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, &device_prop.nativeVectorWidthDouble));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, &device_prop.nativeVectorWidthHalf));
  device_prop.openclCVersion = clRemoveNullChars(device.getInfo<CL_DEVICE_OPENCL_C_VERSION>(&cl_err));
  AT_OPENCL_CHECK(cl_err);
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_PARENT_DEVICE, &device_prop.parentDevice));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_PARTITION_MAX_SUB_DEVICES, &device_prop.partitionMaxSubDevices));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_PARTITION_PROPERTIES, &device_prop.partitionProperties));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_PARTITION_AFFINITY_DOMAIN, &device_prop.partitionAffinityDomain));
  size_t nTypes;
  AT_OPENCL_CHECK(clGetDeviceInfo(device(), CL_DEVICE_PARTITION_TYPE, 0, NULL, &nTypes));
  device_prop.partitionType.resize(nTypes);
  AT_OPENCL_CHECK(clGetDeviceInfo(device(), CL_DEVICE_PARTITION_TYPE, device_prop.partitionType.size(), device_prop.partitionType.data(), NULL));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_PLATFORM, &device_prop.platform));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, &device_prop.preferredVectorWidthChar));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, &device_prop.preferredVectorWidthShort));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, &device_prop.preferredVectorWidthInt));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, &device_prop.preferredVectorWidthLong));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, &device_prop.preferredVectorWidthFloat));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, &device_prop.preferredVectorWidthDouble));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, &device_prop.preferredVectorWidthHalf));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_PRINTF_BUFFER_SIZE, &device_prop.printfBufferSize));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, &device_prop.preferredInteropUserSync));
  device_prop.profile = clRemoveNullChars(device.getInfo<CL_DEVICE_PROFILE>(&cl_err));
  AT_OPENCL_CHECK(cl_err);
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_PROFILING_TIMER_RESOLUTION, &device_prop.profilingTimerResolution));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_QUEUE_PROPERTIES, &device_prop.queueProperties));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_REFERENCE_COUNT, &device_prop.referenceCount));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_SINGLE_FP_CONFIG, &device_prop.singleFpConfig));
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_TYPE, &device_prop.type));
  device_prop.vendor = clRemoveNullChars(device.getInfo<CL_DEVICE_VENDOR>(&cl_err));
  AT_OPENCL_CHECK(cl_err);
  AT_OPENCL_CHECK(device.getInfo(CL_DEVICE_VENDOR_ID, &device_prop.vendorId));
  device_prop.version = clRemoveNullChars(device.getInfo<CL_DEVICE_VERSION>(&cl_err));
  AT_OPENCL_CHECK(cl_err);
  device_prop.driverVersion = clRemoveNullChars(device.getInfo<CL_DRIVER_VERSION>(&cl_err));
  AT_OPENCL_CHECK(cl_err);

  device_properties[device_index] = device_prop;
}

} // anonymous namespace

std::string clRemoveNullChars(const std::string &str) {
  std::string ret;
  std::copy_if(str.begin(), str.end(), std::back_inserter(ret), [](const char& c) {return !!c;});
  return ret;
}

openclDeviceProp* getCurrentDeviceProperties() {
  auto device = c10::opencl::current_device();
  return getDeviceProperties(device);
}

openclDeviceProp* getDeviceProperties(int64_t device) {
  std::call_once(init_flag, initOpenCLContextVectors);
  if (device == -1) device = c10::opencl::current_device();
  AT_ASSERT(device >= 0 && device < num_devices);
  std::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

Allocator* getOpenCLDeviceAllocator() {
  return c10::opencl::OpenCLCachingAllocator::get();
}

} // namespace opencl

} // namespace at
