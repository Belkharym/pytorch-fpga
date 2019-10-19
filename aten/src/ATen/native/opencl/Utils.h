#include <c10/core/ScalarType.h>
#include <caffe2/opencl/context.h>
#include <ATen/opencl/PinnedMemoryAllocator.h>
#include <ATen/native/opencl/OpenCLOperations.h>

#include <string>

namespace at {
namespace native {

inline ScalarType getIntEquivalentOfFloat(const ScalarType type) {
  switch (type)
  {
  case ScalarType::Float:  // passthrough
  case ScalarType::QInt32: // passthrough
  case ScalarType::Int:
    return ScalarType::Int;
    break;
  case ScalarType::Double: // passthrough
  case ScalarType::Long:
    return ScalarType::Long;
    break;
  case ScalarType::Half:   // passthrough
  case ScalarType::BFloat16: // passthrough
  case ScalarType::Short:
    return ScalarType::Short;
    break;
  case ScalarType::QInt8:  // passthrough
  case ScalarType::QUInt8: // passthrough
  case ScalarType::Byte:
    return ScalarType::Byte;
    break;
  case ScalarType::Bool:
    return ScalarType::Bool;
    break;
  
  default:
    return ScalarType::Undefined;
    break;
  }
}

inline std::string getOpenCLKernelTypeSuffix(const ScalarType type) {
  switch (type)
  {
    case ScalarType::Bool:
      return "c";
      break;
    case ScalarType::Byte:
      return "c";
      break;
    case ScalarType::Char:
      return "c";
      break;
    case ScalarType::Short:
      return "s";
      break;
    case ScalarType::Int:
      return "i";
      break;
    case ScalarType::Long:
      return "l";
      break;
    case ScalarType::Half:
      return "h";
      break;
    case ScalarType::Float:
      return "f";
      break;
    case ScalarType::Double:
      return "d";
      break;
    case ScalarType::QInt32:
      return "i";
      break;
    case ScalarType::QInt8:
      return "c";
      break;
    case ScalarType::QUInt8:
      return "c";
      break;

    default:
      return "u"; // For unknown
      break;
  }
}

inline at::native::opencl::OpenCLCastType getOpenCLKernelCastType(const ScalarType type) {
  switch (type)
  {
    case ScalarType::Bool:
      return at::native::opencl::OpenCLCastType::CHAR;
      break;
    case ScalarType::Byte:
      return at::native::opencl::OpenCLCastType::CHAR;
      break;
    case ScalarType::Char:
      return at::native::opencl::OpenCLCastType::CHAR;
      break;
    case ScalarType::Short:
      return at::native::opencl::OpenCLCastType::SHORT;
      break;
    case ScalarType::Int:
      return at::native::opencl::OpenCLCastType::INT;
      break;
    case ScalarType::Long:
      return at::native::opencl::OpenCLCastType::LONG;
      break;
    case ScalarType::Half:
      return at::native::opencl::OpenCLCastType::SHORT; // TODO support halfs
      break;
    case ScalarType::Float:
      return at::native::opencl::OpenCLCastType::FLOAT;
      break;
    case ScalarType::Double:
      return at::native::opencl::OpenCLCastType::DOUBLE;
      break;
    case ScalarType::QInt32:
      return at::native::opencl::OpenCLCastType::INT;
      break;
    case ScalarType::QInt8:
      return at::native::opencl::OpenCLCastType::CHAR;
      break;
    case ScalarType::QUInt8:
      return at::native::opencl::OpenCLCastType::CHAR;
      break;

    default:
      return at::native::opencl::OpenCLCastType::LONG; // For unknowns
      break;
  }
}

inline cl::Buffer* toBuffer(void *ptr, bool *is_host = nullptr) {
  cl::Buffer* ret = caffe2::opencl::getBufferFromPtr(ptr);
  if (is_host) {
    *is_host = false;
  }
  if (ret == nullptr) {
    ret = at::opencl::OpenCLCachingHostAllocator_getBuffer(ptr);
    if (is_host) {
      *is_host = ret != nullptr;
    }
  }
  return ret;
}

inline cl_int syncOpenCLPointer(void *ptr) {
  bool is_host;
  cl::Buffer* buffer = toBuffer(ptr, &is_host);
  if (buffer == nullptr) {
    return CL_INVALID_ARG_VALUE;
  }
  cl_int err;
  auto stream = c10::opencl::getCurrentOpenCLStream();

  size_t buffer_size;
  err = buffer->getInfo(CL_MEM_SIZE, &buffer_size);
  if (err != CL_SUCCESS) {
    return err;
  }

  err = stream.stream()->enqueueReadBuffer(*buffer, CL_FALSE, /*offset=*/0, buffer_size, ptr, NULL, NULL);
  return err;
}

}} // namespace at::native
