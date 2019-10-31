#ifndef THOP_STREAM_INC
#define THOP_STREAM_INC

#include <c10/opencl/OpenCLMacros.h>
#include <c10/opencl/OpenCLStream.h>
#include <c10/opencl/OpenCLFunctions.h>
#include <torch/csrc/python_headers.h>

struct THOPStream {
  PyObject_HEAD
  uint64_t cdata;
  c10::opencl::OpenCLStream opencl_stream;
};
extern PyObject *THOPStreamClass;

void THOPStream_init(PyObject *module);

inline bool THOPStream_Check(PyObject* obj) {
  return THOPStreamClass && PyObject_IsInstance(obj, THOPStreamClass);
}

#endif // THOP_STREAM_INC
