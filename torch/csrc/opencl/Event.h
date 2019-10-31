#ifndef THCP_EVENT_INC
#define THCP_EVENT_INC

#include <ATen/opencl/OpenCLEvent.h>
#include <torch/csrc/python_headers.h>

struct THOPEvent {
  PyObject_HEAD
  at::opencl::OpenCLEvent opencl_event;
};
extern PyObject *THOPEventClass;

void THOPEvent_init(PyObject *module);

inline bool THOPEvent_Check(PyObject* obj) {
  return THOPEventClass && PyObject_IsInstance(obj, THOPEventClass);
}

#endif // THOP_EVENT_INC
