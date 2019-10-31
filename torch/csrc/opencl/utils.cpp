#include <torch/csrc/python_headers.h>
#include <stdarg.h>
#include <string>
#include <torch/csrc/opencl/THOP.h>

#include <torch/csrc/opencl/override_macros.h>

#ifdef USE_OPENCL
// NB: It's a list of *optional* OpenCLStream; when nullopt, that means to use
// whatever the current stream of the device the input is associated with was.
std::vector<c10::optional<c10::opencl::OpenCLStream>> THPUtils_PySequence_to_OpenCLStreamList(PyObject *obj) {
  if (!PySequence_Check(obj)) {
    throw std::runtime_error("Expected a sequence in THPUtils_PySequence_to_OpenCLStreamList");
  }
  THPObjectPtr seq = THPObjectPtr(PySequence_Fast(obj, nullptr));
  if (seq.get() == nullptr) {
    throw std::runtime_error("expected PySequence, but got " + std::string(THPUtils_typename(obj)));
  }

  std::vector<c10::optional<c10::opencl::OpenCLStream>> streams;
  Py_ssize_t length = PySequence_Fast_GET_SIZE(seq.get());
  for (Py_ssize_t i = 0; i < length; i++) {
    PyObject *stream = PySequence_Fast_GET_ITEM(seq.get(), i);

    if (PyObject_IsInstance(stream, THOPStreamClass)) {
      // Spicy hot reinterpret cast!!
      streams.emplace_back( c10::opencl::OpenCLStream::unpack((reinterpret_cast<THOPStream*>(stream))->cdata) );
    } else if (stream == Py_None) {
      streams.emplace_back();
    } else {
      std::runtime_error("Unknown data type found in stream list. Need torch.opencl.Stream or None");
    }
  }
  return streams;
}

#endif
