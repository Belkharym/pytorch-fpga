#include <torch/csrc/opencl/Stream.h>
#include <torch/csrc/opencl/Module.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>

#include <c10/opencl/OpenCLGuard.h>

#include <structmember.h>

PyObject *THOPStreamClass = nullptr;

static PyObject * THOPStream_pynew(
  PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  HANDLE_TH_ERRORS

  int current_device = c10::opencl::current_device();

  int priority = 0;
  uint64_t cdata = 0;

  static char *kwlist[] = {"priority", "_cdata", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "|iK", kwlist, &priority, &cdata)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  c10::opencl::OpenCLStream stream =
    cdata ?
    c10::opencl::OpenCLStream::unpack(cdata) :
    c10::opencl::getStreamFromPool(
      /* isHighPriority */ priority < 0 ? true : false);

  THOPStream* self = (THOPStream *)ptr.get();
  self->cdata = stream.pack();
  new (&self->opencl_stream) c10::opencl::OpenCLStream(stream);

  return (PyObject *)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THOPStream_dealloc(THOPStream *self) {
  self->opencl_stream.~OpenCLStream();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THOPStream_get_device(THOPStream *self, void *unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(self->opencl_stream.device());
  END_HANDLE_TH_ERRORS
}

static PyObject * THOPStream_get_opencl_stream(THOPStream *self, void *unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->opencl_stream.stream());
  END_HANDLE_TH_ERRORS
}

static PyObject * THOPStream_synchronize(THOPStream *self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  with_no_gil([&] { self->opencl_stream.synchronize(); });
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THOPStream_eq(THOPStream *self, THOPStream *other) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->opencl_stream == other->opencl_stream);
  END_HANDLE_TH_ERRORS
}

static struct PyMemberDef THOPStream_members[] = {
  {(char*)"_cdata",
    T_ULONGLONG, offsetof(THOPStream, cdata), READONLY, nullptr},
  {nullptr}
};

static struct PyGetSetDef THOPStream_properties[] = {
  {"device", (getter)THOPStream_get_device, nullptr, nullptr, nullptr},
  {"opencl_stream",
    (getter)THOPStream_get_opencl_stream, nullptr, nullptr, nullptr},
  {nullptr}
};

static PyMethodDef THOPStream_methods[] = {
  {(char*)"synchronize",
    (PyCFunction)THOPStream_synchronize, METH_NOARGS, nullptr},
  {(char*)"__eq__", (PyCFunction)THOPStream_eq, METH_O, nullptr},
  {nullptr}
};

PyTypeObject THOPStreamType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._C._OpenCLStreamBase",          /* tp_name */
  sizeof(THOPStream),                    /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THOPStream_dealloc,        /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  nullptr,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THOPStream_methods,                    /* tp_methods */
  THOPStream_members,                    /* tp_members */
  THOPStream_properties,                 /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THOPStream_pynew,                      /* tp_new */
};


void THOPStream_init(PyObject *module)
{
  THOPStreamClass = (PyObject*)&THOPStreamType;
  if (PyType_Ready(&THOPStreamType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THOPStreamType);
  if (PyModule_AddObject(
      module, "_OpenCLStreamBase", (PyObject *)&THOPStreamType) < 0) {
    throw python_error();
  }
}
