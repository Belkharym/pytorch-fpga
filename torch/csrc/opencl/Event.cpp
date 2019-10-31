#include <torch/csrc/opencl/Event.h>
#include <torch/csrc/opencl/Module.h>
#include <torch/csrc/opencl/Stream.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <c10/opencl/OpenCLGuard.h>
#include <ATen/opencl/OpenCLEvent.h>

#include <structmember.h>

PyObject *THOPEventClass = nullptr;

static PyObject * THOPEvent_pynew(
    PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  HANDLE_TH_ERRORS
  unsigned char enable_timing = 0;
  unsigned char blocking = 0;
  unsigned char interprocess = 0;

  static char *kwlist[] =
    {"enable_timing", "blocking", "interprocess", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|bbb", kwlist,
      &enable_timing, &blocking, &interprocess)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THOPEvent* self = (THOPEvent *)ptr.get();

  new (&self->opencl_event) at::opencl::OpenCLEvent();

  return (PyObject *)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THOPEvent_dealloc(THOPEvent *self) {
  self->opencl_event.~OpenCLEvent();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THOPEvent_get_opencl_event(THOPEvent *self, void *unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->opencl_event.event());
  END_HANDLE_TH_ERRORS
}

static PyObject * THOPEvent_get_device(THOPEvent *self, void *unused) {
  HANDLE_TH_ERRORS
  at::optional<at::Device> device = self->opencl_event.device();
  if (!device) {
    Py_RETURN_NONE;
  }
  return THPDevice_New(device.value());
  END_HANDLE_TH_ERRORS
}

static PyObject * THOPEvent_record(THOPEvent *self, THOPStream *stream) {
  HANDLE_TH_ERRORS
  self->opencl_event.record(stream->opencl_stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THOPEvent_wait(THOPEvent *self, THOPStream *stream) {
  HANDLE_TH_ERRORS
  with_no_gil([&] { self->opencl_event.block(stream->opencl_stream); });
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THOPEvent_query(THOPEvent *self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->opencl_event.query());
  END_HANDLE_TH_ERRORS
}

static PyObject * THOPEvent_synchronize(THOPEvent *self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  with_no_gil([&] { self->opencl_event.synchronize(); });
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef THOPEvent_properties[] = {
  {"device", (getter)THOPEvent_get_device, nullptr, nullptr, nullptr},
  {"opencl_event", (getter)THOPEvent_get_opencl_event, nullptr, nullptr, nullptr},
  {nullptr}
};

static PyMethodDef THOPEvent_methods[] = {
  {(char*)"record", (PyCFunction)THOPEvent_record, METH_O, nullptr},
  {(char*)"wait", (PyCFunction)THOPEvent_wait, METH_O, nullptr},
  {(char*)"query", (PyCFunction)THOPEvent_query, METH_NOARGS, nullptr},
  {(char*)"synchronize", (PyCFunction)THOPEvent_synchronize,
    METH_NOARGS, nullptr},
  {nullptr}
};

PyTypeObject THOPEventType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._C._OpenCLEventBase",           /* tp_name */
  sizeof(THOPEvent),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THOPEvent_dealloc,         /* tp_dealloc */
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
  THOPEvent_methods,                     /* tp_methods */
  0,                                     /* tp_members */
  THOPEvent_properties,                  /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THOPEvent_pynew,                       /* tp_new */
};

void THOPEvent_init(PyObject *module) {
  THOPEventClass = (PyObject*)&THOPEventType;
  if (PyType_Ready(&THOPEventType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THOPEventType);
  if (PyModule_AddObject(
      module, "_OpenCLEventBase", (PyObject *)&THOPEventType) < 0) {
    throw python_error();
  }
}
