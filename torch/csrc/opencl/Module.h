#ifndef THOP_OPENCL_MODULE_INC
#define THOP_OPENCL_MODULE_INC

#ifdef _THP_CORE
void THOPModule_setDevice(int idx);
PyObject * THOPModule_getDevice_wrap(PyObject *self);
PyObject * THOPModule_setDevice_wrap(PyObject *self, PyObject *arg);
PyObject * THOPModule_getDeviceName_wrap(PyObject *self, PyObject *arg);
PyObject * THOPModule_getDriverVersion(PyObject *self);
PyObject * THOPModule_isDriverSufficient(PyObject *self);
PyObject * THOPModule_getCurrentBlasHandle_wrap(PyObject *self);
#endif

#endif
