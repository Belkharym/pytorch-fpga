#include <torch/csrc/python_headers.h>

#include <unordered_map>
#include <thread>
#include <chrono>
#include <sstream>
#include <TH/TH.h>
#include <ATen/ATen.h>
#include <ATen/opencl/OpenCLContext.h>
#include <ATen/OpenCLGenerator.h>
#include <c10/opencl/OpenCLFunctions.h>
#include <c10/opencl/OpenCLCachingAllocator.h>

#include <torch/csrc/opencl/THOP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/opencl_lazy_init.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/opencl/python_comm.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/Generator.h>

using namespace torch;

////////////////////////////////////////////////////////////////////////////////
// OpenCL management methods
////////////////////////////////////////////////////////////////////////////////

void THOPModule_setDevice(int device)
{
  c10::opencl::set_device(device);
}

PyObject * THOPModule_setDevice_wrap(PyObject *self, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to setDevice");
  int64_t device = THPUtils_unpackLong(arg);

  torch::utils::opencl_lazy_init();
  THOPModule_setDevice(device);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THOPModule_getDevice_wrap(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  torch::utils::opencl_lazy_init();
  int device = c10::opencl::current_device();
  return PyLong_FromLong(device);
  END_HANDLE_TH_ERRORS
}

PyObject * THOPModule_getDeviceCount_wrap(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  //torch::utils::opencl_lazy_init();
  return PyLong_FromLong(c10::opencl::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject * THOPModule_set_run_yet_variable_to_false_wrap(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  torch::utils::opencl_set_run_yet_variable_to_false();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THOPModule_getCurrentStream_wrap(
    PyObject * /* unused */, PyObject *device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
    THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  int64_t device = THPUtils_unpackLong(device_index);
  return PyLong_FromUnsignedLongLong(
    c10::opencl::getCurrentOpenCLStream(device).pack());
  END_HANDLE_TH_ERRORS
}

PyObject * THOPModule_getDefaultStream_wrap(
    PyObject * /* unused */, PyObject *device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
    THPUtils_checkLong(device_index), "invalid argument to getDefaultStream");
  int64_t device = THPUtils_unpackLong(device_index);
  return PyLong_FromUnsignedLongLong(
    c10::opencl::getDefaultOpenCLStream(device).pack());
  END_HANDLE_TH_ERRORS
}

PyObject * THOPModule_setStream_wrap(PyObject *self, PyObject *obj)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(PyLong_Check(obj), "invalid stream");
  uint64_t bits = PyLong_AsUnsignedLongLong(obj);
  if (bits == static_cast<uint64_t>(-1) && PyErr_Occurred()) {
    throw python_error();
  }
  auto stream = c10::opencl::OpenCLStream::unpack(bits);
  int device = c10::opencl::current_device();
  if (device != stream.device_index()) {
    THOPModule_setDevice(stream.device_index());
  }
  c10::opencl::setCurrentOpenCLStream(stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THOPModule_isDriverSufficient(PyObject *self, PyObject *noargs)
{
  bool systemHasOpenCL = c10::opencl::checkSystemHasOpenCL();
  if (!systemHasOpenCL) {
    return PyBool_FromLong(0);
  }
  return PyBool_FromLong(1);
}

PyObject * THOPModule_getDriverVersion(PyObject *self, PyObject *noargs)
{
  if (!c10::opencl::checkSystemHasOpenCL()) {
    PyErr_Format(PyExc_RuntimeError,
                    "Missing OpenCL driver");
    return nullptr;
  }
  cl_int err;
  c10::opencl::openclDeviceProp* prop = c10::opencl::getCurrentDeviceProperties(&err);
  if (err != CL_SUCCESS || prop == nullptr) {
    PyErr_Format(PyExc_RuntimeError,
                    "Error calling cudaDriverGetVersion: %d %s",
                    err, c10::opencl::clErrorString(err));
    return nullptr;
  }
  std::string major, minor;
  std::stringstream ss(prop->driverVersion);
  std::getline(ss, major, '.');
  std::getline(ss, minor, '.');
  int driverVersion = -1;
  try {
    driverVersion = std::stol(major) * 100 + std::stol(minor) * 10;
  }
  catch (...) { // no throw
  }

  return PyLong_FromLong((int64_t) driverVersion);
}

PyObject * THOPModule_getCompiledVersion(PyObject *self, PyObject *noargs)
{
  return PyLong_FromLong((long) CL_HPP_TARGET_OPENCL_VERSION);
}

PyObject * THOPModule_openclHostAllocator(PyObject *_unused, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  c10::Allocator* allocator = c10::opencl::OpenCLCachingAllocator::get();
  return PyLong_FromVoidPtr(allocator);
  END_HANDLE_TH_ERRORS
}

PyObject * THOPModule_openclSynchronize(PyObject *_unused, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  C10_OPENCL_CHECK(c10::opencl::openclSynchronize());
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static cl_int opencl_sleep(int64_t cycles) {
  cl_int err = CL_SUCCESS;
  auto stream = c10::opencl::getCurrentOpenCLStream();
  cl::UserEvent event{c10::opencl::opencl_context(), &err};
  if (err != CL_SUCCESS) return err;
  std::vector<cl::Event> events{event};
  err = stream.stream()->enqueueMarkerWithWaitList(&events);
  if (err != CL_SUCCESS) return err;
  auto start = std::chrono::system_clock::now();
  // Busy wait
  int64_t duration = 0;
  while (duration < cycles) {
    duration = (std::chrono::system_clock::now() - start).count();
  }
  return event.setStatus(CL_COMPLETE);
}

// Contrairly to CUDA, OpenCL can't count cycle operations, so this uses CPU cycles.
PyObject * THOPModule_openclSleep(PyObject *_unused, PyObject *cycles)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(cycles), "torch.opencl._sleep(): expected 'int'");
  C10_OPENCL_CHECK(opencl_sleep(THPUtils_unpackLong(cycles)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// We need to ensure that as long as a thread will NEVER loose the GIL as long as
// it holds the OpenCL mutex. Otherwise another thread might be scheduled and try to
// e.g. allocate a new tensor which will cause a deadlock. It's enough to have a
// single global, because it can be only set once (openclMutex is not recursive)
// by the thread that owns the mutex (obviously there can be only one such thread).
static PyGILState_STATE openclMutexGILState;

PyObject * THOPModule_openclLockMutex(PyObject *module, PyObject *noargs)
{
  auto mutex = c10::opencl::OpenCLCachingAllocator::getFreeMutex();
  // This has to be a busy loop because we **absolutely need to** hold the GIL
  // or it's a recipe for a deadlock otherwise (if we let other Python threads
  // run while we have the openclMutex, but not the GIL, they might try to e.g.
  // free a OpenCL tensor and acquire the openclMutex without giving up the GIL,
  // because it happens deep within THO).
  while (true) {
    if (mutex->try_lock())
      break;
    {
      AutoNoGIL no_gil;
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

  openclMutexGILState = PyGILState_Ensure();
  Py_RETURN_NONE;
}

PyObject * THOPModule_openclUnlockMutex(PyObject *module, PyObject *noargs)
{
  auto mutex = c10::opencl::OpenCLCachingAllocator::getFreeMutex();
  PyGILState_Release(openclMutexGILState);
  mutex->unlock();
  Py_RETURN_NONE;
}

PyObject * THOPModule_hasPrimaryContext(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to has_primary_context");
  int64_t device_index = static_cast<int64_t>(THPUtils_unpackLong(arg));
  if (at::detail::getCUDAHooks().hasPrimaryContext(device_index)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject * THOPModule_emptyCache(PyObject *_unused, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  c10::opencl::OpenCLCachingAllocator::emptyCache();
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject * THOPModule_memoryAllocated(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to memory_allocated");
  int device = (int) THPUtils_unpackLong(arg);
  auto memory_allocated = c10::opencl::OpenCLCachingAllocator::currentMemoryAllocated(device);
  return PyLong_FromUnsignedLongLong(memory_allocated);
  END_HANDLE_TH_ERRORS
}

PyObject * THOPModule_maxMemoryAllocated(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to max_memory_allocated");
  int device = (int) THPUtils_unpackLong(arg);
  auto max_memory_allocated = c10::opencl::OpenCLCachingAllocator::maxMemoryAllocated(device);
  return PyLong_FromUnsignedLongLong(max_memory_allocated);
  END_HANDLE_TH_ERRORS
}

PyObject * THOPModule_resetMaxMemoryAllocated(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to reset_max_memory_allocated");
  int device = (int) THPUtils_unpackLong(arg);
  c10::opencl::OpenCLCachingAllocator::resetMaxMemoryAllocated(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject * THOPModule_memoryCached(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to memory_cached");
  int device = (int) THPUtils_unpackLong(arg);
  auto memory_cached = c10::opencl::OpenCLCachingAllocator::currentMemoryCached(device);
  return PyLong_FromUnsignedLongLong(memory_cached);
  END_HANDLE_TH_ERRORS
}

PyObject * THOPModule_maxMemoryCached(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to max_memory_cached");
  int device = (int) THPUtils_unpackLong(arg);
  auto max_memory_cached = c10::opencl::OpenCLCachingAllocator::maxMemoryCached(device);
  return PyLong_FromUnsignedLongLong(max_memory_cached);
  END_HANDLE_TH_ERRORS
}

PyObject * THOPModule_resetMaxMemoryCached(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to reset_max_memory_cached");
  int device = (int) THPUtils_unpackLong(arg);
  c10::opencl::OpenCLCachingAllocator::resetMaxMemoryCached(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

////////////////////////////////////////////////////////////////////////////////
// OpenCL module initialization
////////////////////////////////////////////////////////////////////////////////

static void bindOpenCLDeviceProperties(PyObject* module) {
  using c10::opencl::openclDeviceProp;
  // Add class and method to torch.opencl
  auto m = py::handle(module).cast<py::module>();
  py::class_<openclDeviceProp>(m, "_OpenCLDeviceProperties")
    .def_readonly("addressBits", &openclDeviceProp::addressBits)
    .def_readonly("available", (bool openclDeviceProp::*)&openclDeviceProp::available)
    .def_readonly("builtInKernels", &openclDeviceProp::builtInKernels)
    .def_readonly("compilerAvailable", (bool openclDeviceProp::*)&openclDeviceProp::compilerAvailable)
    .def_readonly("doubleFpConfig", &openclDeviceProp::doubleFpConfig)
    .def_readonly("endianLittle", (bool openclDeviceProp::*)&openclDeviceProp::endianLittle)
    .def_readonly("errorCorrectionSupport", (bool openclDeviceProp::*)&openclDeviceProp::errorCorrectionSupport)
    .def_readonly("executionCapabilities", &openclDeviceProp::executionCapabilities)
    .def_readonly("extensions", &openclDeviceProp::extensions)
    .def_readonly("globalMemCacheSize", &openclDeviceProp::globalMemCacheSize)
    .def_readonly("globalMemCacheType", &openclDeviceProp::globalMemCacheType)
    .def_readonly("globalMemCachelineSize", &openclDeviceProp::globalMemCachelineSize)
    .def_readonly("globalMemSize", &openclDeviceProp::globalMemSize)
    .def_readonly("halfFpConfig", &openclDeviceProp::halfFpConfig)
    .def_readonly("hostUnifiedMemory", (bool openclDeviceProp::*)&openclDeviceProp::hostUnifiedMemory)
    .def_readonly("imageSupport", (bool openclDeviceProp::*)&openclDeviceProp::imageSupport)
    .def_readonly("image2dMaxHeight", &openclDeviceProp::image2dMaxHeight)
    .def_readonly("image2dMaxWidth", &openclDeviceProp::image2dMaxWidth)
    .def_readonly("image3dMaxDepth", &openclDeviceProp::image3dMaxDepth)
    .def_readonly("image3dMaxHeight", &openclDeviceProp::image3dMaxHeight)
    .def_readonly("image3dMaxWidth", &openclDeviceProp::image3dMaxWidth)
    .def_readonly("imageMaxBufferSize", &openclDeviceProp::imageMaxBufferSize)
    .def_readonly("imageMaxArraySize", &openclDeviceProp::imageMaxArraySize)
    .def_readonly("linkerAvailable", (bool openclDeviceProp::*)&openclDeviceProp::linkerAvailable)
    .def_readonly("localMemSize", &openclDeviceProp::localMemSize)
    .def_readonly("localMemType", &openclDeviceProp::localMemType)
    .def_readonly("maxClockFrequency", &openclDeviceProp::maxClockFrequency)
    .def_readonly("maxComputeUnits", &openclDeviceProp::maxComputeUnits)
    .def_readonly("maxConstantArgs", &openclDeviceProp::maxConstantArgs)
    .def_readonly("maxConstantBufferSize", &openclDeviceProp::maxConstantBufferSize)
    .def_readonly("maxMemAllocSize", &openclDeviceProp::maxMemAllocSize)
    .def_readonly("maxParameterSize", &openclDeviceProp::maxParameterSize)
    .def_readonly("maxReadImageArgs", &openclDeviceProp::maxReadImageArgs)
    .def_readonly("maxSamplers", &openclDeviceProp::maxSamplers)
    .def_readonly("maxWorkGroupSize", &openclDeviceProp::maxWorkGroupSize)
    .def_readonly("maxWorkItemDimensions", &openclDeviceProp::maxWorkItemDimensions)
    .def_readonly("maxWorkItemSizes", &openclDeviceProp::maxWorkItemSizes)
    .def_readonly("maxWriteImageArgs", &openclDeviceProp::maxWriteImageArgs)
    .def_readonly("memBaseAddrAlign", &openclDeviceProp::memBaseAddrAlign)
    .def_readonly("minDataTypeAlignSize", &openclDeviceProp::minDataTypeAlignSize)
    .def_readonly("name", &openclDeviceProp::name)
    .def_readonly("nativeVectorWidthChar", &openclDeviceProp::nativeVectorWidthChar)
    .def_readonly("nativeVectorWidthDouble", &openclDeviceProp::nativeVectorWidthDouble)
    .def_readonly("nativeVectorWidthFloat", &openclDeviceProp::nativeVectorWidthFloat)
    .def_readonly("nativeVectorWidthHalf", &openclDeviceProp::nativeVectorWidthHalf)
    .def_readonly("nativeVectorWidthInt", &openclDeviceProp::nativeVectorWidthInt)
    .def_readonly("nativeVectorWidthLong", &openclDeviceProp::nativeVectorWidthLong)
    .def_readonly("nativeVectorWidthShort", &openclDeviceProp::nativeVectorWidthShort)
    .def_readonly("openclCVersion", &openclDeviceProp::openclCVersion)
    .def_readonly("parentDevice", (void* openclDeviceProp::*)&openclDeviceProp::parentDevice)
    .def_readonly("partitionMaxSubDevices", &openclDeviceProp::partitionMaxSubDevices)
    .def_readonly("partitionProperties", &openclDeviceProp::partitionProperties)
    .def_readonly("partitionAffinityDomain", &openclDeviceProp::partitionAffinityDomain)
    .def_readonly("partitionType", &openclDeviceProp::partitionType)
    .def_readonly("platform", (void* openclDeviceProp::*)&openclDeviceProp::platform)
    .def_readonly("preferredVectorWidthChar", &openclDeviceProp::preferredVectorWidthChar)
    .def_readonly("preferredVectorWidthDouble", &openclDeviceProp::preferredVectorWidthDouble)
    .def_readonly("preferredVectorWidthFloat", &openclDeviceProp::preferredVectorWidthFloat)
    .def_readonly("preferredVectorWidthHalf", &openclDeviceProp::preferredVectorWidthHalf)
    .def_readonly("preferredVectorWidthInt", &openclDeviceProp::preferredVectorWidthInt)
    .def_readonly("preferredVectorWidthLong", &openclDeviceProp::preferredVectorWidthLong)
    .def_readonly("preferredVectorWidthShort", &openclDeviceProp::preferredVectorWidthShort)
    .def_readonly("printfBufferSize", &openclDeviceProp::printfBufferSize)
    .def_readonly("preferredInteropUserSync", (bool openclDeviceProp::*)&openclDeviceProp::preferredInteropUserSync)
    .def_readonly("profile", &openclDeviceProp::profile)
    .def_readonly("profilingTimerResolution", &openclDeviceProp::profilingTimerResolution)
    .def_readonly("queueProperties", &openclDeviceProp::queueProperties)
    .def_readonly("referenceCount", &openclDeviceProp::referenceCount)
    .def_readonly("singleFpConfig", &openclDeviceProp::singleFpConfig)
    .def_readonly("deviceType", &openclDeviceProp::type)
    .def_readonly("vendor", &openclDeviceProp::vendor)
    .def_readonly("vendorId", &openclDeviceProp::vendorId)
    .def_readonly("version", &openclDeviceProp::version)
    .def_readonly("driverVersion", &openclDeviceProp::driverVersion)
    .def("__repr__", [](const openclDeviceProp &prop) {
      std::ostringstream stream;
      stream << "_OpenCLDeviceProperties(name='" << prop.name
             << ", version=" << prop.version << ", driver_version=" << prop.driverVersion
             << ", total_memory=" << prop.globalMemSize / (1024 * 1024)
             << "MB, multi_processor_count=" << prop.maxWorkGroupSize << ")";
      return stream.str();
    });
  m.def("_get_device_properties", [](int device) -> openclDeviceProp * {
    return at::opencl::getDeviceProperties(device);
  }, py::return_value_policy::reference);
}

// Callback for python part. Used for additional initialization of python classes
static PyObject * THOPModule_initExtension(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  at::globalContext().lazyInitOpenCL();

  auto m = THPObjectPtr(PyImport_ImportModule("torch.opencl"));
  if (!m) throw python_error();

  // // Register Storage Python objects with DynamicTypes.cpp
  // THOPDoubleStorage_postInit(m);
  // THOPFloatStorage_postInit(m);
  // THOPHalfStorage_postInit(m);
  // THOPLongStorage_postInit(m);
  // THOPIntStorage_postInit(m);
  // THOPShortStorage_postInit(m);
  // THOPCharStorage_postInit(m);
  // THOPByteStorage_postInit(m);
  // THOPBoolStorage_postInit(m);
  // THOPBFloat16Storage_postInit(m);

  bool has_half = true;

  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString doesn't steal reference. So no need to incref.
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  set_module_attr("has_half", has_half ? Py_True : Py_False);

  auto num_gpus = c10::opencl::device_count();
  auto default_opencl_generators = PyTuple_New(static_cast<Py_ssize_t>(num_gpus));
  for(int i = 0; i < num_gpus; i++) {
    auto gen = at::opencl::detail::getDefaultOpenCLGenerator(i);
    auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(gen);
    // This reference is meant to be given away, so no need to incref here.
    PyTuple_SetItem(default_opencl_generators, i, (PyObject*)cast_gen);
  }
  set_module_attr("default_generators", default_opencl_generators);

  bindOpenCLDeviceProperties(m);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static struct PyMethodDef _THOPModule_methods[] = {
  {"_opencl_init",        (PyCFunction)THOPModule_initExtension,    METH_NOARGS,  nullptr},
  {"_opencl_setDevice",   (PyCFunction)THOPModule_setDevice_wrap,   METH_O,       nullptr},
  {"_opencl_getDevice",   (PyCFunction)THOPModule_getDevice_wrap,   METH_NOARGS,  nullptr},
  {"_opencl_getDeviceCount", (PyCFunction)THOPModule_getDeviceCount_wrap, METH_NOARGS, nullptr},
  {"_opencl_set_run_yet_variable_to_false",
    (PyCFunction)THOPModule_set_run_yet_variable_to_false_wrap, METH_NOARGS, nullptr},
  {"_opencl_getCurrentStream",
    (PyCFunction)THOPModule_getCurrentStream_wrap, METH_O, nullptr},
  {"_opencl_getDefaultStream",
    (PyCFunction)THOPModule_getDefaultStream_wrap, METH_O, nullptr},
  {"_opencl_setStream",    (PyCFunction)THOPModule_setStream_wrap,  METH_O, nullptr},
  {"_opencl_isDriverSufficient", (PyCFunction)THOPModule_isDriverSufficient, METH_NOARGS, nullptr},
  {"_opencl_getDriverVersion", (PyCFunction)THOPModule_getDriverVersion, METH_NOARGS, nullptr},
  {"_opencl_getCompiledVersion", (PyCFunction)THOPModule_getCompiledVersion, METH_NOARGS, nullptr},
  {"_opencl_hasPrimaryContext", (PyCFunction) THOPModule_hasPrimaryContext,  METH_O,  nullptr},
  {"_opencl_emptyCache", (PyCFunction) THOPModule_emptyCache,       METH_NOARGS,  nullptr},
  {"_opencl_memoryAllocated", (PyCFunction) THOPModule_memoryAllocated, METH_O,  nullptr},
  {"_opencl_maxMemoryAllocated", (PyCFunction) THOPModule_maxMemoryAllocated, METH_O,  nullptr},
  {"_opencl_resetMaxMemoryAllocated", (PyCFunction) THOPModule_resetMaxMemoryAllocated, METH_O,  nullptr},
  {"_opencl_memoryCached", (PyCFunction) THOPModule_memoryCached, METH_O,  nullptr},
  {"_opencl_maxMemoryCached", (PyCFunction) THOPModule_maxMemoryCached, METH_O,  nullptr},
  {"_opencl_resetMaxMemoryCached", (PyCFunction) THOPModule_resetMaxMemoryCached, METH_O,  nullptr},
  {"_opencl_openclHostAllocator", (PyCFunction)THOPModule_openclHostAllocator, METH_NOARGS, nullptr},
  {"_opencl_synchronize", (PyCFunction)THOPModule_openclSynchronize, METH_NOARGS, nullptr},
  {"_opencl_sleep", (PyCFunction)THOPModule_openclSleep, METH_O, nullptr},
  {"_opencl_lock_mutex",   (PyCFunction)THOPModule_openclLockMutex,   METH_NOARGS,  nullptr},
  {"_opencl_unlock_mutex", (PyCFunction)THOPModule_openclUnlockMutex, METH_NOARGS,  nullptr},
  {nullptr}
};

PyMethodDef* THOPModule_methods() {
  return _THOPModule_methods;
}

namespace torch { namespace opencl {

void initModule(PyObject *module) {
  python::initCommMethods(module);
}

}}
