#ifndef THOP_STORAGE_INC
#define THOP_STORAGE_INC

#define THOPStorageStr TH_CONCAT_STRING_3(torch.opencl.,Real,Storage)
#define THOPStorageClass TH_CONCAT_3(THOP,Real,StorageClass)
#define THOPStorage_(NAME) TH_CONCAT_4(THOP,Real,Storage_,NAME)

#define THOPDoubleStorage_Check(obj) \
    PyObject_IsInstance(obj, THOPDoubleStorageClass)
#define THOPFloatStorage_Check(obj) \
    PyObject_IsInstance(obj, THOPFloatStorageClass)
#define THOPHalfStorage_Check(obj) \
    PyObject_IsInstance(obj, THOPHalfStorageClass)
#define THOPLongStorage_Check(obj) \
    PyObject_IsInstance(obj, THOPLongStorageClass)
#define THOPIntStorage_Check(obj) \
    PyObject_IsInstance(obj, THOPIntStorageClass)
#define THOPShortStorage_Check(obj) \
    PyObject_IsInstance(obj, THOPShortStorageClass)
#define THOPCharStorage_Check(obj) \
    PyObject_IsInstance(obj, THOPCharStorageClass)
#define THOPByteStorage_Check(obj) \
    PyObject_IsInstance(obj, THOPByteStorageClass)
#define THOPBoolStorage_Check(obj) \
    PyObject_IsInstance(obj, THOPBoolStorageClass)
#define THOPBFloat16Storage_Check(obj) \
    PyObject_IsInstance(obj, THOPBFloat16StorageClass)

#define THOPDoubleStorage_CData(obj)      (obj)->cdata
#define THOPFloatStorage_CData(obj)       (obj)->cdata
#define THOPLongStorage_CData(obj)        (obj)->cdata
#define THOPIntStorage_CData(obj)         (obj)->cdata
#define THOPShortStorage_CData(obj)       (obj)->cdata
#define THOPCharStorage_CData(obj)        (obj)->cdata
#define THOPByteStorage_CData(obj)        (obj)->cdata
#define THOPBoolStorage_CData(obj)        (obj)->cdata
#define THOPBFloat16Storage_CData(obj)    (obj)->cdata

#ifdef _THP_CORE
#define THOPStorageType TH_CONCAT_3(THOP,Real,StorageType)
#define THOPStorageBaseStr TH_CONCAT_STRING_3(OpenCL,Real,StorageBase)
#endif

#include <torch/csrc/opencl/override_macros.h>

#endif
