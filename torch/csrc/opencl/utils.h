#ifndef THOP_UTILS_H
#define THOP_UTILS_H

#define THOPUtils_(NAME) TH_CONCAT_4(THOP,Real,Utils_,NAME)

#define THOStoragePtr  TH_CONCAT_3(THO,Real,StoragePtr)
#define THOTensorPtr   TH_CONCAT_3(THO,Real,TensorPtr)
#define THOPStoragePtr TH_CONCAT_3(THOP,Real,StoragePtr)
#define THOPTensorPtr  TH_CONCAT_3(THOP,Real,TensorPtr)

#define THOSTensorPtr  TH_CONCAT_3(THOS,Real,TensorPtr)
#define THOSPTensorPtr TH_CONCAT_3(THOSP,Real,TensorPtr)

std::vector<c10::optional<c10::opencl::OpenCLStream>> THPUtils_PySequence_to_OpenCLStreamList(PyObject *obj);

#endif
