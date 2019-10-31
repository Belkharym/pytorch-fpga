#include <torch/csrc/opencl/undef_macros.h>

#define THWStoragePtr THOStoragePtr
#define THPStoragePtr THOPStoragePtr
#define THWTensorPtr THOTensorPtr
#define THPTensorPtr THOPTensorPtr

#define THWStorage THOStorage
#define THWStorage_(NAME) THOStorage_(NAME)
#define THWTensor THOTensor
#define THWTensor_(NAME) THOTensor_(NAME)

#define THPStorage_(NAME) TH_CONCAT_4(THOP,Real,Storage_,NAME)
#define THPStorageBaseStr THOPStorageBaseStr
#define THPStorageStr THOPStorageStr
#define THPStorageClass THOPStorageClass
#define THPStorageType THOPStorageType

#define THPTensor_(NAME) TH_CONCAT_4(THOP,Real,Tensor_,NAME)
#define THPTensor_stateless_(NAME) TH_CONCAT_4(THOP,Real,Tensor_stateless_,NAME)
#define THPTensor THOPTensor
#define THPTensorStr THOPTensorStr
#define THPTensorBaseStr THOPTensorBaseStr
#define THPTensorClass THOPTensorClass
#define THPTensorType THOPTensorType

#define THPTensorStatelessType THOPTensorStatelessType
#define THPTensorStateless THOPTensorStateless


#define THSPTensorPtr THOSPTensorPtr

#define THSPTensor_(NAME) TH_CONCAT_4(THOSP,Real,Tensor_,NAME)
#define THSPTensor_stateless_(NAME) TH_CONCAT_4(THOSP,Real,Tensor_stateless_,NAME)
#define THSPTensor THOSPTensor
#define THSPTensorStr THOSPTensorStr
#define THSPTensorBaseStr THOSPTensorBaseStr
#define THSPTensorClass THOSPTensorClass
#define THSPTensorType THOSPTensorType

#define THSPTensorStatelessType THOSPTensorStatelessType
#define THSPTensorStateless THOSPTensorStateless


#define LIBRARY_STATE_NOARGS 
#define LIBRARY_STATE 
#define LIBRARY_STATE_TYPE THOState*,
#define LIBRARY_STATE_TYPE_NOARGS THOState*
#define TH_GENERIC_FILE THO_GENERIC_FILE

#define THHostTensor TH_CONCAT_3(TH,Real,Tensor)
#define THHostTensor_(NAME) TH_CONCAT_4(TH,Real,Tensor_,NAME)
#define THHostStorage TH_CONCAT_3(TH,Real,Storage)
#define THHostStorage_(NAME) TH_CONCAT_4(TH,Real,Storage_,NAME)
