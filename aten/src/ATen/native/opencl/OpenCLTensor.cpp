#include <ATen/native/opencl/OpenCLTensor.h>

ptrdiff_t opencl_nElement(const c10::TensorImpl *self) {
  if(opencl_nDimensionLegacyAll(self) == 0) {
    return 0;
  } else {
    return self->numel();
  }
}
