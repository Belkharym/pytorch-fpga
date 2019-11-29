#pragma once

#include <ATen/ATen.h>
#include <ATen/core/TensorBody.h>

#define CAT_ARRAY_BATCH_SIZE 1024
#define CAT_ARRAY_MAX_INPUT_DIMS 4

namespace at { namespace native {

inline int opencl_nDimensionLegacyAll(const c10::TensorImpl& tensor) {
  if (tensor.is_empty()) {
    return 0;
  } else if (tensor.dim() == 0) {
    return 1;
  } else {
    return tensor.dim();
  }
}

ptrdiff_t opencl_nElement(const c10::TensorImpl &self);

int64_t opencl_stride(const TensorImpl &self, int dim);

// NB: Non-retaining
inline c10::StorageImpl* opencl_getStoragePtr(const c10::TensorImpl &tensor) {
  // Within PyTorch, the invariant is that storage_ is always
  // initialized; we never have tensors that don't have any storage.
  // However, for Caffe2, this is not true, because they have permitted
  // tensors to be allocated without specifying what scalar type
  // they should be, only to be filled when GetMutableData is called
  // for the first time (providing the necessary type).  It is an ERROR to
  // invoke any PyTorch operations on such a half-constructed storage,
  // and this check tests for that case.
  TORCH_CHECK(tensor.storage(), "Cannot use PyTorch operations on a half-constructed "
           "tensor.  If this tensor came from Caffe2, please call GetMutableData on "
           "it first; otherwise, this is a bug, please report it.");
  return tensor.storage().unsafeGetStorageImpl();
}

template<typename scalar_t>
scalar_t *opencl_data(const c10::TensorImpl &self)
{
  if(opencl_getStoragePtr(self))
    return opencl_getStoragePtr(self)->data<scalar_t>() + self.storage_offset();
  else
    return NULL;
}

inline void opencl_maybe_zero_dim(c10::TensorImpl &tensor, bool condition_when_zero_dim) {
  bool set_zero_dim = condition_when_zero_dim && tensor.sizes().size() == 1 && tensor.size(0) == 1;
  if (set_zero_dim) {
    tensor.set_sizes_and_strides({}, {});
  }
}

void opencl_setStorageNd(c10::TensorImpl &self, c10::StorageImpl *storage, ptrdiff_t storageOffset, int nDimension, const int64_t *size, const int64_t *stride);

inline void opencl_check_shape_except_dim(
    c10::TensorImpl &first, c10::TensorImpl &second, int dimension)
{
  int first_dims = first.dim();
  int second_dims = second.dim();
  TORCH_CHECK(first_dims == second_dims, "ArgNo=", 0,
      " Tensors must have same number of dimensions: got %d and %d",
      first_dims, second_dims);
  for (int dim = 0; dim < first_dims; dim++) {
    if (dim == dimension) {
      continue;
    }
    int64_t first_dim_size = first.size(dim);
    int64_t second_dim_size = second.size(dim);
    TORCH_CHECK(first_dim_size == second_dim_size, "ArgNo=", 0,
        " Sizes of tensors must match except in dimension %d. Got %lld and %lld in dimension %d",
        dimension, (long long)first_dim_size, (long long)second_dim_size, dim);
  }
}

bool opencl_canUse32BitIndexMath(const c10::TensorImpl& t, ptrdiff_t max_elem=INT32_MAX);

bool opencl_all32BitIndexable(c10::TensorImpl** inputs, int numInputs);

bool opencl_allContiguous(c10::TensorImpl **inputs, int numInputs);

}} // namespace at::native