#pragma once

#include <ATen/ATen.h>
#include <ATen/core/TensorBody.h>


inline int opencl_nDimensionLegacyAll(const c10::TensorImpl* tensor) {
  if (tensor->is_empty()) {
    return 0;
  } else if (tensor->dim() == 0) {
    return 1;
  } else {
    return tensor->dim();
  }
}

ptrdiff_t opencl_nElement(const c10::TensorImpl *self);

// NB: Non-retaining
inline c10::StorageImpl* opencl_getStoragePtr(const c10::TensorImpl* tensor) {
  // Within PyTorch, the invariant is that storage_ is always
  // initialized; we never have tensors that don't have any storage.
  // However, for Caffe2, this is not true, because they have permitted
  // tensors to be allocated without specifying what scalar type
  // they should be, only to be filled when GetMutableData is called
  // for the first time (providing the necessary type).  It is an ERROR to
  // invoke any PyTorch operations on such a half-constructed storage,
  // and this check tests for that case.
  TORCH_CHECK(tensor->storage(), "Cannot use PyTorch operations on a half-constructed "
           "tensor.  If this tensor came from Caffe2, please call GetMutableData on "
           "it first; otherwise, this is a bug, please report it.");
  return tensor->storage().unsafeGetStorageImpl();
}

inline void opencl_maybe_zero_dim(c10::TensorImpl *tensor, bool condition_when_zero_dim) {
  bool set_zero_dim = condition_when_zero_dim && tensor->sizes().size() == 1 && tensor->size(0) == 1;
  if (set_zero_dim) {
    tensor->set_sizes_and_strides({}, {});
  }
}

