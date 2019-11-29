#pragma once

#include <ATen/ATen.h>

#include <c10/opencl/OpenCLGuard.h>
#include <ATen/native/opencl/OpenCLTensor.h>

namespace at { namespace native {

void opencl_resizeNd(c10::TensorImpl *self, int nDimension, const int64_t *size, const int64_t *stride);
void opencl_resize(c10::TensorImpl *self, at::IntArrayRef size, at::IntArrayRef stride);
void opencl_resize(c10::StorageImpl *self, ptrdiff_t size);
void opencl_resizeAs(c10::TensorImpl *self, c10::TensorImpl *src);

// These functions are called by native::resize_ as well as (legacy) THC resize.
// They are not in THC/THCTensor.cpp because the at namespace is easier
// to benchmark than THC; I can't get gbenchmark to call fns from THTensor.cpp

inline const int64_t* opencl_getSizePtr(c10::TensorImpl* tensor) {
  return tensor->sizes().data();
}

inline void maybe_resize_storage_opencl(TensorImpl* self, int64_t new_size) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in Resize.h)
  if (new_size > 0) {
    if (!opencl_getStoragePtr(*self)) {
      AT_ERROR("Tensor: invalid null storage");
    }
    if (new_size + self->storage_offset() > self->storage().numel()) {
      opencl_resize(
          opencl_getStoragePtr(*self),
          new_size + self->storage_offset());
    }
  }
}

inline TensorImpl* resize_impl_opencl_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool device_guard = true) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  // NB: We don't need to hold the device guard when calling from TH
  at::opencl::OptionalOpenCLGuard guard;
  if (device_guard) {
    guard.set_index(self->storage().device().index());
  }

  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    // NB: storage size can be different from numel.
    for (size_t dim = 0; dim < size.size(); ++dim) {
      // FIXME: Don't rely on storage_size being negative because this
      // may not be true for some edge cases.
      if (size[dim] == 0) {
        storage_size = 0;
        break;
      }
      storage_size += (size[dim] - 1) * stride.value()[dim];
    }
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }
  maybe_resize_storage_opencl(self, storage_size);

  return self;
}

Tensor& resize_opencl_(Tensor& self, IntArrayRef size, c10::optional<c10::MemoryFormat> mem_format);

}}
