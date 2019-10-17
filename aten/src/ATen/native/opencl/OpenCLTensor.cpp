#include <ATen/native/opencl/OpenCLTensor.h>
#include <ATen/native/opencl/Resize.h>

namespace at { namespace native {

ptrdiff_t opencl_nElement(const c10::TensorImpl *self) {
  if(opencl_nDimensionLegacyAll(self) == 0) {
    return 0;
  } else {
    return self->numel();
  }
}

int64_t opencl_stride(const TensorImpl *self, int dim) {
  TORCH_CHECK((dim >= 0) && (dim < self->dim()), "opencl_stride ArgNo=", 2, " out of range");
  return self->stride(dim);
}

// NB: Steals ownership of storage
void opencl_stealAndSetStoragePtr(c10::TensorImpl* tensor, c10::StorageImpl* storage) {
  // Caffe2 might have tensors whose storages are null, but we
  // don't allow it in PyTorch.
  AT_ASSERT(storage);
  // Caffe2 also has uninitialized dtype states, which we disallow here
  AT_ASSERT(tensor->storage().dtype() == storage->dtype());

  // We used to allow this, but this breaks device caching.
  // Let's put an actual error message for this one.
  TORCH_CHECK(tensor->storage().device() == storage->device(),
            "Attempted to set the storage of a tensor on device \"", tensor->storage().device(),
             "\" to a storage on different device \"", storage->device(),
            "\".  This is no longer allowed; the devices must match.");
  tensor->set_storage(at::Storage(c10::intrusive_ptr<c10::StorageImpl>::reclaim(storage)));
}

void opencl_setStorageNd(c10::TensorImpl *self, c10::StorageImpl *storage, ptrdiff_t storageOffset, int nDimension, const int64_t *size, const int64_t *stride) {
  /* storage */
  if(self->storage().unsafeGetStorageImpl() != storage)
  {
    TORCH_CHECK(self->storage().unsafeGetStorageImpl(), "Tensor: invalid null storage");
    auto data_type = self->storage().unsafeGetStorageImpl()->dtype();
    if (storage) {
      c10::raw::intrusive_ptr::incref(storage);
      opencl_stealAndSetStoragePtr(self, storage);
    } else {
      opencl_stealAndSetStoragePtr(self, c10::make_intrusive<at::StorageImpl>(
                                            data_type,
                                            0,
                                            self->storage().allocator(),
                                            true).release());
    }
  }

  /* storageOffset */
  TORCH_CHECK(storageOffset >= 0, "Tensor: invalid storage offset");
  self->set_storage_offset(storageOffset);

  /* size and stride */
  at::native::opencl_resizeNd(self, nDimension, size, stride);
}

bool opencl_canUse32BitIndexMath(const TensorImpl* t, ptrdiff_t max_elem) {
  ptrdiff_t elements = opencl_nElement(t);
  if (elements >= max_elem) {
    return false;
  }
  if (t->dim() == 0) {
    return true;
  }

  ptrdiff_t offset = 0;
  ptrdiff_t linearId = elements - 1;

  for (int i = opencl_nDimensionLegacyAll(t) - 1; i >= 0; --i) {
    ptrdiff_t curDimIndex =
      linearId % t->size(i);
    ptrdiff_t curDimOffset = curDimIndex *
      opencl_stride(t, i);
    offset += curDimOffset;
    linearId /= t->size(i);
  }

  if (offset >= max_elem) {
    return false;
  }

  return true;
}

bool opencl_all32BitIndexable(c10::TensorImpl** inputs, int numInputs) {
  for (int i = 0; i < numInputs; ++i) {
    if (!opencl_canUse32BitIndexMath(inputs[i])) {
      return false;
    }
  }
  return true;
}

bool opencl_allContiguous(c10::TensorImpl **inputs, int numInputs) {
  TORCH_INTERNAL_ASSERT(numInputs > 0);
  for (int i = 0; i < numInputs; ++i) {
    if (!inputs[i]->is_contiguous()) {
      return false;
    }
  }
  return true;
}

}} // namespace at::native
