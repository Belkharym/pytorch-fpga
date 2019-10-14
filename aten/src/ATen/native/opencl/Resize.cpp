#include <ATen/native/opencl/Resize.h>

#include <c10/opencl/OpenCLFunctions.h>
#include <ATen/native/ResizeCommon.h>

namespace at { namespace native {

void opencl_resizeNd(c10::TensorImpl *self, int nDimension, const int64_t *size, const int64_t *stride) {
  TORCH_CHECK(nDimension >= 0, "resizeNd nDimension must be non-negative");
  at::IntArrayRef sizes(size, nDimension);
  at::optional<at::IntArrayRef> strides;
  if (stride) {
    strides = at::IntArrayRef(stride, nDimension);
  }
  at::native::resize_impl_opencl_(self, sizes, strides, /*device_guard=*/false);
}

void opencl_resize(c10::TensorImpl *self, at::IntArrayRef size, at::IntArrayRef stride) {
  if(stride.data()) {
    TORCH_CHECK(stride.size() == size.size(), 3, "invalid stride");
  }

#ifdef DEBUG
  TORCH_INTERNAL_ASSERT(size.size() <= INT_MAX);
#endif
  opencl_resizeNd(self, size.size(), size.data(), stride.data());
}

void opencl_resize(c10::StorageImpl *self, ptrdiff_t size)
{
  TORCH_CHECK(size >= 0, 2, "invalid size");
  TORCH_INTERNAL_ASSERT(self->allocator() != nullptr);
  int device;
  device = c10::opencl::current_device();

  if (!self->resizable())
    TORCH_CHECK(false, "Trying to resize storage that is not resizable");

  size_t itemsize = self->itemsize();

  if(size == 0)
  {
    self->set_data_ptr(at::DataPtr(nullptr, at::Device(at::DeviceType::OPENCL, device)));
    self->set_numel(0);
  }
  else
  {
    at::DataPtr data =
      self->allocator()->allocate(size * itemsize);

    if (self->data_ptr()) {
      // Enable p2p access when the memcpy is across devices
      CopyBytes(std::min(self->numel(), size) * itemsize,
                  self->data(),
                  self->device(),
                  data.get(),
                  data.device(),
                  /*async=*/true);
    }

    // Destructively overwrite data_ptr
    self->set_data_ptr(std::move(data));
    self->set_numel(size);
  }
}

Tensor& resize_opencl_(Tensor& self, IntArrayRef size) {
#ifdef BUILD_NAMEDTENSOR
  if (self.has_names()) {
    return resize_named_tensor_(self, size);
  }
#endif
  auto* self_ = self.unsafeGetTensorImpl();
  resize_impl_opencl_(self_, size, /*strides=*/c10::nullopt);
  self_->maybe_zero_dim(size.size() == 0);
  return self;
}

}} // namespace at::native