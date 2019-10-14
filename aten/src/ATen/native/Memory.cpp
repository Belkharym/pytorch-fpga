#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/detail/OpenCLHooksInterface.h>
#include <c10/util/Exception.h>
#include <c10/core/Storage.h>

namespace at {
namespace native {

bool is_pinned(const Tensor& self) {
  if (self.type().backend() == Backend::CUDA) {
    return detail::getCUDAHooks().isPinnedPtr(self.storage().data());
  } else {
    return detail::getOpenCLHooks().isPinnedPtr(self.storage().data());
  }
}

Tensor pin_memory(const Tensor& self) {
  if (self.type().backend() != Backend::CPU) {
    AT_ERROR("cannot pin '", self.type().toString(), "' only dense CPU tensors can be pinned");
  }
  if (self.is_pinned()) {
    return self;
  }
  at::Allocator* allocator;
  if (self.type().backend() == Backend::CUDA) {
    allocator = detail::getCUDAHooks().getPinnedMemoryAllocator();
  } else {
    allocator = detail::getOpenCLHooks().getPinnedMemoryAllocator();
  }
  auto storage = Storage(
      self.dtype(),
      detail::computeStorageSize(self.sizes(), self.strides()),
      allocator,
      /*resizable=*/false
  );
  auto tensor = at::empty({0}, self.options()).set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

// Exposes at::has_internal_overlap as an operator for testing purposes
int64_t _debug_has_internal_overlap(const Tensor& self) {
  return static_cast<int64_t>(at::has_internal_overlap(self));
}

}
}
