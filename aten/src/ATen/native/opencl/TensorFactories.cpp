#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorFactories.h>
#include <c10/util/Exception.h>
#include <ATen/Backend.h>

#include <c10/opencl/OpenCLFunctions.h>
#include <aten/src/ATen/native/opencl/OpenCLOperations.h>

namespace at {
namespace native {

Tensor empty_opencl(IntArrayRef size, const TensorOptions& options, c10::optional<MemoryFormat> optional_memory_format) {
  TORCH_INTERNAL_ASSERT(options.backend() == at::Backend::OpenCL);
  TORCH_INTERNAL_ASSERT(!options.is_variable());  // is_variable should have been 'unpacked'  // TODO: remove this when Variable and Tensor are merged
  TORCH_CHECK(!options.pinned_memory(), "Only dense CPU tensors can be pinned");
  check_size_nonnegative(size);

  auto* allocator = c10::GetAllocator(DeviceType::OPENCL);
  int64_t nelements = prod_intlist(size);
  auto dtype = options.dtype();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
    dtype,
    nelements,
    allocator->allocate(nelements * dtype.itemsize()),
    allocator,
    /*resizeable=*/true);

  auto tensor = at::detail::make_tensor<TensorImpl>(storage_impl, TensorTypeId::OpenCLTensorId);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  auto memory_format = optional_memory_format.value_or(MemoryFormat::Contiguous);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  return tensor;
}

static void pointwise_op(Tensor& a, const Tensor& b, const Tensor& c, at::native::opencl::OpenCLOperations op) {
  // TODO Call OpenCL kernel.
}

static void logical_tensor(Tensor& self_, const Tensor& t1, const Tensor& t2, at::native::opencl::OpenCLOperations op) {
  TORCH_CHECK(t1.sizes().equals(t2.sizes()), "sizes don't match.");
  self_.resize_as_(t1);
  // See THC_logicalTensor in aten/src/THC/THCTensorMathCompareT.cuh for implementation details
}

Tensor _eq_opencl(const Tensor &self, const Tensor& other) {
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  // TODO Perform checks and call logical_tensor with OpenCLOperations::EQ
  // Check _th_eq from build/aten/src/ATen/LegacyTHFunctionsCUDA.cpp for implementation details.
  return result;
}

Tensor _eq_opencl(const Tensor &self, Scalar other) {
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  // TODO Perform checks and call logical_tensor with OpenCLOperations::EQ
  // Check _th_eq from build/aten/src/ATen/LegacyTHFunctionsCUDA.cpp for implementation details.
  return result;
}

}} // namespace at::native
