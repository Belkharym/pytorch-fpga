#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorFactories.h>
#include <c10/util/Exception.h>
#include <ATen/Backend.h>
#include <ATen/Utils.h>
#include <ATen/native/opencl/Resize.h>

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

static void pointwise_op3(StorageImpl* c, const StorageImpl* a, const StorageImpl* b, at::native::opencl::OpenCLOperationsPointwise3 op) {
  // DONE Call OpenCL kernel.
  auto opt_kernel = c10::opencl::opencl_kernel("pointwise_op_3f");
  if (!opt_kernel.has_value()) {
    TORCH_WARN("No value for kernel \"pointwise_op_3f\"");
    return;
  }
  cl::Kernel pointwise_op_f = opt_kernel.value();
  pointwise_op_f.setArg<cl_mem>(0, (*(cl::Buffer*)a->data_ptr().get())());
  pointwise_op_f.setArg<cl_mem>(1, (*(cl::Buffer*)b->data_ptr().get())());
  pointwise_op_f.setArg<cl_mem>(2, (*(cl::Buffer*)c->data_ptr().get())());
  pointwise_op_f.setArg<at::native::opencl::OpenCLOperationsPointwise3>(3, op);
  auto stream = caffe2::opencl::getCurrentOpenCLStream(a->device().index());
  cl::Event event;
  stream.stream()->enqueueNDRangeKernel(pointwise_op_f, 0, a->numel(), cl::NullRange, NULL, &event);
  event.wait();
}

static void pointwise_op2s(StorageImpl* c, const StorageImpl* a, const Scalar b, at::native::opencl::OpenCLOperationsPointwise3 op) {
  // DONE Call OpenCL kernel.
  auto opt_kernel = c10::opencl::opencl_kernel("pointwise_op_2fv");
  if (!opt_kernel.has_value()) {
    TORCH_WARN("No value for kernel \"pointwise_op_2fv\"");
    return;
  }
  cl::Kernel pointwise_op_f = opt_kernel.value();
  pointwise_op_f.setArg<cl_mem>(0, (*(cl::Buffer*)a->data_ptr().get())());
  pointwise_op_f.setArg<float>(1, b.toFloat());
  pointwise_op_f.setArg<cl_mem>(2, (*(cl::Buffer*)c->data_ptr().get())());
  pointwise_op_f.setArg<at::native::opencl::OpenCLOperationsPointwise3>(3, op);
  auto stream = caffe2::opencl::getCurrentOpenCLStream(a->device().index());
  cl::Event event;
  stream.stream()->enqueueNDRangeKernel(pointwise_op_f, 0, a->numel(), cl::NullRange, NULL, &event);
  event.wait();
}

static void pointwise_op(StorageImpl* b, const StorageImpl* a, at::native::opencl::OpenCLOperationsPointwise op) {
  // DONE Call OpenCL kernel.
  auto opt_kernel = c10::opencl::opencl_kernel("pointwise_op_f");
  if (!opt_kernel.has_value()) {
    TORCH_WARN("No value for kernel \"pointwise_op_f\"");
    return;
  }
  cl::Kernel pointwise_op_f = opt_kernel.value();
  pointwise_op_f.setArg<cl_mem>(0, (*(cl::Buffer*)a->data_ptr().get())());
  pointwise_op_f.setArg<cl_mem>(1, (*(cl::Buffer*)b->data_ptr().get())());
  pointwise_op_f.setArg<at::native::opencl::OpenCLOperationsPointwise>(2, op);
  auto stream = caffe2::opencl::getCurrentOpenCLStream(a->device().index());
  cl::Event event;
  stream.stream()->enqueueNDRangeKernel(pointwise_op_f, 0, a->numel(), cl::NullRange, NULL, &event);
  event.wait();
}

// See THC_logicalTensor in aten/src/THC/THCTensorMathCompareT.cuh for implementation details
static void logical_tensor(TensorImpl *self_, const TensorImpl *t1, const TensorImpl *t2, at::native::opencl::OpenCLOperationsPointwise3 op) {
  opencl_resize(self_, t1->sizes(), {});
  TORCH_CHECK(opencl_nElement(t1) == opencl_nElement(t2), "sizes don't match");
  pointwise_op3(self_->storage().unsafeGetStorageImpl(), t1->storage().unsafeGetStorageImpl(), t2->storage().unsafeGetStorageImpl(), op);
}

static void logical_tensor(TensorImpl *self_, const TensorImpl *t1, const Scalar t2, at::native::opencl::OpenCLOperationsPointwise3 op) {
  opencl_resize(self_, t1->sizes(), {});
  pointwise_op2s(self_->storage().unsafeGetStorageImpl(), t1->storage().unsafeGetStorageImpl(), t2, op);
}

Tensor _eq_opencl(const Tensor &self, const Tensor& other) {
  // TODO Implement this function for every scalar_type
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_eq_opencl", false, c10::Backend::OpenCL, ScalarType::Float);
  auto other_ = checked_tensor_unwrap(other, "other", 2, "_eq_opencl", false, c10::Backend::OpenCL, ScalarType::Float);
  logical_tensor(result_, self_, other_, at::native::opencl::OpenCLOperationsPointwise3::EQ);
  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  return result;
}

Tensor _eq_opencl(const Tensor &self, Scalar other) {
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_eq_opencl", false, c10::Backend::OpenCL, ScalarType::Float);
  logical_tensor(result_, self_, other, at::native::opencl::OpenCLOperationsPointwise3::EQ);
  result_->maybe_zero_dim(self_->dim() == 0);
  return result;
}

Tensor _ne_opencl(const Tensor &self, const Tensor& other) {
  // TODO Implement this function for every scalar_type
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_ne_opencl", false, c10::Backend::OpenCL, ScalarType::Float);
  auto other_ = checked_tensor_unwrap(other, "other", 2, "_ne_opencl", false, c10::Backend::OpenCL, ScalarType::Float);
  logical_tensor(result_, self_, other_, at::native::opencl::OpenCLOperationsPointwise3::NE);
  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  return result;
}

Tensor _ne_opencl(const Tensor &self, Scalar other) {
  // TODO Implement this function for every scalar_type
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_ne_opencl", false, c10::Backend::OpenCL, ScalarType::Float);
  logical_tensor(result_, self_, other, at::native::opencl::OpenCLOperationsPointwise3::NE);
  result_->maybe_zero_dim(self_->dim() == 0);
  return result;
}

Tensor & _abs_out_opencl(Tensor &result, const Tensor &self) {
  auto result_ = checked_tensor_unwrap(result, "result", 1, "_abs_out_opencl", false, c10::Backend::OpenCL, ScalarType::Float);
  auto self_ = checked_tensor_unwrap(self, "self", 2, "_abs_out_opencl", false, c10::Backend::OpenCL, ScalarType::Float);

  opencl_resize(result_, self_->sizes(), {});
  pointwise_op(result_->storage().unsafeGetStorageImpl(), self_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise::ABS);

  return result;
}

}} // namespace at::native
