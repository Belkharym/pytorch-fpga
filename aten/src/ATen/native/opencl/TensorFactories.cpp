#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorFactories.h>
#include <c10/util/Exception.h>
#include <ATen/Backend.h>
#include <ATen/Utils.h>
#include <ATen/native/opencl/Resize.h>
#include <ATen/NamedTensorUtils.h>

#include <c10/opencl/OpenCLFunctions.h>
#include <aten/src/ATen/native/opencl/OpenCLOperations.h>
#include <aten/src/ATen/native/opencl/Utils.h>

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

static ScalarType getIntEquivalentOfFloat(const ScalarType type) {
  switch (type)
  {
  case ScalarType::Float:  // passthrough
  case ScalarType::QInt32: // passthrough
  case ScalarType::Int:
    return ScalarType::Int;
    break;
  case ScalarType::Double: // passthrough
  case ScalarType::Long:
    return ScalarType::Long;
    break;
  case ScalarType::Half:   // passthrough
  case ScalarType::BFloat16: // passthrough
  case ScalarType::Short:
    return ScalarType::Short;
    break;
  case ScalarType::QInt8:  // passthrough
  case ScalarType::QUInt8: // passthrough
  case ScalarType::Byte:
    return ScalarType::Byte;
    break;
  case ScalarType::Bool:
    return ScalarType::Bool;
    break;
  
  default:
    return ScalarType::Undefined;
    break;
  }
}

static void pointwise_op3(StorageImpl* c, const StorageImpl* a, const StorageImpl* b, at::native::opencl::OpenCLOperationsPointwise3 op, const ScalarType scalar_type) {
  // DONE Call OpenCL kernel.
  auto kernel_name = "pointwise_op_3" + getOpenCLKernelTypeSuffix(scalar_type);
  auto opt_kernel = c10::opencl::opencl_kernel(kernel_name);
  if (!opt_kernel) {
    TORCH_WARN("No value for kernel \"", kernel_name, "\"");
    return;
  }
  cl::Kernel pointwise_op = opt_kernel.value();
  pointwise_op.setArg<cl_mem>(0, (*(cl::Buffer*)a->data_ptr().get())());
  pointwise_op.setArg<cl_mem>(1, (*(cl::Buffer*)b->data_ptr().get())());
  pointwise_op.setArg<cl_mem>(2, (*(cl::Buffer*)c->data_ptr().get())());
  pointwise_op.setArg<at::native::opencl::OpenCLOperationsPointwise3>(3, op);
  auto stream = caffe2::opencl::getCurrentOpenCLStream(a->device().index());
  stream.stream()->enqueueNDRangeKernel(pointwise_op, 0, a->numel(), 1);
  stream.stream()->finish();
}

template <c10::ScalarType T, typename S = decltype(c10::impl::ScalarTypeToCPPType<T>::t)>
static void pointwise_op2_s(StorageImpl* c, const StorageImpl* a, const Scalar b, at::native::opencl::OpenCLOperationsPointwise3 op) {
  // DONE Call OpenCL kernel.
  auto kernel_name = "pointwise_op_2" + getOpenCLKernelTypeSuffix(T) + "_s";
  auto opt_kernel = c10::opencl::opencl_kernel(kernel_name);
  if (!opt_kernel) {
    TORCH_WARN("No value for kernel \"", kernel_name, "\"");
    return;
  }
  cl::Kernel pointwise_op = opt_kernel.value();
  pointwise_op.setArg<cl_mem>(0, (*(cl::Buffer*)a->data_ptr().get())());
  pointwise_op.setArg<S>(1, b.to<S>());
  pointwise_op.setArg<cl_mem>(2, (*(cl::Buffer*)c->data_ptr().get())());
  pointwise_op.setArg<at::native::opencl::OpenCLOperationsPointwise3>(3, op);
  auto stream = caffe2::opencl::getCurrentOpenCLStream(a->device().index());
  stream.stream()->enqueueNDRangeKernel(pointwise_op, 0, a->numel(), 1);
  stream.stream()->finish();
}

static void pointwise_op(StorageImpl* b, const StorageImpl* a, at::native::opencl::OpenCLOperationsPointwise op, const ScalarType scalar_type) {
  // DONE Call OpenCL kernel.
  auto kernel_name = "pointwise_op_" + getOpenCLKernelTypeSuffix(scalar_type);
  auto opt_kernel = c10::opencl::opencl_kernel(kernel_name);
  if (!opt_kernel) {
    TORCH_WARN("No value for kernel \"", kernel_name, "\"");
    return;
  }
  cl::Kernel pointwise_op = opt_kernel.value();
  pointwise_op.setArg<cl_mem>(0, (*(cl::Buffer*)a->data_ptr().get())());
  pointwise_op.setArg<cl_mem>(1, (*(cl::Buffer*)b->data_ptr().get())());
  pointwise_op.setArg<opencl::OpenCLOperationsPointwise>(2, op);
  auto stream = caffe2::opencl::getCurrentOpenCLStream(a->device().index());
  stream.stream()->enqueueNDRangeKernel(pointwise_op, /*offset=*/0, a->numel(), 1);
  stream.stream()->finish();
}

// See THC_logicalTensor in aten/src/THC/THCTensorMathCompareT.cuh for implementation details
static void logical_tensor(TensorImpl *self_, const TensorImpl *t1, const TensorImpl *t2, opencl::OpenCLOperationsPointwise3 op) {
  opencl_resize(self_, t1->sizes(), {});
  TORCH_CHECK(opencl_nElement(t1) == opencl_nElement(t2), "sizes don't match");
  // TORCH_CHECK(!(op == opencl::OpenCLOperationsPointwise3::BAND && (
  //                 isFloatingType(typeMetaToScalarType(t1->dtype())) ||
  //                 isFloatingType(typeMetaToScalarType(t2->dtype()))
  //             )), "BitWise operation not supported on floating point types.");
  pointwise_op3(self_->storage().unsafeGetStorageImpl(), t1->storage().unsafeGetStorageImpl(), t2->storage().unsafeGetStorageImpl(), op, typeMetaToScalarType(t1->dtype()));
}

static void logical_tensor(TensorImpl *self_, const TensorImpl *t1, const Scalar t2, opencl::OpenCLOperationsPointwise3 op) {
  opencl_resize(self_, t1->sizes(), {});
  auto scalar_type = typeMetaToScalarType(t1->dtype());
  switch (scalar_type)
  {
#define DEFINE_OPENCL_LOGICAL_TENSOR_CASE(type, name) \
  case ScalarType::name: \
    pointwise_op2_s<ScalarType::name, type>(self_->storage().unsafeGetStorageImpl(), t1->storage().unsafeGetStorageImpl(), t2, op); \
    break;
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_OPENCL_LOGICAL_TENSOR_CASE)
#undef DEFINE_OPENCL_LOGICAL_TENSOR_CASE

  default:
    TORCH_CHECK(false, "logical_tensor not supported on OpenCLType for ", scalar_type);
    break;
  }
}

Tensor _eq_opencl(const Tensor &self, const Tensor& other) {
  // TODO Implement this function for every scalar_type
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_eq_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  auto other_ = checked_tensor_unwrap(other, "other", 2, "_eq_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  logical_tensor(result_, self_, other_, at::native::opencl::OpenCLOperationsPointwise3::EQ);
  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  result.to(getIntEquivalentOfFloat(result.scalar_type()));
  return result;
}

Tensor _eq_opencl(const Tensor &self, Scalar other) {
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_eq_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  logical_tensor(result_, self_, other, at::native::opencl::OpenCLOperationsPointwise3::EQ);
  result_->maybe_zero_dim(self_->dim() == 0);
  result.to(getIntEquivalentOfFloat(result.scalar_type()));
  return result;
}

Tensor _ne_opencl(const Tensor &self, const Tensor& other) {
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_ne_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  auto other_ = checked_tensor_unwrap(other, "other", 2, "_ne_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  logical_tensor(result_, self_, other_, at::native::opencl::OpenCLOperationsPointwise3::NE);
  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  result.to(getIntEquivalentOfFloat(result.scalar_type()));
  return result;
}

Tensor _ne_opencl(const Tensor &self, Scalar other) {
  // TODO Implement this function for every scalar_type
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_ne_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  logical_tensor(result_, self_, other, at::native::opencl::OpenCLOperationsPointwise3::NE);
  result_->maybe_zero_dim(self_->dim() == 0);
  result.to(getIntEquivalentOfFloat(result.scalar_type()));
  return result;
}

Tensor & _abs_out_opencl(Tensor &result, const Tensor &self) {
  auto result_ = checked_tensor_unwrap(result, "result", 1, "_abs_out_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  auto self_ = checked_tensor_unwrap(self, "self", 2, "_abs_out_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  opencl_resize(result_, self_->sizes(), {});
  pointwise_op(result_->storage().unsafeGetStorageImpl(), self_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise::ABS, self.scalar_type());

  return result;
}

Tensor _and_opencl(const Tensor & self, const Tensor & other) {
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_and_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  auto other_ = checked_tensor_unwrap(other, "other", 2, "_and_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  logical_tensor(result_, self_, other_, at::native::opencl::OpenCLOperationsPointwise3::BAND);
  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  result.to(getIntEquivalentOfFloat(result.scalar_type()));
  return result;
}

Tensor _and_opencl(const Tensor & self, Scalar other) {
  // TODO Implement this function for every scalar_type
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_and_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  logical_tensor(result_, self_, other, at::native::opencl::OpenCLOperationsPointwise3::BAND);
  result_->maybe_zero_dim(self_->dim() == 0);
  result.to(getIntEquivalentOfFloat(result.scalar_type()));
  return result;
}

Tensor masked_select_opencl(const Tensor & self, const Tensor & mask) {
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::CPUTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  masked_select_cpu(self.toBackend(Backend::CPU), mask.toBackend(Backend::CPU));
  return result.toBackend(Backend::OpenCL);
}

Tensor & _ceil_out_opencl(Tensor &out, const Tensor &self) {
  auto result_ = checked_tensor_unwrap(out, "out", 1, "_ceil_out_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  auto self_ = checked_tensor_unwrap(self, "self", 2, "_ceil_out_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  opencl_resize(result_, self_->sizes(), {});
  pointwise_op(result_->storage().unsafeGetStorageImpl(), self_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise::CEIL, self.scalar_type());

  return out;
}

}} // namespace at::native
