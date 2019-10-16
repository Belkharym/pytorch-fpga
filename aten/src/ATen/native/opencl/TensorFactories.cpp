#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorFactories.h>
#include <c10/util/Exception.h>
#include <ATen/Backend.h>
#include <ATen/Utils.h>
#include <ATen/native/opencl/Resize.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/LegacyTHFunctionsCPU.h>

#include <c10/opencl/OpenCLFunctions.h>
#include <aten/src/ATen/opencl/Exceptions.h>
#include <aten/src/ATen/opencl/OpenCLContext.h>
#include <aten/src/ATen/native/opencl/OpenCLOperations.h>
#include <aten/src/ATen/native/opencl/Utils.h>

namespace at {
namespace native {

Tensor empty_opencl(IntArrayRef size, const TensorOptions& options, c10::optional<MemoryFormat> optional_memory_format) {
  TORCH_INTERNAL_ASSERT(options.backend() == at::Backend::OpenCL);
  TORCH_INTERNAL_ASSERT(!options.is_variable());  // is_variable should have been 'unpacked'  // TODO: remove this when Variable and Tensor are merged
  TORCH_CHECK(!options.pinned_memory(), "Only dense CPU tensors can be pinned");
  check_size_nonnegative(size);

  auto* allocator = at::opencl::getOpenCLDeviceAllocator();
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

static void pointwise_op3(StorageImpl* c, const StorageImpl* a, const StorageImpl* b, at::native::opencl::OpenCLOperationsPointwise3 op, const ScalarType scalar_type) {
  // DONE Call OpenCL kernel.
  auto kernel_name = "pointwise_op_3" + getOpenCLKernelTypeSuffix(scalar_type);
  auto opt_kernel = c10::opencl::opencl_kernel(kernel_name);
  TORCH_INTERNAL_ASSERT(opt_kernel.has_value(), "No value for kernel \"", kernel_name, "\"");
  cl::Kernel pointwise_op = opt_kernel.value();
  AT_OPENCL_CHECK(pointwise_op.setArg<cl_mem>(0, (*toBuffer(a->data_ptr().get()))()));
  AT_OPENCL_CHECK(pointwise_op.setArg<cl_mem>(1, (*toBuffer(b->data_ptr().get()))()));
  AT_OPENCL_CHECK(pointwise_op.setArg<cl_mem>(2, (*toBuffer(c->data_ptr().get()))()));
  AT_OPENCL_CHECK(pointwise_op.setArg<at::native::opencl::OpenCLOperationsPointwise3>(3, op));
  auto stream = at::opencl::getCurrentOpenCLStream(a->device().index());
  AT_OPENCL_CHECK(stream.stream()->enqueueNDRangeKernel(pointwise_op, 0, a->numel(), 1));
  AT_OPENCL_CHECK(syncOpenCLPointer(c->data_ptr().get()));
  AT_OPENCL_CHECK(stream.stream()->finish());
}

template <c10::ScalarType T, typename S = decltype(c10::impl::ScalarTypeToCPPType<T>::t)>
static void pointwise_op2_s(StorageImpl* c, const StorageImpl* a, const Scalar b, at::native::opencl::OpenCLOperationsPointwise3 op) {
  // DONE Call OpenCL kernel.
  auto kernel_name = "pointwise_op_2" + getOpenCLKernelTypeSuffix(T) + "_s";
  auto opt_kernel = c10::opencl::opencl_kernel(kernel_name);
  TORCH_INTERNAL_ASSERT(opt_kernel.has_value(), "No value for kernel \"", kernel_name, "\"");
  cl::Kernel pointwise_op = opt_kernel.value();
  AT_OPENCL_CHECK(pointwise_op.setArg<cl_mem>(0, (*toBuffer(a->data_ptr().get()))()));
  auto scalar = b.to<S>();
  AT_OPENCL_CHECK(pointwise_op.setArg<S>(1, scalar));
  AT_OPENCL_CHECK(pointwise_op.setArg<cl_mem>(2, (*toBuffer(c->data_ptr().get()))()));
  AT_OPENCL_CHECK(pointwise_op.setArg<at::native::opencl::OpenCLOperationsPointwise3>(3, op));
  auto stream = at::opencl::getCurrentOpenCLStream(a->device().index());
  AT_OPENCL_CHECK(stream.stream()->enqueueNDRangeKernel(pointwise_op, 0, a->numel(), 1));
  AT_OPENCL_CHECK(syncOpenCLPointer(c->data_ptr().get()));
  AT_OPENCL_CHECK(stream.stream()->finish());
}

static void pointwise_op2(StorageImpl* b, const StorageImpl* a, at::native::opencl::OpenCLOperationsPointwise2 op, const ScalarType scalar_type) {
  // DONE Call OpenCL kernel.
  auto kernel_name = "pointwise_op_2" + getOpenCLKernelTypeSuffix(scalar_type);
  auto opt_kernel = c10::opencl::opencl_kernel(kernel_name);
  TORCH_INTERNAL_ASSERT(opt_kernel.has_value(), "No value for kernel \"", kernel_name, "\"");
  cl::Kernel pointwise_op = opt_kernel.value();
  AT_OPENCL_CHECK(pointwise_op.setArg<cl_mem>(0, (*toBuffer(a->data_ptr().get()))()));
  AT_OPENCL_CHECK(pointwise_op.setArg<cl_mem>(1, (*toBuffer(b->data_ptr().get()))()));
  AT_OPENCL_CHECK(pointwise_op.setArg<opencl::OpenCLOperationsPointwise2>(2, op));
  auto stream = at::opencl::getCurrentOpenCLStream(a->device().index());
  AT_OPENCL_CHECK(stream.stream()->enqueueNDRangeKernel(pointwise_op, /*offset=*/0, a->numel(), 1));
  AT_OPENCL_CHECK(syncOpenCLPointer(b->data_ptr().get()));
  AT_OPENCL_CHECK(stream.stream()->finish());
}

Tensor & _abs_out_opencl(Tensor &result, const Tensor &self) {
  auto result_ = checked_tensor_unwrap(result, "result", 1, "_abs_out_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  auto self_ = checked_tensor_unwrap(self, "self", 2, "_abs_out_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  opencl_resize(result_, self_->sizes(), {});
  TORCH_CHECK(opencl_nElement(result_) == opencl_nElement(self_), "sizes don't match");
  pointwise_op2(result_->storage().unsafeGetStorageImpl(), self_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise2::ABS, self.scalar_type());

  return result;
}

Tensor & _ceil_out_opencl(Tensor &out, const Tensor &self) {
  auto result_ = checked_tensor_unwrap(out, "out", 1, "_ceil_out_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  auto self_ = checked_tensor_unwrap(self, "self", 2, "_ceil_out_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  opencl_resize(result_, self_->sizes(), {});
  pointwise_op(result_->storage().unsafeGetStorageImpl(), self_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise::CEIL, self.scalar_type());

  return out;
}

Tensor _and_opencl(const Tensor & self, const Tensor & other) {
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_and_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  auto other_ = checked_tensor_unwrap(other, "other", 2, "_and_opencl", false, c10::Backend::OpenCL, self.scalar_type());

  opencl_resize(result_, self_->sizes(), {});
  TORCH_CHECK(opencl_nElement(result_) == opencl_nElement(self_), "sizes don't match");
  pointwise_op3(result_->storage().unsafeGetStorageImpl(), self_->storage().unsafeGetStorageImpl(), other_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::BAND, self.scalar_type());

  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  return result.to(getIntEquivalentOfFloat(result.scalar_type()));
}

Tensor _and_opencl(const Tensor & self, Scalar other) {
  // TODO Implement this function for every scalar_type
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_and_opencl", false, c10::Backend::OpenCL, self.scalar_type());

  opencl_resize(result_, self_->sizes(), {});
  TORCH_CHECK(opencl_nElement(result_) == opencl_nElement(self_), "sizes don't match");
  auto scalar_type = self.scalar_type();
  switch (scalar_type)
  {
#define DEFINE_OPENCL_AND_CASE(type, name) \
    case ScalarType::name: \
      pointwise_op2_s<ScalarType::name, type>(result_->storage().unsafeGetStorageImpl(), self_->storage().unsafeGetStorageImpl(), other, at::native::opencl::OpenCLOperationsPointwise3::BAND); \
      break;
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_OPENCL_AND_CASE)
#undef DEFINE_OPENCL_AND_CASE

  default:
    TORCH_CHECK(false, "logical_tensor not supported on OpenCLType for ", scalar_type);
    break;
  }

  result_->maybe_zero_dim(self_->dim() == 0);
  result.to(getIntEquivalentOfFloat(result.scalar_type()));
  return result;
}

Tensor masked_select_opencl(const Tensor & self, const Tensor & mask) {
  return masked_select_cpu(self.toBackend(Backend::CPU), mask.toBackend(Backend::CPU)).toBackend(Backend::OpenCL);
}

Tensor & _ceil_out_opencl(Tensor &out, const Tensor &self) {
  auto result_ = checked_tensor_unwrap(out, "out", 1, "_ceil_out_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  auto self_ = checked_tensor_unwrap(self, "self", 2, "_ceil_out_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  opencl_resize(result_, self_->sizes(), {});
  pointwise_op2(result_->storage().unsafeGetStorageImpl(), self_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise2::CEIL, self.scalar_type());

  return out;
}

Tensor & _zero_opencl(Tensor & self) {
  TensorImpl* self_ = checked_tensor_unwrap(self, "self", 2, "_zero_opencl", false, c10::Backend::OpenCL, self.scalar_type());
  if (self_->is_contiguous()) {
    auto stream = at::opencl::getCurrentOpenCLStream(self_->device().index());
    auto scalar_type = self.scalar_type();
    switch (scalar_type)
    {
#define DEFINE_OPENCL_AND_CASE(type, name) \
      case ScalarType::name: { \
        type pattern{0}; \
        AT_OPENCL_CHECK(stream.stream()->enqueueFillBuffer(*toBuffer(self_->data()), /*pattern=*/pattern, /*offset=*/0, self_->numel() * self_->itemsize())); \
        break; \
      }
      AT_FORALL_SCALAR_TYPES_AND2(Half, Bool, DEFINE_OPENCL_AND_CASE)
#undef DEFINE_OPENCL_AND_CASE
    default:
      TORCH_CHECK(false, "logical_tensor not supported on OpenCLType for ", scalar_type);
      break;
    }
  } else {
    auto kernel_name = "cast_" + getOpenCLKernelTypeSuffix(typeMetaToScalarType(self_->dtype())) + "_i_s";
    auto kernel_opt = c10::opencl::opencl_kernel(kernel_name);
    TORCH_INTERNAL_ASSERT(kernel_opt.has_value(), "Kernel not found \"", kernel_name, "\"");
    auto stream = at::opencl::getCurrentOpenCLStream(self_->device().index());
    auto kernel = kernel_opt.value();
<<<<<<< HEAD
    AT_OPENCL_CHECK(kernel.setArg<int>(0, 0));
    AT_OPENCL_CHECK(kernel.setArg<cl_mem>(1, (*(cl::Buffer*)self_->data())()));
=======
    AT_OPENCL_CHECK(stream.stream()->enqueueNDRangeKernel(kernel, /*offset=*/0, self_->numel(), 1));
    AT_OPENCL_CHECK(syncOpenCLPointer(self_->data()));
  }
  return self;

Tensor empty_strided_opencl(IntArrayRef size, IntArrayRef stride, const TensorOptions& options) {
  auto t = at::native::empty_opencl({0}, options);
  at::native::resize_impl_opencl_(t.unsafeGetTensorImpl(), size, stride);
  return t;
}

Tensor & _opencl_min_out(Tensor &result, const Tensor &self, const Tensor &other) {
  auto result_ = checked_tensor_unwrap(result, "result", 0, "_opencl_min_out", false, Backend::OpenCL, self.scalar_type());
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_opencl_min_out", false, Backend::OpenCL, self.scalar_type());
  auto other_ = checked_tensor_unwrap(other, "other", 2, "_opencl_min_out", false, Backend::OpenCL, self.scalar_type());
  AT_OPENCL_CHECK(opencl_nElement(self_) ==
                  opencl_nElement(other_), "sizes do not match");

  if (result_ != self_) {
    opencl_resizeAs(result_, self_);
  }
  pointwise_op3(result_->storage().unsafeGetStorageImpl(), self_->storage().unsafeGetStorageImpl(), other_->storage().unsafeGetStorageImpl(), opencl::OpenCLOperationsPointwise3::MIN, self.scalar_type());
  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  return result;
}

Tensor _opencl_min(const Tensor &self, const Tensor &other) {
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_opencl_min", false, c10::Backend::OpenCL, self.scalar_type());
  auto other_ = checked_tensor_unwrap(other, "other", 2, "_opencl_min", false, c10::Backend::OpenCL, self.scalar_type());

  opencl_resize(result_, self_->sizes(), {});
  TORCH_CHECK(opencl_nElement(result_) == opencl_nElement(self_), "sizes don't match");
  pointwise_op3(result_->storage().unsafeGetStorageImpl(), self_->storage().unsafeGetStorageImpl(), other_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::MIN, self.scalar_type());

  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  return result;
}

Tensor _opencl_min(const Tensor &self) {
  return at::native::legacy::cpu::_th_min(self.toBackend(Backend::CPU)).toBackend(Backend::OpenCL);
}

Tensor & _opencl_max_out(Tensor &result, const Tensor &self, const Tensor &other) {
  auto result_ = checked_tensor_unwrap(result, "result", 0, "_opencl_max_out", false, Backend::OpenCL, self.scalar_type());
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_opencl_max_out", false, Backend::OpenCL, self.scalar_type());
  auto other_ = checked_tensor_unwrap(other, "other", 2, "_opencl_max_out", false, Backend::OpenCL, self.scalar_type());
  AT_OPENCL_CHECK(opencl_nElement(self_) ==
                  opencl_nElement(other_), "sizes do not match");

  if (result_ != self_) {
    opencl_resizeAs(result_, self_);
  }
  pointwise_op3(result_->storage().unsafeGetStorageImpl(), self_->storage().unsafeGetStorageImpl(), other_->storage().unsafeGetStorageImpl(), opencl::OpenCLOperationsPointwise3::MAX, self.scalar_type());
  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  return result;
}

Tensor _opencl_max(const Tensor &self, const Tensor &other) {
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_opencl_max", false, c10::Backend::OpenCL, self.scalar_type());
  auto other_ = checked_tensor_unwrap(other, "other", 2, "_opencl_max", false, c10::Backend::OpenCL, self.scalar_type());

  opencl_resize(result_, self_->sizes(), {});
  TORCH_CHECK(opencl_nElement(result_) == opencl_nElement(self_), "sizes don't match");
  pointwise_op3(result_->storage().unsafeGetStorageImpl(), self_->storage().unsafeGetStorageImpl(), other_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::MAX, self.scalar_type());

  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  return result;
}

Tensor _opencl_max(const Tensor &self) {
  return at::native::legacy::cpu::_th_max(self.toBackend(Backend::CPU)).toBackend(Backend::OpenCL);
}

}} // namespace at::native
