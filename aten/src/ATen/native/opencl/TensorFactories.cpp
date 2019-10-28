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

static cl::Buffer &toBuffer(const StorageImpl* s) {
  return *toBuffer(s->data_ptr().get());
}

static void pointwise_op3(StorageImpl* c, const StorageImpl* a, const StorageImpl* b, at::native::opencl::OpenCLOperationsPointwise3 op, const ScalarType scalar_type) {
  // DONE Call OpenCL kernel.
  auto kernel_name = "pointwise_op_3";
  auto stream = at::opencl::getCurrentOpenCLStream(a->device().index());
  auto pointwise_op = c10::opencl::opencl_kernel_func<OpenCLPointwise3Functor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange{(size_t)a->numel()}, 1});
  AT_OPENCL_CHECK(pointwise_op(
      toBuffer(a),
      toBuffer(b),
      toBuffer(c),
      op,
      getOpenCLKernelCastType(scalar_type)));
  AT_OPENCL_CHECK(syncOpenCLPointer(c->data_ptr().get()));
  AT_OPENCL_CHECK(stream.stream()->finish());
}

template <c10::ScalarType T, typename S = decltype(c10::impl::ScalarTypeToCPPType<T>::t)>
static void pointwise_op2_s(StorageImpl* c, const StorageImpl* a, const Scalar b, at::native::opencl::OpenCLOperationsPointwise3 op) {
  auto kernel_name = "pointwise_op_2s";
  auto stream = at::opencl::getCurrentOpenCLStream(a->device().index());
  auto pointwise_op = c10::opencl::opencl_kernel_func<OpenCLPointwise3Functor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange{(size_t)a->numel()}, 1});

  Tensor scalar_tensor = at::native::empty_opencl({1}, TensorOptions{T}.merge_in({a->device()}));
  auto scalar_tensor_ = scalar_tensor.storage().unsafeGetStorageImpl();
  S value_s = b.to<S>();
  AT_OPENCL_CHECK(stream.stream()->enqueueWriteBuffer(*toBuffer(scalar_tensor_->data()), CL_TRUE, 0, sizeof(S), &value_s));

  AT_OPENCL_CHECK(pointwise_op(
      toBuffer(a),
      toBuffer(scalar_tensor_),
      toBuffer(c),
      op,
      getOpenCLKernelCastType(T)));
  AT_OPENCL_CHECK(syncOpenCLPointer(c->data_ptr().get()));
  AT_OPENCL_CHECK(stream.stream()->finish());
}

static void pointwise_op2(StorageImpl* b, const StorageImpl* a, at::native::opencl::OpenCLOperationsPointwise2 op, const ScalarType scalar_type) {
  // DONE Call OpenCL kernel.
  auto kernel_name = "pointwise_op_2";
  auto stream = at::opencl::getCurrentOpenCLStream(a->device().index());
  auto pointwise_op = c10::opencl::opencl_kernel_func<OpenCLPointwise2Functor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange{(size_t)a->numel()}, 1});
  AT_OPENCL_CHECK(pointwise_op(
      toBuffer(a),
      toBuffer(b),
      op,
      getOpenCLKernelCastType(scalar_type)));
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

  TORCH_CHECK(isIntegralType(self.scalar_type(), true) && isIntegralType(other.scalar_type(), true), "_and_opencl operation is undefined for floating point scalar types (", self.scalar_type(), ", ", other.scalar_type(), ").");

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

  TORCH_CHECK(isIntegralType(self.scalar_type(), true), "_and_opencl operation is undefined for floating point scalar types (", self.scalar_type(), ").");

  opencl_resize(result_, self_->sizes(), {});
  TORCH_CHECK(opencl_nElement(result_) == opencl_nElement(self_), "sizes don't match");
  auto scalar_type = self.scalar_type();
  switch (scalar_type)
  {
#define DEFINE_OPENCL_AND_CASE(type, name) \
    case ScalarType::name: \
      pointwise_op2_s<ScalarType::name, type>(result_->storage().unsafeGetStorageImpl(), self_->storage().unsafeGetStorageImpl(), other, at::native::opencl::OpenCLOperationsPointwise3::BAND); \
      break;
    AT_FORALL_SCALAR_TYPES_AND(Bool, DEFINE_OPENCL_AND_CASE)
#undef DEFINE_OPENCL_AND_CASE

  default:
    TORCH_CHECK(false, "_and_opencl not supported on OpenCLType for ", scalar_type);
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

  auto stream = at::opencl::getCurrentOpenCLStream(self_->device().index());
  auto kernel_name = "cast_s";
  auto kernel = c10::opencl::opencl_kernel_func<OpenCLCastFunctor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange{(size_t)self_->numel()}, 1});

  Tensor scalar_tensor = at::native::empty({1}, c10::nullopt, self.options().merge_in(TensorOptions{ScalarType::Int}));
  int value_0 = 0;
  AT_OPENCL_CHECK(stream.stream()->enqueueWriteBuffer(*toBuffer(scalar_tensor.data_ptr()), CL_TRUE, 0, sizeof(int), &value_0));
  AT_OPENCL_CHECK(kernel(
    *toBuffer(scalar_tensor.data_ptr()),
    *toBuffer(self_->data()),
    at::native::opencl::OpenCLPtrType::INT,
    getOpenCLKernelCastType(typeMetaToScalarType(self_->dtype()))));
  AT_OPENCL_CHECK(syncOpenCLPointer(self_->data()));

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
  return self.toBackend(Backend::CPU).max().toBackend(Backend::OpenCL);
}

Tensor empty_strided_opencl(IntArrayRef size, IntArrayRef stride, const TensorOptions& options) {
  auto t = at::native::empty_opencl({0}, options);
  at::native::resize_impl_opencl_(t.unsafeGetTensorImpl(), size, stride);
  return t;
}

Tensor & _opencl_set_(Tensor & self, Storage source) {
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_opencl_set_", false, Backend::OpenCL, self.scalar_type());
  auto source_ = checked_storage(source, "source", 2, DeviceType::OPENCL, at::scalarTypeToTypeMeta(self.scalar_type()));
  at::IntArrayRef size_{static_cast<int64_t>(source.size())};
  at::IntArrayRef stride_{};
  opencl_setStorageNd(self_, source_.unsafeGetStorageImpl(), 0, size_.size(), size_.data(), stride_.data());
  self_->maybe_zero_dim(false);
  return self;
}

Tensor & _opencl_set_(Tensor & self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_opencl_set_", false, Backend::OpenCL, self.scalar_type());
  auto source_ = checked_storage(source, "source", 2, DeviceType::OPENCL, at::scalarTypeToTypeMeta(self.scalar_type()));
  opencl_setStorageNd(self_, source_.unsafeGetStorageImpl(), 0, size.size(), size.data(), stride.data());
  self_->maybe_zero_dim(false);
  return self;
}

Tensor & _opencl_set_(Tensor &self) {
  at::IntArrayRef size_{0};
  at::IntArrayRef stride_{};
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_opencl_set_", false, c10::Backend::OpenCL, self.scalar_type());
  opencl_setStorageNd(self_,
                         NULL,
                         0,
                         size_.size(),
                         size_.data(),
                         stride_.data());
  self_->maybe_zero_dim(false);
  return self;
}

Tensor & _opencl_set_(Tensor &self, Tensor &src) {
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_opencl_set_", false, c10::Backend::OpenCL, self.scalar_type());
  auto src_ = checked_tensor_unwrap(src, "src", 2, "_opencl_set_", false, c10::Backend::OpenCL, self.scalar_type());
  if(self_ != src_) {
    opencl_setStorageNd(self_,
                           src_->storage().unsafeGetStorageImpl(),
                           src_->storage_offset(),
                           src_->dim(),
                           src_->sizes().data(),
                           src_->strides().data());
  }
  self_->maybe_zero_dim(src_->dim() == 0);
  return self;
}

Tensor _opencl_cat(TensorList tensors, int64_t dim) {
  std::vector<Tensor> cpuTensors;
  std::transform(tensors.begin(), tensors.end(), std::back_inserter(cpuTensors), [](const Tensor& t) -> Tensor {return t.toBackend(Backend::CPU);});
  return at::native::cat(cpuTensors, dim).toBackend(Backend::OpenCL);
}

Tensor & _opencl_cat_out(Tensor & self, TensorList tensors, int64_t dim) {
  std::vector<Tensor> cpuTensors;
  std::transform(tensors.begin(), tensors.end(), std::back_inserter(cpuTensors), [](const Tensor& t) -> Tensor {return t.toBackend(Backend::CPU);});
  auto self2 = self.toBackend(Backend::CPU);
  at::native::cat_out(self2, cpuTensors, dim);
  self = self2.toBackend(Backend::OpenCL);
  return self;
}

Tensor & _opencl_remainder_out(Tensor & result, const Tensor & self, Scalar other) {
  auto scalar_type = self.scalar_type();
  auto result_ = checked_tensor_unwrap(result, "result", 0, "_opencl_remainder", false, Backend::OpenCL, scalar_type);
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_opencl_remainder", false, Backend::OpenCL, scalar_type);

  // The implementation applies fmod to the floating point types.
  //TORCH_CHECK(isIntegralType(self.scalar_type(), true), "Remainder only applies to integral types");

  opencl_resizeAs(result_, self_);
  TORCH_CHECK(opencl_nElement(result_) == opencl_nElement(self_), "sizes don't match");
  switch (scalar_type)
  {
#define DEFINE_OPENCL_AND_CASE(type, name) \
    case ScalarType::name: \
      pointwise_op2_s<ScalarType::name, type>(result_->storage().unsafeGetStorageImpl(), self_->storage().unsafeGetStorageImpl(), other, at::native::opencl::OpenCLOperationsPointwise3::REM); \
      break;
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_OPENCL_AND_CASE)
#undef DEFINE_OPENCL_AND_CASE

  default:
    TORCH_CHECK(false, "_opencl_remainder_out not supported on OpenCLType for ", scalar_type);
    break;
  }

  result_->maybe_zero_dim(self_->dim() == 0);
  return result;
}

Tensor _opencl_remainder(const Tensor & self, Scalar other) {
  auto scalar_type = self.scalar_type();
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(scalar_type), 0, self.storage().allocator(), true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_opencl_remainder", false, Backend::OpenCL, scalar_type);

  // The implementation applies fmod to the floating point types.
  //TORCH_CHECK(isIntegralType(scalar_type, true), "Remainder only applies to integral types");

  opencl_resizeAs(result_, self_);
  TORCH_CHECK(opencl_nElement(result_) == opencl_nElement(self_), "sizes don't match");
  switch (scalar_type)
  {
#define DEFINE_OPENCL_AND_CASE(type, name) \
    case ScalarType::name: \
      pointwise_op2_s<ScalarType::name, type>(result_->storage().unsafeGetStorageImpl(), self_->storage().unsafeGetStorageImpl(), other, at::native::opencl::OpenCLOperationsPointwise3::REM); \
      break;
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_OPENCL_AND_CASE)
#undef DEFINE_OPENCL_AND_CASE

  default:
    TORCH_CHECK(false, "_opencl_remainder not supported on OpenCLType for ", scalar_type);
    break;
  }

  result_->maybe_zero_dim(self_->dim() == 0);
  return result;
}

Tensor & _opencl_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) {
  auto scalar_type = self.scalar_type();
  auto result_ = checked_tensor_unwrap(result, "result", 0, "_opencl_remainder_out", false, Backend::OpenCL, scalar_type);
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_opencl_remainder_out", false, Backend::OpenCL, scalar_type);
  auto other_ = checked_tensor_unwrap(other, "other", 2, "_opencl_remainder_out", false, Backend::OpenCL, scalar_type);

  TORCH_CHECK(opencl_nElement(result_) == opencl_nElement(self_), "sizes don't match");
  opencl_resizeAs(result_, self_);

  pointwise_op3(result_->storage().unsafeGetStorageImpl(), self_->storage().unsafeGetStorageImpl(), other_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::REM, scalar_type);

  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  return result;
}

Tensor _opencl_remainder(const Tensor & self, const Tensor & other) {
  auto scalar_type = self.scalar_type();
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(scalar_type), 0, self.storage().allocator(), true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_opencl_remainder", false, Backend::OpenCL, scalar_type);
  auto other_ = checked_tensor_unwrap(self, "other", 2, "_opencl_remainder", false, Backend::OpenCL, scalar_type);

  // The implementation applies fmod to the floating point types.
  //TORCH_CHECK(isIntegralType(self.scalar_type(), true), "Remainder only applies to integral types");

  TORCH_CHECK(opencl_nElement(result_) == opencl_nElement(self_), "sizes don't match");
  opencl_resizeAs(result_, self_);

  pointwise_op3(result_->storage().unsafeGetStorageImpl(), self_->storage().unsafeGetStorageImpl(), other_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::REM, scalar_type);

  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  return result;
}

}} // namespace at::native
