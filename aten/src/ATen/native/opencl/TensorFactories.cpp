#include <ATen/ATen.h>

#include <aten/src/ATen/opencl/Exceptions.h> // This include must be before ATen/native/DispatchStub.h

#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>

#include <ATen/InitialTensorOptions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/util/Exception.h>
#include <ATen/Backend.h>
#include <ATen/Utils.h>
#include <ATen/native/opencl/Resize.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/LegacyTHFunctionsCPU.h>


#include <c10/opencl/OpenCLFunctions.h>
#include <aten/src/ATen/opencl/OpenCLContext.h>
#include <aten/src/ATen/native/opencl/OpenCLOperations.h>
#include <aten/src/ATen/native/opencl/Utils.h>


namespace at {
namespace native {

Tensor empty_opencl(IntArrayRef size, const TensorOptions& options, c10::optional<MemoryFormat> optional_memory_format) {
  AT_ASSERT(options.backend() == at::Backend::OpenCL);
  TORCH_INTERNAL_ASSERT(impl::variable_excluded_from_dispatch());
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

  auto&& tensor = at::detail::make_tensor<TensorImpl>(storage_impl, TensorTypeId::OpenCLTensorId);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  auto memory_format = optional_memory_format.value_or(MemoryFormat::Contiguous);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  return std::move(tensor);
}

Tensor& uniform_opencl_(Tensor& self, double from, double to, Generator* gen) {
  Backend backend = Backend::CPU;
  if (detail::getCUDAHooks().hasCUDA()) {
    backend = Backend::CUDA;
  }
  auto self_ = self.toBackend(backend);
  self.copy_(self_.uniform_(from, to, gen));
  return self;
}

Tensor& random_opencl_(Tensor& self, Generator* gen) {
  Backend backend = Backend::CPU;
  if (detail::getCUDAHooks().hasCUDA()) {
    backend = Backend::CUDA;
  }
  auto self_ = self.toBackend(backend);
  self.copy_(self_.random_(gen));
  return self;
}

Tensor& normal_opencl_(Tensor& self, double mean, double std, Generator* gen) {
  Backend backend = Backend::CPU;
  if (detail::getCUDAHooks().hasCUDA()) {
    backend = Backend::CUDA;
  }
  auto self_ = self.toBackend(backend);
  self.copy_(self_.normal_(mean, std, gen));
  return self;
}

static cl::Buffer &toBuffer(const Tensor& s) {
  return *toBuffer(s.data_ptr());
}

static void pointwise_op3(Tensor& c, const Tensor& a, const Tensor& b, at::native::opencl::OpenCLOperationsPointwise3 op, const ScalarType scalar_type) {
  // DONE Call OpenCL kernel.
  static const std::string kernel_name = "pointwise_op_3";
  auto stream = at::opencl::getCurrentOpenCLStream(a.device().index());
  auto pointwise_op = c10::opencl::opencl_kernel_func<OpenCLPointwise3Functor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange{(size_t)a.storage_offset()}, cl::NDRange{(size_t)a.numel()}, 1});
  AT_OPENCL_CHECK(pointwise_op(
      toBuffer(a),
      toBuffer(b),
      toBuffer(c),
      op,
      getOpenCLKernelCastType(scalar_type)));
  AT_OPENCL_CHECK(syncOpenCLPointer(c.data_ptr(), stream));
  stream.synchronize();
}

template <c10::ScalarType T, typename S = decltype(c10::impl::ScalarTypeToCPPType<T>::t)>
static void pointwise_op2_s(Tensor& c, const Tensor& a, const Scalar b, at::native::opencl::OpenCLOperationsPointwise3 op, bool invert = false) {
  static const std::string kernel_name = "pointwise_op_2s";
  auto stream = at::opencl::getCurrentOpenCLStream(a.device().index());
  auto pointwise_op = c10::opencl::opencl_kernel_func<OpenCLPointwise2sFunctor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange{(size_t)a.storage_offset()}, cl::NDRange{(size_t)a.numel()}, 1});

  auto scalar_buffer = at::native::scalar_buffer_opencl<S>(b, a.device().index());

  AT_OPENCL_CHECK(pointwise_op(
      toBuffer(a),
      toBuffer(scalar_buffer),
      toBuffer(c),
      op,
      getOpenCLKernelCastType(T),
      invert));
  AT_OPENCL_CHECK(syncOpenCLPointer(c.data_ptr(), stream));
  stream.synchronize();
}

static void pointwise_op2(Tensor& b, const Tensor& a, at::native::opencl::OpenCLOperationsPointwise2 op, const ScalarType scalar_type) {
  // DONE Call OpenCL kernel.
  static const std::string kernel_name = "pointwise_op_2";
  auto stream = at::opencl::getCurrentOpenCLStream(a.device().index());
  auto pointwise_op = c10::opencl::opencl_kernel_func<OpenCLPointwise2Functor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange{(size_t)a.storage_offset()}, cl::NDRange{(size_t)a.numel()}, 1});
  AT_OPENCL_CHECK(pointwise_op(
      toBuffer(a),
      toBuffer(b),
      op,
      getOpenCLKernelCastType(scalar_type)));
  AT_OPENCL_CHECK(syncOpenCLPointer(b.data_ptr(), stream));
  stream.synchronize();
}

static void abs_kernel_opencl(at::TensorIterator& iter) {
  pointwise_op2(iter.tensor(0), iter.tensor(1), at::native::opencl::OpenCLOperationsPointwise2::ABS, iter.common_dtype());
}

Tensor _and_opencl(const Tensor & self, const Tensor & other) {
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = self.unsafeGetTensorImpl();
  auto other_ = other.unsafeGetTensorImpl();

  // TORCH_CHECK(isIntegralType(self.scalar_type(), true) && isIntegralType(other.scalar_type(), true), "_and_opencl operation is undefined for floating point scalar types (", self.scalar_type(), ", ", other.scalar_type(), ").");

  opencl_resize(result_, self_->sizes(), {});
  TORCH_CHECK(opencl_nElement(*result_) == opencl_nElement(*self_), "sizes don't match");
  pointwise_op3(result, self, other, at::native::opencl::OpenCLOperationsPointwise3::BAND, self.scalar_type());

  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  return result.to(getIntEquivalentOfFloat(result.scalar_type()));
}

Tensor _and_opencl(const Tensor & self, Scalar other) {
  // TODO Implement this function for every scalar_type
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = self.unsafeGetTensorImpl();

  // TORCH_CHECK(isIntegralType(self.scalar_type(), true), "_and_opencl operation is undefined for floating point scalar types (", self.scalar_type(), ").");

  opencl_resize(result_, self_->sizes(), {});
  TORCH_CHECK(opencl_nElement(*result_) == opencl_nElement(*self_), "sizes don't match");
  auto scalar_type = self.scalar_type();
  switch (scalar_type)
  {
#define DEFINE_OPENCL_AND_CASE(type, name) \
    case ScalarType::name: \
      pointwise_op2_s<ScalarType::name, type>(result, self, other, at::native::opencl::OpenCLOperationsPointwise3::BAND); \
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

void ceil_kernel_opencl(TensorIterator &iter) {
  pointwise_op2(iter.tensor(0), iter.tensor(1), at::native::opencl::OpenCLOperationsPointwise2::CEIL, iter.common_dtype());
}

Tensor & _opencl_min_out(Tensor &result, const Tensor &self, const Tensor &other) {
  auto result_ = result.unsafeGetTensorImpl();
  auto self_ = self.unsafeGetTensorImpl();
  auto other_ = other.unsafeGetTensorImpl();
  // TORCH_CHECK(c10::isIntegralType(self.scalar_type(), true), "_opencl_min_out does not support non integral types");
  AT_OPENCL_CHECK(opencl_nElement(*self_) ==
                  opencl_nElement(*other_), "sizes do not match");

  if (result_ != self_) {
    opencl_resizeAs(result_, self_);
  }
  pointwise_op3(result, self, other, opencl::OpenCLOperationsPointwise3::MIN, self.scalar_type());
  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  return result;
}

Tensor _opencl_min(const Tensor &self, const Tensor &other) {
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = self.unsafeGetTensorImpl();
  auto other_ = other.unsafeGetTensorImpl();
  // TORCH_CHECK(c10::isIntegralType(self.scalar_type(), true), "_opencl_min does not support non integral types");

  opencl_resize(result_, self_->sizes(), {});
  TORCH_CHECK(opencl_nElement(*result_) == opencl_nElement(*self_), "sizes don't match");
  pointwise_op3(result, self, other, at::native::opencl::OpenCLOperationsPointwise3::MIN, self.scalar_type());

  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  return result;
}

Tensor _opencl_min(const Tensor &self) {
  return at::native::legacy::cpu::_th_min(self.toBackend(Backend::CPU)).toBackend(Backend::OpenCL);
}

Tensor & _opencl_max_out(Tensor &result, const Tensor &self, const Tensor &other) {
  auto result_ = result.unsafeGetTensorImpl();
  auto self_ = self.unsafeGetTensorImpl();
  auto other_ = other.unsafeGetTensorImpl();
  // TORCH_CHECK(c10::isIntegralType(self.scalar_type(), true), "_opencl_max_out does not support non integral types");
  AT_OPENCL_CHECK(opencl_nElement(*self_) ==
                  opencl_nElement(*other_), "sizes do not match");

  if (result_ != self_) {
    opencl_resizeAs(result_, self_);
  }
  pointwise_op3(result, self, other, opencl::OpenCLOperationsPointwise3::MAX, self.scalar_type());
  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  return result;
}

Tensor _opencl_max(const Tensor &self, const Tensor &other) {
  auto allocator = c10::GetAllocator(DeviceType::OPENCL);
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = self.unsafeGetTensorImpl();
  auto other_ = other.unsafeGetTensorImpl();
  // TORCH_CHECK(c10::isIntegralType(self.scalar_type(), true), "_opencl_max does not support non integral types");

  opencl_resize(result_, self_->sizes(), {});
  TORCH_CHECK(opencl_nElement(*result_) == opencl_nElement(*self_), "sizes don't match");
  pointwise_op3(result, self, other, at::native::opencl::OpenCLOperationsPointwise3::MAX, self.scalar_type());

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
  auto self_ = self.unsafeGetTensorImpl();
  auto source_ = checked_storage(source, "source", 2, DeviceType::OPENCL, at::scalarTypeToTypeMeta(self.scalar_type()));
  at::IntArrayRef size_{static_cast<int64_t>(source.size())};
  at::IntArrayRef stride_{};
  opencl_setStorageNd(*self_, source_.unsafeGetStorageImpl(), 0, size_.size(), size_.data(), stride_.data());
  self_->maybe_zero_dim(false);
  return self;
}

Tensor & _opencl_set_(Tensor & self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  auto self_ = self.unsafeGetTensorImpl();
  auto source_ = checked_storage(source, "source", 2, DeviceType::OPENCL, at::scalarTypeToTypeMeta(self.scalar_type()));
  opencl_setStorageNd(*self_, source_.unsafeGetStorageImpl(), 0, size.size(), size.data(), stride.data());
  self_->maybe_zero_dim(false);
  return self;
}

Tensor & _opencl_set_(Tensor &self) {
  at::IntArrayRef size_{0};
  at::IntArrayRef stride_{};
  auto self_ = self.unsafeGetTensorImpl();
  opencl_setStorageNd(*self_,
                         NULL,
                         0,
                         size_.size(),
                         size_.data(),
                         stride_.data());
  self_->maybe_zero_dim(false);
  return self;
}

Tensor & _opencl_set_(Tensor &self, Tensor &src) {
  auto self_ = self.unsafeGetTensorImpl();
  auto src_ = src.unsafeGetTensorImpl();
  if(self_ != src_) {
    opencl_setStorageNd(*self_,
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
  auto result_ = result.unsafeGetTensorImpl();
  auto self_ = self.unsafeGetTensorImpl();

  // The implementation applies fmod to the floating point types.
  //TORCH_CHECK(isIntegralType(self.scalar_type(), true), "Remainder only applies to integral types");

  opencl_resizeAs(result_, self_);
  TORCH_CHECK(opencl_nElement(*result_) == opencl_nElement(*self_), "sizes don't match");
  switch (scalar_type)
  {
#define DEFINE_OPENCL_AND_CASE(type, name) \
    case ScalarType::name: \
      pointwise_op2_s<ScalarType::name, type>(result, self, other, at::native::opencl::OpenCLOperationsPointwise3::REM); \
      break;
    AT_FORALL_SCALAR_TYPES(DEFINE_OPENCL_AND_CASE)
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
  auto self_ = self.unsafeGetTensorImpl();

  // The implementation applies fmod to the floating point types.
  //TORCH_CHECK(isIntegralType(scalar_type, true), "Remainder only applies to integral types");

  opencl_resizeAs(result_, self_);
  TORCH_CHECK(opencl_nElement(*result_) == opencl_nElement(*self_), "sizes don't match");
  switch (scalar_type)
  {
#define DEFINE_OPENCL_AND_CASE(type, name) \
    case ScalarType::name: \
      pointwise_op2_s<ScalarType::name, type>(result, self, other, at::native::opencl::OpenCLOperationsPointwise3::REM); \
      break;
    AT_FORALL_SCALAR_TYPES(DEFINE_OPENCL_AND_CASE)
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
  auto result_ = result.unsafeGetTensorImpl();
  auto self_ = self.unsafeGetTensorImpl();
  auto other_ = other.unsafeGetTensorImpl();
  // TORCH_CHECK(c10::isIntegralType(self.scalar_type(), true), "_opencl_remainder_out does not support non integral types");

  TORCH_CHECK(opencl_nElement(*result_) == opencl_nElement(*self_), "sizes don't match");
  opencl_resizeAs(result_, self_);

  pointwise_op3(result, self, other, at::native::opencl::OpenCLOperationsPointwise3::REM, scalar_type);

  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  return result;
}

Tensor _opencl_remainder(const Tensor & self, const Tensor & other) {
  auto scalar_type = self.scalar_type();
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(scalar_type), 0, self.storage().allocator(), true),TensorTypeId::OpenCLTensorId).release();
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
  auto self_ = self.unsafeGetTensorImpl();
  auto other_ = self.unsafeGetTensorImpl();
  // TORCH_CHECK(c10::isIntegralType(self.scalar_type(), true), "_opencl_remainder does not support non integral types");

  // The implementation applies fmod to the floating point types.
  //TORCH_CHECK(isIntegralType(self.scalar_type(), true), "Remainder only applies to integral types");

  TORCH_CHECK(opencl_nElement(*result_) == opencl_nElement(*self_), "sizes don't match");
  opencl_resizeAs(result_, self_);

  pointwise_op3(result, self, other, at::native::opencl::OpenCLOperationsPointwise3::REM, scalar_type);

  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0);
  return result;
}

REGISTER_DISPATCH(abs_stub, &abs_kernel_opencl);
REGISTER_DISPATCH(ceil_stub, &ceil_kernel_opencl);

}} // namespace at::native
