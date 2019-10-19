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
  auto kernel_name = "pointwise_op_3" + getOpenCLKernelTypeSuffix(scalar_type);
  auto opt_kernel = c10::opencl::opencl_kernel(kernel_name);
  TORCH_INTERNAL_ASSERT(opt_kernel.has_value(), "No value for kernel \"", kernel_name, "\"");
  cl::Kernel pointwise_op = opt_kernel.value();
  auto stream = at::opencl::getCurrentOpenCLStream(a->device().index());
  AT_OPENCL_CHECK(c10::opencl::runKernel(pointwise_op, {*stream.stream(), a->numel(), 1},
      toBuffer(a),
      toBuffer(b),
      toBuffer(c),
      op));
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
  auto stream = at::opencl::getCurrentOpenCLStream(a->device().index());
  AT_OPENCL_CHECK(c10::opencl::runKernel(pointwise_op, {*stream.stream(), a->numel(), 1},
      toBuffer(a),
      b.to<S>(),
      toBuffer(c),
      op));
  AT_OPENCL_CHECK(syncOpenCLPointer(c->data_ptr().get()));
  AT_OPENCL_CHECK(stream.stream()->finish());
}

static void pointwise_op2(StorageImpl* b, const StorageImpl* a, at::native::opencl::OpenCLOperationsPointwise2 op, const ScalarType scalar_type) {
  // DONE Call OpenCL kernel.
  auto kernel_name = "pointwise_op_2" + getOpenCLKernelTypeSuffix(scalar_type);
  auto opt_kernel = c10::opencl::opencl_kernel(kernel_name);
  TORCH_INTERNAL_ASSERT(opt_kernel.has_value(), "No value for kernel \"", kernel_name, "\"");
  cl::Kernel pointwise_op = opt_kernel.value();
  auto stream = at::opencl::getCurrentOpenCLStream(a->device().index());
  AT_OPENCL_CHECK(c10::opencl::runKernel(pointwise_op, {*stream.stream(), a->numel(), 1},
      toBuffer(a),
      toBuffer(b),
      op));
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

  auto kernel_name = "cast_is";
  auto kernel_opt = c10::opencl::opencl_kernel(kernel_name);
  TORCH_INTERNAL_ASSERT(kernel_opt.has_value(), "Kernel not found \"", kernel_name, "\"");
  auto stream = at::opencl::getCurrentOpenCLStream(self_->device().index());
  auto kernel = kernel_opt.value();
  int scalar_0 = 0;
  AT_OPENCL_CHECK(c10::opencl::runKernel(kernel, {*stream.stream(), self_->numel(), 1},
    (int)0,
    *toBuffer(self_->data()),
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
  return at::native::legacy::cpu::_th_max(self.toBackend(Backend::CPU)).toBackend(Backend::OpenCL);
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

template <typename T, typename IndexType>
struct CatArrInputTensor {
  T* input;
  IndexType offset;
  IndexType dimSize;
  IndexType nElements;
};

template<typename IndexType, unsigned int MaxDims>
struct OutputTensorSizeStride {
  IndexType outputSize[MaxDims];
  IndexType outputStride[MaxDims];
};

Tensor _opencl_cat(TensorList tensors, int64_t dim) {
  std::vector<Tensor> cpuTensors;
  std::transform(tensors.begin(), tensors.end(), std::back_inserter(cpuTensors), [](const Tensor& t) -> Tensor {return t.toBackend(Backend::CPU);});
  return at::native::legacy::cpu::_th_cat(cpuTensors, dim).toBackend(Backend::OpenCL);
}

}} // namespace at::native
