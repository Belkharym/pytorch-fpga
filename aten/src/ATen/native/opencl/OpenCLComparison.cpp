#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/Exception.h>
#include <ATen/Backend.h>
#include <ATen/Utils.h>
#include <ATen/native/opencl/Resize.h>
#include <ATen/NamedTensorUtils.h>

#include <c10/opencl/OpenCLFunctions.h>
#include <aten/src/ATen/opencl/Exceptions.h>
#include <aten/src/ATen/native/opencl/OpenCLOperations.h>
#include <aten/src/ATen/native/opencl/Utils.h>

namespace at {
namespace native {

static cl::Buffer &toBuffer(const StorageImpl*s) {
  return (*toBuffer(s->data_ptr().get()));
}

static void pointwise_op_comp3(StorageImpl* c, const StorageImpl* a, const StorageImpl* b, at::native::opencl::OpenCLOperationsComp3 op, const ScalarType scalar_type) {
  // DONE Call OpenCL kernel.
  auto kernel_name = "pointwise_op_comp_3";
  auto stream = at::opencl::getCurrentOpenCLStream(a->device().index());
  auto pointwise_op = c10::opencl::opencl_kernel_func<OpenCLComp3Functor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange{(size_t)a->numel()}, 1});
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
static void pointwise_op_comp2_s(StorageImpl* c, const StorageImpl* a, const Scalar b, at::native::opencl::OpenCLOperationsComp3 op) {
  auto kernel_name = "pointwise_op_comp_2s";
  auto stream = at::opencl::getCurrentOpenCLStream(a->device().index());
  auto pointwise_op = c10::opencl::opencl_kernel_func<OpenCLComp3Functor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange{(size_t)a->numel()}, 1});
  
  Tensor scalar_tensor = at::native::scalar_tensor_opencl<T>(b, TensorOptions{T}.merge_in({a->device()}));
  auto scalar_tensor_ = scalar_tensor.storage().unsafeGetStorageImpl();

  AT_OPENCL_CHECK(pointwise_op(
      toBuffer(a),
      toBuffer(scalar_tensor_),
      toBuffer(c),
      op,
      getOpenCLKernelCastType(T)));
  AT_OPENCL_CHECK(syncOpenCLPointer(c->data_ptr().get(), stream));
  AT_OPENCL_CHECK(stream.stream()->finish());
}

// See THC_logicalTensor in aten/src/THC/THCTensorMathCompareT.cuh for implementation details
static void logical_tensor(TensorImpl *self_, const TensorImpl *t1, const TensorImpl *t2, opencl::OpenCLOperationsComp3 op) {
  opencl_resize(self_, t1->sizes(), {});
  TORCH_CHECK(opencl_nElement(t1) == opencl_nElement(t2), "sizes don't match");
  // TORCH_CHECK(!(op == opencl::OpenCLOperationsPointwise3::BAND && (
  //                 isFloatingType(typeMetaToScalarType(t1->dtype())) ||
  //                 isFloatingType(typeMetaToScalarType(t2->dtype()))
  //             )), "BitWise operation not supported on floating point types.");
  pointwise_op_comp3(self_->storage().unsafeGetStorageImpl(), t1->storage().unsafeGetStorageImpl(), t2->storage().unsafeGetStorageImpl(), op, typeMetaToScalarType(t1->dtype()));
}

static void logical_tensor(TensorImpl *self_, const TensorImpl *t1, const Scalar t2, opencl::OpenCLOperationsComp3 op) {
  opencl_resize(self_, t1->sizes(), {});
  auto scalar_type = typeMetaToScalarType(t1->dtype());
  switch (scalar_type)
  {
#define DEFINE_OPENCL_LOGICAL_TENSOR_CASE(type, name) \
  case ScalarType::name: \
    pointwise_op_comp2_s<ScalarType::name, type>(self_->storage().unsafeGetStorageImpl(), t1->storage().unsafeGetStorageImpl(), t2, op); \
    break;
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_OPENCL_LOGICAL_TENSOR_CASE)
#undef DEFINE_OPENCL_LOGICAL_TENSOR_CASE

  default:
    TORCH_CHECK(false, "logical_tensor not supported on OpenCLType for ", scalar_type);
    break;
  }
}

#define DEFINE_FOR_ALL_COMP(_) \
_(eq, EQ) \
_(ne, NE) \
_(gt, GT) \
_(lt, LT) \
_(ge, GE) \
_(le, LE)

#define DEFINE_LOGICAL_OP(label, OP) \
Tensor _##label##_opencl(const Tensor &self, const Tensor& other) { \
  auto allocator = c10::GetAllocator(DeviceType::OPENCL); \
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release(); \
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_)); \
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_" #label "_opencl", false, c10::Backend::OpenCL, self.scalar_type()); \
  auto other_ = checked_tensor_unwrap(other, "other", 2, "_" #label "_opencl", false, c10::Backend::OpenCL, self.scalar_type()); \
  logical_tensor(result_, self_, other_, at::native::opencl::OpenCLOperationsComp3::OP); \
  result_->maybe_zero_dim(self_->dim() == 0 && other_->dim() == 0); \
  return result.to(getIntEquivalentOfFloat(result.scalar_type())).to(ScalarType::Bool); \
}

DEFINE_FOR_ALL_COMP(DEFINE_LOGICAL_OP)

#define DEFINE_LOGICAL_OP_SCALAR(label, OP) \
Tensor _##label##_opencl(const Tensor &self, Scalar other) { \
  auto allocator = c10::GetAllocator(DeviceType::OPENCL); \
  auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(scalarTypeToTypeMeta(self.scalar_type()), 0, allocator, true),TensorTypeId::OpenCLTensorId).release(); \
  auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_)); \
  auto self_ = checked_tensor_unwrap(self, "self", 1, "_" #label "_opencl", false, c10::Backend::OpenCL, self.scalar_type()); \
  logical_tensor(result_, self_, other, at::native::opencl::OpenCLOperationsComp3::OP); \
  result_->maybe_zero_dim(self_->dim() == 0); \
  return result.to(getIntEquivalentOfFloat(result.scalar_type())).to(ScalarType::Bool); \
}

DEFINE_FOR_ALL_COMP(DEFINE_LOGICAL_OP_SCALAR)

}} // namespace at::native