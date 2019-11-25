#include <ATen/ATen.h>

#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

#include <ATen/Backend.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/native/TensorFactories.h>
#include <c10/util/Exception.h>
#include <ATen/Backend.h>
#include <ATen/Utils.h>
#include <ATen/native/opencl/Resize.h>

#include <c10/opencl/OpenCLFunctions.h>
#include <aten/src/ATen/native/opencl/OpenCLOperations.h>
#include <c10/opencl/OpenCLFunctions.h>
#include <aten/src/ATen/opencl/Exceptions.h>
#include <aten/src/ATen/native/opencl/OpenCLOperations.h>
#include <aten/src/ATen/native/opencl/Utils.h>

#define AT_FORALL_INTEGER_TYPES(_)  \
    _(uint8_t, Byte)                \
    _(int8_t, Char)                 \
    _(int16_t, Short)               \
    _(int, Int)                     \
    _(int64_t, Long)

#define AT_FORALL_INTEGER_TYPES_AND(SCALARTYPE, _)                         \
  _(uint8_t, Byte)                                                         \
  _(int8_t, Char)                                                          \
  _(int16_t, Short)                                                        \
  _(int, Int)                                                              \
  _(int64_t, Long)                                                         \
  _(decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::SCALARTYPE>::t), SCALARTYPE)

namespace at {
namespace native {

static cl::Buffer &toBuffer(const StorageImpl* s) {
    return (*toBuffer(s->data_ptr().get()));
}

template <c10::ScalarType T, typename S = decltype(c10::impl::ScalarTypeToCPPType<T>::t)>
static void pointwise_op3s(const StorageImpl* a, const StorageImpl* b, StorageImpl* out, const Scalar alpha, at::native::opencl::OpenCLOperationsPointwise3s op) {
  // DONE Call OpenCL kernel.
  static const std::string kernel_name = "operation_3_s";
  auto stream = at::opencl::getCurrentOpenCLStream(a->device().index());
  auto pointwise_op = c10::opencl::opencl_kernel_func<OpenCLPointwise3sFunctor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange{(size_t)a->numel()}, 1});
  
  Tensor scalar_tensor = at::native::scalar_tensor_opencl<T>(alpha, TensorOptions{T}.merge_in({a->device()}));
  auto scalar_tensor_ = scalar_tensor.storage().unsafeGetStorageImpl();

  AT_OPENCL_CHECK(pointwise_op(
      toBuffer(a),
      toBuffer(b),
      toBuffer(out),
      toBuffer(scalar_tensor_),
      op,
      getOpenCLKernelCastType(T), 
      getOpenCLKernelCastType(T)));
  AT_OPENCL_CHECK(syncOpenCLPointer(out->data_ptr().get(), stream));
  stream.synchronize();
}


static void pointwise_op3(const StorageImpl* a, const StorageImpl* b, StorageImpl* out, at::native::opencl::OpenCLOperationsPointwise3 op, const ScalarType scalar_type) {
  // DONE Call OpenCL kernel.
  static const std::string kernel_name = "pointwise_op_3";
  auto stream = at::opencl::getCurrentOpenCLStream(a->device().index());
  auto pointwise_op = c10::opencl::opencl_kernel_func<OpenCLPointwise3Functor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange{(size_t)a->numel()}, 1});

  AT_OPENCL_CHECK(pointwise_op(
      toBuffer(a),
      toBuffer(b),
      toBuffer(out),
      op,
      getOpenCLKernelCastType(scalar_type)));

  AT_OPENCL_CHECK(syncOpenCLPointer(out->data_ptr().get(), stream));
  stream.synchronize();
}

template <c10::ScalarType T, typename S = decltype(c10::impl::ScalarTypeToCPPType<T>::t)>
static void pointwise_op2s(const StorageImpl* a, const Scalar b, StorageImpl* c, at::native::opencl::OpenCLOperationsPointwise3 op) {
  static const std::string kernel_name = "pointwise_op_2s";
  auto stream = at::opencl::getCurrentOpenCLStream(a->device().index());
  auto pointwise_op = c10::opencl::opencl_kernel_func<OpenCLPointwise3Functor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange{(size_t)a->numel()}, 1});

  Tensor scalar_tensor = at::native::scalar_tensor_opencl<T>(b, TensorOptions{T}.merge_in({a->device()}));
  auto scalar_tensor_ = scalar_tensor.storage().unsafeGetStorageImpl();

  AT_OPENCL_CHECK(pointwise_op(
      toBuffer(a),
      toBuffer(scalar_tensor_),
      toBuffer(c),
      op,
      getOpenCLKernelCastType(T)));

  AT_OPENCL_CHECK(syncOpenCLPointer(c->data_ptr().get(), stream));
  stream.synchronize();
}

// STUB


void add_kernel_opencl(TensorIterator& iter, Scalar alpha) {
    auto scalar_type = iter.tensor(1).scalar_type();
    auto other_ = checked_tensor_unwrap(iter.tensor(2), "other", 1, "add_kernel_opencl", false, c10::Backend::OpenCL, iter.tensor(1).scalar_type());
    auto self_ = checked_tensor_unwrap(iter.tensor(1), "self", 2, "add_kernel_opencl", false, c10::Backend::OpenCL, iter.tensor(1).scalar_type());
    auto out_ = checked_tensor_unwrap(iter.tensor(0), "out", 3, "add_kernel_opencl", false, c10::Backend::OpenCL, iter.tensor(1).scalar_type());

    switch (scalar_type)
    {
#define DEFINE_OPENCL_ADD_CASE(type, name) \
        case ScalarType::name: { \
            if (iter.is_scalar(1)) { \
                AT_OPENCL_CHECK(syncOpenCLPointer(iter.tensor(1).data_ptr())); \
                AT_OPENCL_CHECK(at::opencl::getCurrentOpenCLStream().stream()->finish()); \
                pointwise_op2s<ScalarType::name, type>(other_->storage().unsafeGetStorageImpl(), Scalar(iter.scalar_value<type>(1) * alpha.to<type>()), out_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::ADD); \
            } else if (iter.is_scalar(2)) { \
                AT_OPENCL_CHECK(syncOpenCLPointer(iter.tensor(2).data_ptr())); \
                AT_OPENCL_CHECK(at::opencl::getCurrentOpenCLStream().stream()->finish()); \
                pointwise_op2s<ScalarType::name, type>(self_->storage().unsafeGetStorageImpl(), Scalar(iter.scalar_value<type>(2) * alpha.to<type>()), out_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::ADD); \
            } else { \
                TORCH_CHECK(opencl_nElement(self_) == opencl_nElement(other_), "sizes don't match"); \
                TORCH_CHECK(opencl_nElement(out_) == opencl_nElement(self_), "sizes don't match"); \
                pointwise_op3s<ScalarType::name, type>(self_->storage().unsafeGetStorageImpl(), other_->storage().unsafeGetStorageImpl(), out_->storage().unsafeGetStorageImpl(), alpha, at::native::opencl::OpenCLOperationsPointwise3s::ADDS); \
            } \
            break; \
        }
        AT_FORALL_SCALAR_TYPES(DEFINE_OPENCL_ADD_CASE)
#undef DEFINE_OPENCL_ADD_CASE

    default:
        TORCH_CHECK(false, "logical_tensor not supported on OpenCLType for ", scalar_type);
        break;
    }
}

void sub_kernel_opencl(TensorIterator& iter, Scalar alpha) {
    add_kernel_opencl(iter, -alpha);
}

static void op_scalar_opencl(const StorageImpl* a, Scalar b, StorageImpl* out, at::native::opencl::OpenCLOperationsPointwise3 op, ScalarType scalar_type) {
    switch (scalar_type)
    {
#define DEFINE_OPENCL_ARITH_OP(type, name) \
    case ScalarType::name: \
        pointwise_op2s<ScalarType::name, type>(a, b, out, op); \
        break;
        AT_FORALL_SCALAR_TYPES(DEFINE_OPENCL_ARITH_OP)
#undef DEFINE_OPENCL_ARITH_OP
    default:
        TORCH_CHECK(false, op, " not supported on OpenCLType for ", scalar_type);
        break;
    }
}

void mul_kernel_opencl(TensorIterator& iter) {
    auto scalar_type = iter.tensor(1).scalar_type();
    auto other_ = checked_tensor_unwrap(iter.tensor(2), "other", 1, "mul_kernel_opencl", false, c10::Backend::OpenCL, scalar_type);
    auto self_ = checked_tensor_unwrap(iter.tensor(1), "self", 2, "mul_kernel_opencl", false, c10::Backend::OpenCL, scalar_type);
    auto out_ = checked_tensor_unwrap(iter.tensor(0), "out", 3, "mul_kernel_opencl", false, c10::Backend::OpenCL, scalar_type);

    // TORCH_CHECK(c10::isIntegralType(scalar_type, true), "mul_kernel_opencl does not support non integral types");

    if (iter.is_scalar(1)) {
        AT_OPENCL_CHECK(syncOpenCLPointer(iter.tensor(1).data_ptr()));
        AT_OPENCL_CHECK(at::opencl::getCurrentOpenCLStream().stream()->finish());
        op_scalar_opencl(other_->storage().unsafeGetStorageImpl(), iter.tensor(1).item(), out_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::MUL, scalar_type);
    } else if (iter.is_scalar(2)) {
        AT_OPENCL_CHECK(syncOpenCLPointer(iter.tensor(2).data_ptr()));
        AT_OPENCL_CHECK(at::opencl::getCurrentOpenCLStream().stream()->finish());
        op_scalar_opencl(self_->storage().unsafeGetStorageImpl(), iter.tensor(2).item(), out_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::MUL, scalar_type);
    } else {
        TORCH_CHECK(opencl_nElement(self_) == opencl_nElement(other_), "sizes don't match");
        TORCH_CHECK(opencl_nElement(out_) == opencl_nElement(self_), "sizes don't match");
        pointwise_op3(self_->storage().unsafeGetStorageImpl(), other_->storage().unsafeGetStorageImpl(), out_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::MUL, scalar_type);
    }
}

void div_kernel_opencl(TensorIterator& iter) {
    auto scalar_type = iter.tensor(1).scalar_type();
    auto other_ = checked_tensor_unwrap(iter.tensor(2), "other", 1, "div_kernel_opencl", false, c10::Backend::OpenCL, scalar_type);
    auto self_ = checked_tensor_unwrap(iter.tensor(1), "self", 2, "div_kernel_opencl", false, c10::Backend::OpenCL, scalar_type);
    auto out_ = checked_tensor_unwrap(iter.tensor(0), "out", 3, "div_kernel_opencl", false, c10::Backend::OpenCL, scalar_type);

    // TORCH_CHECK(c10::isIntegralType(scalar_type, true), "div_kernel_opencl does not support non integral types");

    if (iter.is_scalar(1)) {
        AT_OPENCL_CHECK(syncOpenCLPointer(iter.tensor(1).data_ptr()));
        AT_OPENCL_CHECK(at::opencl::getCurrentOpenCLStream().stream()->finish());
        op_scalar_opencl(other_->storage().unsafeGetStorageImpl(), iter.tensor(1).item(), out_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::DIV, scalar_type);
    } else if (iter.is_scalar(2)) {
        AT_OPENCL_CHECK(syncOpenCLPointer(iter.tensor(2).data_ptr()));
        AT_OPENCL_CHECK(at::opencl::getCurrentOpenCLStream().stream()->finish());
        op_scalar_opencl(self_->storage().unsafeGetStorageImpl(), iter.tensor(2).item(), out_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::DIV, scalar_type);
    } else {
        TORCH_CHECK(opencl_nElement(self_) == opencl_nElement(other_), "sizes don't match");
        TORCH_CHECK(opencl_nElement(out_) == opencl_nElement(self_), "sizes don't match");
        pointwise_op3(self_->storage().unsafeGetStorageImpl(), other_->storage().unsafeGetStorageImpl(), out_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::DIV, scalar_type);
    }
}

void logical_xor_kernel_opencl(TensorIterator& iter) {
    auto scalar_type = iter.tensor(1).scalar_type();
    auto other_ = checked_tensor_unwrap(iter.tensor(2), "other", 1, "logical_xor_kernel_opencl", false, c10::Backend::OpenCL, scalar_type);
    auto self_ = checked_tensor_unwrap(iter.tensor(1), "self", 2, "logical_xor_kernel_opencl", false, c10::Backend::OpenCL, scalar_type);
    auto out_ = checked_tensor_unwrap(iter.tensor(0), "out", 3, "logical_xor_kernel_opencl", false, c10::Backend::OpenCL, scalar_type);

    // TORCH_CHECK(c10::isIntegralType(scalar_type, true), "logical_xor_kernel_opencl does not support non integral types");

    if (iter.is_scalar(1)) {
        AT_OPENCL_CHECK(syncOpenCLPointer(iter.tensor(1).data_ptr()));
        AT_OPENCL_CHECK(at::opencl::getCurrentOpenCLStream().stream()->finish());
        op_scalar_opencl(other_->storage().unsafeGetStorageImpl(), iter.tensor(1).item(), out_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::BXOR, scalar_type);
    } else if (iter.is_scalar(2)) {
        AT_OPENCL_CHECK(syncOpenCLPointer(iter.tensor(2).data_ptr()));
        AT_OPENCL_CHECK(at::opencl::getCurrentOpenCLStream().stream()->finish());
        op_scalar_opencl(self_->storage().unsafeGetStorageImpl(), iter.tensor(2).item(), out_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::BXOR, scalar_type);
    } else {
        TORCH_CHECK(opencl_nElement(self_) == opencl_nElement(other_), "sizes don't match");
        TORCH_CHECK(opencl_nElement(out_) == opencl_nElement(self_), "sizes don't match");
        pointwise_op3(self_->storage().unsafeGetStorageImpl(), other_->storage().unsafeGetStorageImpl(), out_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::BXOR, scalar_type);
    }
}

void atan2_kernel_opencl(TensorIterator& iter) {
    auto scalar_type = iter.tensor(1).scalar_type();
    auto other_ = checked_tensor_unwrap(iter.tensor(2), "other", 1, "atan2_kernel_opencl", false, c10::Backend::OpenCL, scalar_type);
    auto self_ = checked_tensor_unwrap(iter.tensor(1), "self", 2, "atan2_kernel_opencl", false, c10::Backend::OpenCL, scalar_type);
    auto out_ = checked_tensor_unwrap(iter.tensor(0), "out", 3, "atan2_kernel_opencl", false, c10::Backend::OpenCL, scalar_type);

    // TORCH_CHECK(c10::isIntegralType(scalar_type, true), "atan2_kernel_opencl does not support non integral types");

    if (iter.is_scalar(1)) {
        AT_OPENCL_CHECK(syncOpenCLPointer(iter.tensor(1).data_ptr()));
        AT_OPENCL_CHECK(at::opencl::getCurrentOpenCLStream().stream()->finish());
        op_scalar_opencl(other_->storage().unsafeGetStorageImpl(), iter.tensor(1).item(), out_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::ATAN2, scalar_type);
    } else if (iter.is_scalar(2)) {
        AT_OPENCL_CHECK(syncOpenCLPointer(iter.tensor(2).data_ptr()));
        AT_OPENCL_CHECK(at::opencl::getCurrentOpenCLStream().stream()->finish());
        op_scalar_opencl(self_->storage().unsafeGetStorageImpl(), iter.tensor(2).item(), out_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::ATAN2, scalar_type);
    } else {
        TORCH_CHECK(opencl_nElement(self_) == opencl_nElement(other_), "sizes don't match");
        TORCH_CHECK(opencl_nElement(out_) == opencl_nElement(self_), "sizes don't match");
        pointwise_op3(self_->storage().unsafeGetStorageImpl(), other_->storage().unsafeGetStorageImpl(), out_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::ATAN2, scalar_type);
    }
}


REGISTER_OPENCL_DISPATCH(add_stub, &add_kernel_opencl);
REGISTER_OPENCL_DISPATCH(sub_stub, &sub_kernel_opencl);
REGISTER_OPENCL_DISPATCH(div_stub, &div_kernel_opencl);
REGISTER_OPENCL_DISPATCH(mul_stub, &mul_kernel_opencl);
REGISTER_OPENCL_DISPATCH(atan2_stub, &atan2_kernel_opencl);
REGISTER_OPENCL_DISPATCH(logical_xor_stub, &logical_xor_kernel_opencl);

}} // namespace at::native
