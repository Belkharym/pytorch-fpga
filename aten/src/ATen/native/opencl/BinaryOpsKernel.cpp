#include <ATen/ATen.h>

#include <aten/src/ATen/opencl/Exceptions.h> // This include must be before ATen/native/DispatchStub.h

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

static cl::Buffer &toBuffer(const Tensor& s) {
    return (*toBuffer(s.data_ptr()));
}

template <typename S>
static void pointwise_op3s(const Tensor& a, const Tensor& b, Tensor& out, const Scalar alpha, at::native::opencl::OpenCLOperationsPointwise3s op, c10::ScalarType T) {
  // DONE Call OpenCL kernel.
  static const std::string kernel_name = "operation_3_s";
  auto stream = at::opencl::getCurrentOpenCLStream(a.device().index());
  TORCH_WARN("size a: ", a.numel(), "; size b: ", b.numel(), "; size out: ", out.numel());
  auto pointwise_op = c10::opencl::opencl_kernel_func<OpenCLPointwise3sFunctor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange{(size_t)a.storage_offset()}, cl::NDRange{(size_t)a.numel()}, cl::NDRange{1}});
  
  at::Tensor scalar_buffer = at::native::scalar_buffer_opencl<S>(alpha, a.device().index());

  AT_OPENCL_CHECK(pointwise_op(
      toBuffer(a),
      toBuffer(b),
      toBuffer(out),
      toBuffer(scalar_buffer),
      op,
      getOpenCLKernelCastType(T), 
      getOpenCLKernelCastType(T)));
  AT_OPENCL_CHECK(syncOpenCLPointer(out.data_ptr(), stream));
  stream.synchronize();
}


static void pointwise_op3(const Tensor& a, const Tensor& b, Tensor& out, at::native::opencl::OpenCLOperationsPointwise3 op, const ScalarType scalar_type) {
  // DONE Call OpenCL kernel.
  static const std::string kernel_name = "pointwise_op_3";
  auto stream = at::opencl::getCurrentOpenCLStream(a.device().index());
  auto pointwise_op = c10::opencl::opencl_kernel_func<OpenCLPointwise3Functor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange{(size_t)a.storage_offset()}, cl::NDRange{(size_t)a.numel()}, 1});

  AT_OPENCL_CHECK(pointwise_op(
      toBuffer(a),
      toBuffer(b),
      toBuffer(out),
      op,
      getOpenCLKernelCastType(scalar_type)));

  AT_OPENCL_CHECK(syncOpenCLPointer(out.data_ptr(), stream));
  stream.synchronize();
}

template <typename S>
static void pointwise_op2s(const Tensor& a, const Scalar b, Tensor& c, at::native::opencl::OpenCLOperationsPointwise3 op, c10::ScalarType T, bool invert = false) {
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

// STUB


void add_kernel_opencl(TensorIterator& iter, Scalar alpha_scalar) {
    AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, iter.common_dtype(), "add_opencl/sub_opencl", [&]() {
        auto alpha = alpha_scalar.to<scalar_t>();
        if (iter.is_scalar(1)) {
            auto stream = at::opencl::getCurrentOpenCLStream(iter.device(1).index());
            AT_OPENCL_CHECK(syncOpenCLPointer(iter.data_ptr(1), stream));
            AT_OPENCL_CHECK(stream.stream()->finish());
            Tensor tmp = at::native::empty_like(iter.tensor(2), iter.tensor(1).options(), iter.tensor(1).suggest_memory_format()).fill_(iter.scalar_value<scalar_t>(1));
            pointwise_op3s<scalar_t>(tmp, iter.tensor(2), iter.tensor(0), alpha_scalar, at::native::opencl::OpenCLOperationsPointwise3s::ADDS, iter.common_dtype());
        } else if (iter.is_scalar(2)) {
            auto stream = at::opencl::getCurrentOpenCLStream(iter.device(2).index());
            AT_OPENCL_CHECK(syncOpenCLPointer(iter.data_ptr(2), stream));
            AT_OPENCL_CHECK(stream.stream()->finish());
            pointwise_op2s<scalar_t>(iter.tensor(1), Scalar(iter.scalar_value<scalar_t>(2) * alpha), iter.tensor(0), at::native::opencl::OpenCLOperationsPointwise3::ADD, iter.common_dtype());
        } else {
            pointwise_op3s<scalar_t>(iter.tensor(1), iter.tensor(2), iter.tensor(0), alpha_scalar, at::native::opencl::OpenCLOperationsPointwise3s::ADDS, iter.common_dtype());
        }
    });
}

void sub_kernel_opencl(TensorIterator& iter, Scalar alpha_scalar) {
    add_kernel_opencl(iter, -alpha_scalar);
}

static void op_scalar_opencl(TensorIterator& iter, const std::string &op_name, at::native::opencl::OpenCLOperationsPointwise3 op) {
    AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, iter.common_dtype(), op_name, [&]() {
        if (iter.is_scalar(1)) {
            auto stream = at::opencl::getCurrentOpenCLStream(iter.device(1).index());
            AT_OPENCL_CHECK(syncOpenCLPointer(iter.data_ptr(1), stream));
            AT_OPENCL_CHECK(stream.stream()->finish());
            pointwise_op2s<scalar_t>(iter.tensor(2), Scalar(iter.scalar_value<scalar_t>(1)), iter.tensor(0), op, iter.common_dtype(), /*invert=*/true);
        } else if (iter.is_scalar(2)) {
            auto stream = at::opencl::getCurrentOpenCLStream(iter.device(2).index());
            AT_OPENCL_CHECK(syncOpenCLPointer(iter.data_ptr(2), stream));
            AT_OPENCL_CHECK(stream.stream()->finish());
            pointwise_op2s<scalar_t>(iter.tensor(1), Scalar(iter.scalar_value<scalar_t>(2)), iter.tensor(0), op, iter.common_dtype());
        } else {
            pointwise_op3(iter.tensor(1), iter.tensor(2), iter.tensor(0), op, iter.common_dtype());
        }
    });
}

void mul_kernel_opencl(TensorIterator& iter) {
    // TORCH_CHECK(c10::isIntegralType(scalar_type, true), "mul_kernel_opencl does not support non integral types");

    op_scalar_opencl(iter, "mul_opencl", at::native::opencl::OpenCLOperationsPointwise3::MUL);
}

void div_kernel_opencl(TensorIterator& iter) {
    // TORCH_CHECK(c10::isIntegralType(scalar_type, true), "div_kernel_opencl does not support non integral types");

    op_scalar_opencl(iter, "div_opencl", at::native::opencl::OpenCLOperationsPointwise3::DIV);
}

void logical_xor_kernel_opencl(TensorIterator& iter) {
    // TORCH_CHECK(c10::isIntegralType(scalar_type, true), "logical_xor_kernel_opencl does not support non integral types");

    op_scalar_opencl(iter, "div_opencl", at::native::opencl::OpenCLOperationsPointwise3::BXOR);
}

void atan2_kernel_opencl(TensorIterator& iter) {
    // TORCH_CHECK(c10::isIntegralType(scalar_type, true), "atan2_kernel_opencl does not support non integral types");

    op_scalar_opencl(iter, "div_opencl", at::native::opencl::OpenCLOperationsPointwise3::ATAN2);
}


REGISTER_DISPATCH(add_stub, &add_kernel_opencl);
REGISTER_DISPATCH(sub_stub, &sub_kernel_opencl);
REGISTER_DISPATCH(div_stub, &div_kernel_opencl);
REGISTER_DISPATCH(mul_stub, &mul_kernel_opencl);
REGISTER_DISPATCH(atan2_stub, &atan2_kernel_opencl);
REGISTER_DISPATCH(logical_xor_stub, &logical_xor_kernel_opencl);

}} // namespace at::native
