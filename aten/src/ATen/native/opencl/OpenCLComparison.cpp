#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/Exception.h>
#include <ATen/Backend.h>
#include <ATen/Utils.h>

#include <aten/src/ATen/opencl/Exceptions.h> // This include must be before ATen/native/DispatchStub.h

#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

#include <c10/opencl/OpenCLFunctions.h>
#include <aten/src/ATen/native/opencl/OpenCLOperations.h>
#include <aten/src/ATen/native/opencl/Utils.h>

namespace at {
namespace native {

static cl::Buffer &toBuffer(const Tensor& s) {
  return (*toBuffer(s.storage().data()));
}

static void pointwise_op_comp3(Tensor& c, const Tensor& a, const Tensor& b, at::native::opencl::OpenCLOperationsComp3 op, const ScalarType scalar_type) {
  // DONE Call OpenCL kernel.
  static const std::string kernel_name = "pointwise_op_comp_3";
  auto stream = at::opencl::getCurrentOpenCLStream(a.device().index());
  auto pointwise_op = c10::opencl::opencl_kernel_func<OpenCLComp3Functor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange{(size_t)a.storage_offset()}, cl::NDRange{(size_t)a.numel()}, 1});
  AT_OPENCL_CHECK(pointwise_op(
      toBuffer(a),
      toBuffer(b),
      toBuffer(c),
      op,
      getOpenCLKernelCastType(scalar_type)));
  AT_OPENCL_CHECK(syncOpenCLPointer(c.data_ptr(), stream));
  stream.synchronize();
}

template <typename S>
static void pointwise_op_comp2_s(Tensor& c, const Tensor& a, const Scalar b, at::native::opencl::OpenCLOperationsComp3 op, c10::ScalarType T) {
  static const std::string kernel_name = "pointwise_op_comp_2s";
  auto stream = at::opencl::getCurrentOpenCLStream(a.device().index());
  auto pointwise_op = c10::opencl::opencl_kernel_func<OpenCLComp3Functor>(kernel_name, cl::EnqueueArgs{*stream.stream(), cl::NDRange{(size_t)a.storage_offset()}, cl::NDRange{(size_t)a.numel()}, 1});

  auto scalar_buffer = at::native::scalar_buffer_opencl<S>(b, a.device().index());

  AT_OPENCL_CHECK(pointwise_op(
      toBuffer(a),
      toBuffer(scalar_buffer),
      toBuffer(c),
      op,
      getOpenCLKernelCastType(T)));
  AT_OPENCL_CHECK(syncOpenCLPointer(c.data_ptr(), stream));
  stream.synchronize();
}

static void logical_tensor(TensorIterator& iter, const std::string &op_name, opencl::OpenCLOperationsComp3 op, opencl::OpenCLOperationsComp3 op_inv_order) {
    AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, iter.common_dtype(), op_name, [&]() {
        if (iter.is_scalar(1)) {
            auto stream = at::opencl::getCurrentOpenCLStream(iter.device(1).index());
            AT_OPENCL_CHECK(syncOpenCLPointer(iter.data_ptr(1), stream));
            AT_OPENCL_CHECK(stream.stream()->finish());
            pointwise_op_comp2_s<scalar_t>(iter.tensor(0), iter.tensor(2), iter.tensor(1).item(), op_inv_order, iter.common_dtype());
        } else if (iter.is_scalar(2)) {
            auto stream = at::opencl::getCurrentOpenCLStream(iter.device(2).index());
            AT_OPENCL_CHECK(syncOpenCLPointer(iter.data_ptr(2), stream));
            AT_OPENCL_CHECK(stream.stream()->finish());
            pointwise_op_comp2_s<scalar_t>(iter.tensor(0), iter.tensor(1), iter.tensor(2).item(), op, iter.common_dtype());
        } else {
            pointwise_op_comp3(iter.tensor(0), iter.tensor(1), iter.tensor(2), op, iter.common_dtype());
        }
    });
}

void lt_kernel_opencl(TensorIterator& iter) {
  logical_tensor(iter, "lt_opencl", at::native::opencl::OpenCLOperationsComp3::LT, at::native::opencl::OpenCLOperationsComp3::GT);
}

void le_kernel_opencl(TensorIterator& iter) {
  logical_tensor(iter, "le_opencl", at::native::opencl::OpenCLOperationsComp3::LE, at::native::opencl::OpenCLOperationsComp3::GE);
}

void gt_kernel_opencl(TensorIterator& iter) {
  logical_tensor(iter, "gt_opencl", at::native::opencl::OpenCLOperationsComp3::GT, at::native::opencl::OpenCLOperationsComp3::LT);
}

void ge_kernel_opencl(TensorIterator& iter) {
  logical_tensor(iter, "ge_opencl", at::native::opencl::OpenCLOperationsComp3::GE, at::native::opencl::OpenCLOperationsComp3::LE);
}

void eq_kernel_opencl(TensorIterator& iter) {
  logical_tensor(iter, "eq_opencl", at::native::opencl::OpenCLOperationsComp3::EQ, at::native::opencl::OpenCLOperationsComp3::EQ);
}

void ne_kernel_opencl(TensorIterator& iter) {
  logical_tensor(iter, "ne_opencl", at::native::opencl::OpenCLOperationsComp3::NE, at::native::opencl::OpenCLOperationsComp3::NE);
}

REGISTER_OPENCL_DISPATCH(lt_stub, &lt_kernel_opencl);
REGISTER_OPENCL_DISPATCH(le_stub, &le_kernel_opencl);
REGISTER_OPENCL_DISPATCH(gt_stub, &gt_kernel_opencl);
REGISTER_OPENCL_DISPATCH(ge_stub, &ge_kernel_opencl);
REGISTER_OPENCL_DISPATCH(eq_stub, &eq_kernel_opencl);
REGISTER_OPENCL_DISPATCH(ne_stub, &ne_kernel_opencl);

}} // namespace at::native