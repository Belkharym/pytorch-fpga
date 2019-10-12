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
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorFactories.h>
#include <c10/util/Exception.h>
#include <ATen/Backend.h>
#include <ATen/Utils.h>
#include <ATen/native/opencl/Resize.h>

#include <c10/opencl/OpenCLFunctions.h>
#include <aten/src/ATen/native/opencl/OpenCLOperations.h>
#include <c10/opencl/OpenCLFunctions.h>
#include <aten/src/ATen/native/opencl/OpenCLOperations.h>

namespace at { namespace native {

static std::string getKernelTypeSuffix(const ScalarType type) {
  switch (type)
  {
    case ScalarType::Bool:
      return "c";
      break;
    case ScalarType::Byte:
      return "c";
      break;
    case ScalarType::Char:
      return "c";
      break;
    case ScalarType::Short:
      return "s";
      break;
    case ScalarType::Int:
      return "i";
      break;
    case ScalarType::Long:
      return "l";
      break;
    case ScalarType::Half:
      return "h";
      break;
    case ScalarType::Float:
      return "f";
      break;
    case ScalarType::Double:
      return "d";
      break;
    case ScalarType::QInt32:
      return "i";
      break;
    case ScalarType::QInt8:
      return "c";
      break;
    case ScalarType::QUInt8:
      return "c";
      break;

    default:
      return "u"; // For unknown
      break;
  }
}

static void operation(const StorageImpl* a, const StorageImpl* b, StorageImpl* out, const Scalar alpha, at::native::opencl::OpenCLOperationsPointwise op, const ScalarType scalar_type) {
  // DONE Call OpenCL kernel.
  auto kernel_name = "operation_" +  getKernelTypeSuffix(scalar_type);
  auto opt_kernel = c10::opencl::opencl_kernel(kernel_name);
  if (!opt_kernel) {
    TORCH_WARN("No value for kernel \"", kernel_name, "\"");
    return;
  }
  cl::Kernel pointwise_op = opt_kernel.value();
  pointwise_op.setArg<cl_mem>(0, (*(cl::Buffer*)a->data_ptr().get())());
  pointwise_op.setArg<cl_mem>(1, (*(cl::Buffer*)b->data_ptr().get())());
  pointwise_op.setArg<float>(2, alpha.to<float>());
  pointwise_op.setArg<cl_mem>(3, (*(cl::Buffer*)out->data_ptr().get())());
  pointwise_op.setArg<at::native::opencl::OpenCLOperationsPointwise>(2, op);
  auto stream = caffe2::opencl::getCurrentOpenCLStream(a->device().index());
  stream.stream()->enqueueNDRangeKernel(pointwise_op, /*offset=*/0, a->numel());
  stream.stream()->finish();
}

Tensor & add_out(Tensor &out, Scalar alpha, const Tensor& other, const Tensor &self){
    auto other_ = checked_tensor_unwrap(out, "out", 1, "add_out_opencl", false, c10::Backend::OpenCL, self.scalar_type());
    auto self_ = checked_tensor_unwrap(self, "self", 2, "add_out_opencl", false, c10::Backend::OpenCL, self.scalar_type());
    auto out_ = checked_tensor_unwrap(out, "out", 3, "add_out_opencl", false, c10::Backend::OpenCL, self.scalar_type());
    if (alpha.isFloatingPoint())
        operation(self_->storage().unsafeGetStorageImpl(), other_->storage().unsafeGetStorageImpl(), out_->storage().unsafeGetStorageImpl(),
        alpha, at::native::opencl::OpenCLOperationsPointwise::ADD, self.scalar_type());
    return out;
}

// static void sub_kernel_cuda(TensorIterator& iter, Scalar alpha_scalar) {
//   add_kernel_cuda(iter, -alpha_scalar);
// }

// void div_kernel_cuda(TensorIterator& iter) {
//   if (!isIntegralType(iter.dtype(), /*includeBool*/ false) && iter.is_cpu_scalar(2)) {
//     // optimization for floating-point types: if the second operand is a CPU
//     // scalar, compute a * reciprocal(b). Note that this may lose one bit of
//     // precision compared to computing the division.
//     AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "div_cuda", [&]() {
//       auto inv_b = scalar_t(1.0 / iter.scalar_value<scalar_t>(2));
//       iter.remove_operand(2);
//       gpu_kernel(iter, [inv_b]GPU_LAMBDA(scalar_t a) -> scalar_t {
//         return a * inv_b;
//       });
//     });
//   } else {
//     AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.dtype(), "div_cuda", [&]() {
//       gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
//         return a / b;
//       });
//     });
//   }
// }

// void mul_kernel_cuda(TensorIterator& iter) {
//   if (iter.dtype() == ScalarType::Bool) {
//     // Workaround for the error: '*' in boolean context, suggest '&&' instead [-Werror=int-in-bool-context]
//     gpu_kernel_with_scalars(iter, []GPU_LAMBDA(bool a, bool b) -> bool {
//       return a && b;
//     });
//   } else {
//     AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.dtype(), "mul_cuda", [&]() {
//       gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
//         return a * b;
//       });
//     });
//   }
// }

// void atan2_kernel_cuda(TensorIterator& iter) {
//   AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "atan2_cuda", [&]() {
//     gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
//       return THCNumerics<scalar_t>::atan2(a, b);
//     });
//   });
// }

// void logical_xor_kernel_cuda(TensorIterator& iter) {
//   gpu_kernel(iter, []GPU_LAMBDA(bool a, bool b) -> bool {
//     return a != b;
//   });
// }

// REGISTER_OPENCL_DISPATCH(add_stub, &add_kernel_cuda);
// REGISTER_OPENCL_DISPATCH(sub_stub, &sub_kernel_cuda);
// REGISTER_OPENCL_DISPATCH(div_stub, &div_kernel_cuda);
// REGISTER_OPENCL_DISPATCH(mul_stub, &mul_kernel_cuda);
// REGISTER_OPENCL_DISPATCH(atan2_stub, &atan2_kernel_cuda);
// REGISTER_OPENCL_DISPATCH(logical_xor_stub, &logical_xor_kernel_cuda);

}} // namespace at::native
