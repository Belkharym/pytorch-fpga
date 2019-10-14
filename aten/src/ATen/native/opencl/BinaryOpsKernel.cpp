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
#include <aten/src/ATen/native/opencl/OpenCLOperations.h>
#include <aten/src/ATen/native/opencl/Utils.h>

namespace at {
namespace native {

template <c10::ScalarType T, typename S = decltype(c10::impl::ScalarTypeToCPPType<T>::t)>
static void pointwise_op3s(const StorageImpl* a, const StorageImpl* b, StorageImpl* out, const Scalar alpha, at::native::opencl::OpenCLOperationsPointwise3s op) {
  // DONE Call OpenCL kernel.
  auto kernel_name = "pointwise_op3" + getOpenCLKernelTypeSuffix(T) + "_s";
  auto opt_kernel = c10::opencl::opencl_kernel(kernel_name);
  if (!opt_kernel) {
    TORCH_WARN("No value for kernel \"", kernel_name, "\"");
    return;
  }
  cl::Kernel pointwise_op = opt_kernel.value();
  pointwise_op.setArg<cl_mem>(0, (*(cl::Buffer*)a->data_ptr().get())());
  pointwise_op.setArg<cl_mem>(1, (*(cl::Buffer*)b->data_ptr().get())());
  pointwise_op.setArg<S>(2, alpha.to<S>());
  pointwise_op.setArg<cl_mem>(3, (*(cl::Buffer*)out->data_ptr().get())());
  pointwise_op.setArg<at::native::opencl::OpenCLOperationsPointwise3s>(4, op);
  auto stream = caffe2::opencl::getCurrentOpenCLStream(a->device().index());
  stream.stream()->enqueueNDRangeKernel(pointwise_op, /*offset=*/0, a->numel(), 1);
  stream.stream()->finish();
}


static void pointwise_op3(const StorageImpl* a, const StorageImpl* b, StorageImpl* out, at::native::opencl::OpenCLOperationsPointwise3 op, const ScalarType scalar_type) {
  // DONE Call OpenCL kernel.
  auto kernel_name = "pointwise_op3" + getOpenCLKernelTypeSuffix(scalar_type);
  auto opt_kernel = c10::opencl::opencl_kernel(kernel_name);
  if (!opt_kernel) {
    TORCH_WARN("No value for kernel \"", kernel_name, "\"");
    return;
  }
  cl::Kernel pointwise_op = opt_kernel.value();
  pointwise_op.setArg<cl_mem>(0, (*(cl::Buffer*)a->data_ptr().get())());
  pointwise_op.setArg<cl_mem>(1, (*(cl::Buffer*)b->data_ptr().get())());
  pointwise_op.setArg<cl_mem>(2, (*(cl::Buffer*)out->data_ptr().get())());
  pointwise_op.setArg<at::native::opencl::OpenCLOperationsPointwise3>(3, op);
  auto stream = caffe2::opencl::getCurrentOpenCLStream(a->device().index());
  stream.stream()->enqueueNDRangeKernel(pointwise_op, /*offset=*/0, a->numel(), 1);
  stream.stream()->finish();
}


// ADD FUNCTION

Tensor & opencl_add_out(Tensor &out, const Tensor &self, const Tensor& other , Scalar alpha){
    auto other_ = checked_tensor_unwrap(other, "other", 1, "add_out_opencl", false, c10::Backend::OpenCL, self.scalar_type());
    auto self_ = checked_tensor_unwrap(self, "self", 2, "add_out_opencl", false, c10::Backend::OpenCL, self.scalar_type());
    auto out_ = checked_tensor_unwrap(out, "out", 3, "add_out_opencl", false, c10::Backend::OpenCL, self.scalar_type());
    
    auto scalar_type = self.scalar_type();
    switch (scalar_type)
    {
#define DEFINE_OPENCL_ADD_CASE(type, name) \
        case ScalarType::name: \
            pointwise_op3s<ScalarType::name, type>(self_->storage().unsafeGetStorageImpl(), other_->storage().unsafeGetStorageImpl(), out_->storage().unsafeGetStorageImpl(), alpha, at::native::opencl::OpenCLOperationsPointwise3s::ADD); \
            break;
        AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_OPENCL_ADD_CASE)
#undef DEFINE_OPENCL_ADD_CASE

    default:
      TORCH_CHECK(false, "logical_tensor not supported on OpenCLType for ", scalar_type);
      break;
    }
    return out;
}

Tensor & opencl_add_(Tensor &self, const Tensor& other , Scalar alpha){
    return opencl_add_out(self, self, other, alpha);
}

Tensor opencl_add(const Tensor &self, const Tensor& other , Scalar alpha){
    Tensor out =  at::native::empty_opencl(self.sizes(), self.options());
    opencl_add_out(out, self, other, alpha);
    return out;
}

// SUB FUNCTION

Tensor & opencl_sub_out(Tensor &out, const Tensor &self, const Tensor& other , Scalar alpha){
    auto other_ = checked_tensor_unwrap(other, "other", 1, "opencl_sub_out", false, c10::Backend::OpenCL, self.scalar_type());
    auto self_ = checked_tensor_unwrap(self, "self", 2, "opencl_sub_out", false, c10::Backend::OpenCL, self.scalar_type());
    auto out_ = checked_tensor_unwrap(out, "out", 3, "opencl_sub_out", false, c10::Backend::OpenCL, self.scalar_type());
    
    auto scalar_type = self.scalar_type();
    switch (scalar_type)
    {
#define DEFINE_OPENCL_ADD_CASE(type, name) \
        case ScalarType::name: \
            pointwise_op3s<ScalarType::name, type>(self_->storage().unsafeGetStorageImpl(), other_->storage().unsafeGetStorageImpl(), out_->storage().unsafeGetStorageImpl(), alpha, at::native::opencl::OpenCLOperationsPointwise3s::SUB); \
            break;
        AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_OPENCL_ADD_CASE)
#undef DEFINE_OPENCL_ADD_CASE

    default:
      TORCH_CHECK(false, "logical_tensor not supported on OpenCLType for ", scalar_type);
      break;
    }
    return out;
}

Tensor & opencl_sub_(Tensor &self, const Tensor& other , Scalar alpha){
    return opencl_sub_out(self, self, other, alpha);
}

Tensor opencl_sub(const Tensor &self, const Tensor& other , Scalar alpha){
    Tensor out =  at::native::empty_opencl(self.sizes(), self.options());
    opencl_sub_out(out, self, other, alpha);
    return out;
}

// MUL FUNCTION


Tensor & opencl_mul_out(Tensor &out, const Tensor &self, const Tensor& other){
    auto other_ = checked_tensor_unwrap(other, "other", 1, "opencl_mul_out", false, c10::Backend::OpenCL, self.scalar_type());
    auto self_ = checked_tensor_unwrap(self, "self", 2, "opencl_mul_out", false, c10::Backend::OpenCL, self.scalar_type());
    auto out_ = checked_tensor_unwrap(out, "out", 3, "opencl_mul_out", false, c10::Backend::OpenCL, self.scalar_type());
    
    pointwise_op3(self_->storage().unsafeGetStorageImpl(), other_->storage().unsafeGetStorageImpl(), out_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::MUL, self.scalar_type());
    return out;
}

Tensor & opencl_mul_(Tensor &self, const Tensor& other){
    return opencl_mul_out(self, self, other);
}

Tensor opencl_mul(const Tensor &self, const Tensor& other){
    Tensor out =  at::native::empty_opencl(self.sizes(), self.options());
    opencl_mul_out(out, self, other);
    return out;
}

// DIV FUNCTION


Tensor & opencl_div_out(Tensor &out, const Tensor &self, const Tensor& other){
    auto other_ = checked_tensor_unwrap(other, "other", 1, "opencl_mul_out", false, c10::Backend::OpenCL, self.scalar_type());
    auto self_ = checked_tensor_unwrap(self, "self", 2, "opencl_mul_out", false, c10::Backend::OpenCL, self.scalar_type());
    auto out_ = checked_tensor_unwrap(out, "out", 3, "opencl_mul_out", false, c10::Backend::OpenCL, self.scalar_type());
    
    pointwise_op3(self_->storage().unsafeGetStorageImpl(), other_->storage().unsafeGetStorageImpl(), out_->storage().unsafeGetStorageImpl(), at::native::opencl::OpenCLOperationsPointwise3::DIV, self.scalar_type());
    return out;
}

Tensor & opencl_div_(Tensor &self, const Tensor& other){
    return opencl_div_out(self, self, other);
}

Tensor opencl_div(const Tensor &self, const Tensor& other){
    Tensor out =  at::native::empty_opencl(self.sizes(), self.options());
    opencl_div_out(out, self, other);
    return out;
}

}} // namespace at::native
