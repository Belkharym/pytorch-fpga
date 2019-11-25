#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/opencl/OpenCLContext.h>
#include <ATen/opencl/OpenCLEvent.h>
#include <c10/opencl/OpenCLStream.h>
#include <c10/opencl/OpenCLGuard.h>
#include <ATen/opencl/PinnedMemoryAllocator.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/opencl/Utils.h>
#include <ATen/native/opencl/OpenCLOperations.h>

namespace at {
namespace native {

using namespace at::opencl;

// device-to-device copy, does type conversion
static void copy_device_to_device(TensorIterator& iter, bool non_blocking) {
  int64_t numel = iter.numel();

  // We can memcpy the memory if both tensors have the same type AND both
  // tensors are contiguous after dimension coalescing and reordering.
  bool same_type = iter.dtype(0) == iter.dtype(1);
  bool memcpy_eligible = same_type && iter.is_contiguous();

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  OpenCLGuard device_guard(src_device);

  // We always perform the copy on the source device, using the current stream
  // on the source device, and we fully synchronize on both src and dst's
  // current streams for completion of the copy. We have to explicitly do this
  // for non-contig copies. This mimics the behavior of cross-device
  // cudaMemcpyAsync on the default stream.
  OpenCLStream copy_stream = getCurrentOpenCLStream(src_device.index());
  if (src_device != dst_device) {
    // This is a cross-device copy on the src current stream and dst current
    // stream. We perform a two-way barrier between both devices' streams
    // before the copy. This ensures that any write-after-write and
    // write-after-read dependencies on the destination side are handled, so
    // that no one is operating on the dst memory when we perform the copy.
    // src waits on dst barrier (src already waits on src)
    OpenCLEvent dst_ready;
    device_guard.set_device(dst_device);
    dst_ready.record(getCurrentOpenCLStream(dst_device.index()));

    device_guard.set_device(src_device);
    dst_ready.block(copy_stream);
  }

  if (memcpy_eligible) {
    // Perform the copy
    AT_OPENCL_CHECK(copy_stream.stream()->enqueueCopyBuffer(
        *toBuffer(iter.data_ptr(1)),
        *toBuffer(iter.data_ptr(0)),
        0,
        0,
        numel));
  } else {
    auto kernel_name = "cast";
    auto cast_kernel = opencl_kernel_func<OpenCLCastFunctor>(kernel_name, cl::EnqueueArgs{*copy_stream.stream(), cl::NDRange{(size_t)numel}, 1});
    AT_OPENCL_CHECK(cast_kernel(
        *toBuffer(iter.data_ptr(1)),
        *toBuffer(iter.data_ptr(0)),
        getOpenCLKernelCastType(iter.dtype(1)),
        getOpenCLKernelCastType(iter.dtype(0))));
  }
  AT_OPENCL_CHECK(syncOpenCLPointer(iter.data_ptr(0)));

  if (src_device != dst_device) {
    // dst waits on src barrier (dst already waits on dst). We cannot
    // operate on dst's copy until the copy is complete.

    // Still on src_device, record stream event
    OpenCLEvent src_ready;
    src_ready.record(copy_stream);

    device_guard.set_device(dst_device);
    src_ready.block(getCurrentOpenCLStream(dst_device.index()));
  }
  if (!non_blocking) {
    copy_stream.synchronize();
  }
}

static bool copy_requires_temporaries(TensorIterator& iter) {
  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  if (dst_device == src_device) {
    // We never require temporaries for copies on the same Device.
    TORCH_INTERNAL_ASSERT(dst_device.is_opencl() && src_device.is_opencl());
    return false;
  }

  bool same_dtype = iter.dtype(0) == iter.dtype(1);
  if (same_dtype && iter.is_contiguous()) {
    // Contiguous same-dtype copies can always use enqueueCopyBuffer
    return false;
  } else {
    // The remaining cases require temporaries. For example, this includes
    // non-contiguous copies between CPU and Device.
    return true;
  }
}

static void copy_kernel_opencl(TensorIterator& iter, bool non_blocking) {
  AT_ASSERT(iter.ntensors() == 2);

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  if (copy_requires_temporaries(iter)) {
    // NB: this involves recursive calls to copy. Be careful that those copies
    // don't require temporaries or you will cause an infinite recursion!
    auto& dst = iter.tensor(0);
    Tensor dst_contig;
    Tensor src_contig;

    // Type conversions are performed on the CPU for CPU-Device copies and on
    // the src device for Device-Device copies.
    if (iter.device_type(0) == kOPENCL) {
      dst_contig = dst.is_contiguous() ? dst : at::empty_like(dst);
      src_contig = iter.tensor(1).to(iter.dtype(0)).expand_as(dst).contiguous();
    } else {
      bool same_type = iter.dtype(0) == iter.dtype(1);
      dst_contig = (dst.is_contiguous() && same_type) ? dst : at::empty_like(dst, iter.dtype(1));
      src_contig = iter.tensor(1).expand_as(dst).contiguous();
    }

    // perform a same-dtype copy on contiguous tensors
    TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
    TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
    dst_contig.copy_(src_contig, non_blocking);

    // if necessary, copy back into dst
    if (!dst_contig.is_same(dst)) {
      TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
      dst.copy_(dst_contig, non_blocking);
    }
    return;
  }

  // Copy on Device (or between Devices)
  if (dst_device.is_opencl() && src_device.is_opencl()) {
    copy_device_to_device(iter, non_blocking);
    return;
  }

  void* dst = iter.data_ptr(0);
  void* src = iter.data_ptr(1);
  int64_t nbytes = iter.numel() * iter.element_size(0);
  OpenCLStream stream = getCurrentOpenCLStream();

  // Copy between CPU and Device
  at::opencl::OptionalOpenCLGuard device_guard;
  if (dst_device.is_opencl() && src_device.is_cpu()) {
    device_guard.set_device(dst_device);
    stream = getCurrentOpenCLStream();
    AT_OPENCL_CHECK(stream.stream()->enqueueWriteBuffer((*toBuffer(dst)), !non_blocking, 0, nbytes, src), "Cannot write from opencl buffer [device #", dst_device, ";", c10::opencl::opencl_platform().getInfo<CL_PLATFORM_NAME>(), "]");
  } else if (dst_device.is_cpu() && src_device.is_opencl()) {
    device_guard.set_device(src_device);
    stream = getCurrentOpenCLStream();
    AT_OPENCL_CHECK(stream.stream()->enqueueReadBuffer((*toBuffer(src)), !non_blocking, 0, nbytes, dst));
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupported devices in OpenCL copy_()");
  }

  if (non_blocking) {
    void* ptr = (dst_device == kCPU ? dst : src);
    // TODO find a way to ensure that, when we try to access to the host pointer (when dst is host),
    // we block until the read is done.
    AT_OPENCL_CHECK(OpenCLCachingHostAllocator_recordEvent(ptr, stream));
    // TODO Find how CUDA do to ensure that when we use the tensors, everything is synchronized.
    if (dst_device == kCPU) stream.synchronize();
  } else {
    stream.synchronize();
  }
}

REGISTER_DISPATCH(copy_stub, &copy_kernel_opencl);

} // namespace native
} // namespace at
