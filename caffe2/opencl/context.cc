#include "context.h"

#include <c10/opencl/OpenCLFunctions.h>
#include <c10/opencl/OpenCLGuard.h>
#include <c10/util/Logging.h>
#include <c10/util/Exception.h>

#include <vector>
#include <mutex>

namespace caffe2 {
namespace opencl {

OpenCLContext::OpenCLContext() {}

Device OpenCLContext::device() const {
    return Device{OPENCL, c10::opencl::current_device()};
}

/* Sorry for the naming, will get rid of this in future diff */
DeviceType OpenCLContext::device_type() const {
    return DeviceType::OPENCL;
}

void OpenCLContext::SwitchToDevice(int stream_id) {
    const auto& stream = c10::opencl::getCurrentOpenCLStream(stream_id);
    c10::opencl::setCurrentOpenCLStream(stream);
    c10::opencl::set_device(stream.device_index());
}

void OpenCLContext::WaitEvent(const caffe2::Event& ev) {
    const auto& cl_ev = *reinterpret_cast<cl::Event*>(ev.event_.get());
    cl_ev.wait();
}

void OpenCLContext::Record(caffe2::Event* ev, const char* err_msg) const {
    const auto device_index = ev->GetDeviceOption().device_id();
    Stream stream = Stream(Stream::DEFAULT, Device(DeviceType::OPENCL, -1));
    TORCH_CHECK(device_index == -1 || device_index == stream.device_index(),
    "Event device index ",
    device_index,
    " does not match recording stream's device index ",
    stream.device_index(),
    ".");

    cl::Event* opencl_event = static_cast<cl::Event*>(ev->event_.get());
    c10::opencl::OpenCLStream opencl_stream{stream};

    // Moves to stream's device to record
    c10::opencl::OpenCLStreamGuard guard{stream};
    opencl_stream.stream()->enqueueMarkerWithWaitList(NULL, opencl_event);
}

void OpenCLContext::FinishDeviceComputation() {
    c10::opencl::getCurrentOpenCLStream().synchronize();
}

// shared mutex to lock out alloc / free during NCCL launches
std::mutex& OpenCLContext::mutex() {
  static std::mutex m;
  return m;
}

void OpenCLContext::CopyBytesSameDevice(
    size_t nbytes,
    const void* src,
    void* dst) {
    c10::opencl::OpenCLGuard guard{c10::opencl::current_device()};
    c10::opencl::OpenCLStream stream = c10::opencl::getCurrentOpenCLStream();
    cl_int err = stream.stream()->enqueueCopyBuffer(*(cl::Buffer*)src,
        *(cl::Buffer*)dst,
        /* src_offset*/ 0,
        /* dst_offset */ 0,
        nbytes,
        NULL,
        NULL);
    TORCH_CHECK(err != CL_SUCCESS, "OpenCL Error : cannot copy bytes from CPU to OpenCL device.");
}

void OpenCLContext::CopyBytesFromCPU(size_t nbytes, const void* src, void* dst) {
    c10::opencl::OpenCLStream stream = c10::opencl::getCurrentOpenCLStream();
    cl_int err = stream.stream()->enqueueWriteBuffer(*(cl::Buffer*)dst,
        /* blocking */ CL_TRUE,
        /* offset */ 0,
        nbytes,
        src,
        NULL,
        NULL);
    TORCH_CHECK(err != CL_SUCCESS, "OpenCL Error : cannot copy bytes from CPU to OpenCL device.");
}

void OpenCLContext::CopyBytesToCPU(size_t nbytes, const void* src, void* dst) {
    c10::opencl::OpenCLStream stream = c10::opencl::getCurrentOpenCLStream();
    cl_int err = stream.stream()->enqueueReadBuffer(*(cl::Buffer*)src,
        /* blocking */ CL_TRUE,
        /* offset */ 0,
        nbytes,
        dst,
        NULL,
        NULL);
    TORCH_CHECK(err != CL_SUCCESS, "OpenCL Error : cannot copy bytes from CPU to OpenCL device.");
}

void OpenCLContext::CopyBytesAsync(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device) {
    c10::opencl::OpenCLStream stream = c10::opencl::getCurrentOpenCLStream();
    cl_int err = stream.stream()->enqueueReadBuffer(*(cl::Buffer*)src,
        /* blocking */ CL_TRUE,
        /* offset */ 0,
        nbytes,
        dst,
        NULL,
        NULL);
    TORCH_CHECK(err != CL_SUCCESS, "OpenCL Error : cannot copy bytes from CPU to OpenCL device.");
}

void OpenCLContext::CopyBytesSync(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device) {}

} // namespace opencl

struct DefaultOpenCLAllocator final : public at::Allocator {
    DefaultOpenCLAllocator() {}
    ~DefaultOpenCLAllocator() override {}
    at::DataPtr allocate(size_t nbytes) const override {
        // Lock the mutex
        std::lock_guard<std::mutex> lock(opencl::OpenCLContext::mutex());
        void* ptr = nullptr;

        if (nbytes != 0) {
            cl_int err;
            ptr = new cl::Buffer{CL_MEM_READ_WRITE, nbytes, NULL, &err};
            TORCH_CHECK(err != CL_SUCCESS, "OpenCL Error : Cannot allocate buffer of ", nbytes, " byte(s). (", err, ")");
        }
        return {ptr, ptr, &Delete, at::Device(OPENCL, c10::opencl::current_device())};
    }

    at::DeleterFnPtr raw_deleter() const override {
        return &Delete;
    }

private:
    static void Delete(void* ptr) {
        // lock the mutex
        std::lock_guard<std::mutex> lock(opencl::OpenCLContext::mutex());

        // If memory pool is not set up, use simple cudaFree.
        delete reinterpret_cast<cl::Buffer*>(ptr);
    }
};

static DefaultOpenCLAllocator g_opencl_alloc;
REGISTER_ALLOCATOR(OPENCL, &g_opencl_alloc);

} // namespace caffe2

namespace at {

REGISTER_CONTEXT(DeviceType::OPENCL, caffe2::opencl::OpenCLContext);

} // namespace at
