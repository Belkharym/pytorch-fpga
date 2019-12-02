#include "context.h"

#include <c10/opencl/OpenCLFunctions.h>
#include <c10/opencl/OpenCLCachingAllocator.h>
#include <c10/opencl/OpenCLGuard.h>
#include <c10/util/Logging.h>
#include <c10/util/Exception.h>
#include <c10/core/CopyBytes.h>
#include <ATen/opencl/PinnedMemoryAllocator.h>
#include <aten/src/ATen/native/opencl/Utils.h>

#include <vector>
#include <mutex>
#include <memory>

#ifdef _WIN32
#define ALIGNED_MALLOC(size, alignment) ::_aligned_malloc(size, alignment)
#else
#define ALIGNED_MALLOC(size, alignment) ::aligned_alloc(alignment, size)
#endif // _WIN32

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

static inline cl::Buffer* toBuffer(void *ptr) {
  auto ret = caffe2::opencl::getBufferFromPtr(ptr);
  if (!ret) {
    ret = at::opencl::OpenCLCachingHostAllocator_getBuffer(ptr);
  }
  return static_cast<cl::Buffer*>(ret ? &ret.value().get() : nullptr);
}

void OpenCLContext::CopyBytesSameDevice(
    size_t nbytes,
    const void* src,
    void* dst) {
    c10::opencl::OpenCLGuard guard{c10::opencl::current_device()};
    c10::opencl::OpenCLStream stream = c10::opencl::getCurrentOpenCLStream();
    cl_int err = stream.stream()->enqueueCopyBuffer(*toBuffer(const_cast<void*>(src)),
        *toBuffer(dst),
        /* src_offset*/ 0,
        /* dst_offset */ 0,
        nbytes,
        NULL,
        NULL);
    TORCH_CHECK(err == CL_SUCCESS, "OpenCL Error : cannot copy bytes from OpenCL device to OpenCL device.");
    C10_OPENCL_CHECK(at::native::syncOpenCLPointer(dst, stream));
}

void OpenCLContext::CopyBytesFromCPU(size_t nbytes, const void* src, void* dst) {
    c10::opencl::OpenCLStream stream = c10::opencl::getCurrentOpenCLStream();
    cl_int err = stream.stream()->enqueueWriteBuffer(*toBuffer(dst),
        /* blocking */ CL_TRUE,
        /* offset */ 0,
        nbytes,
        src,
        NULL,
        NULL);
    TORCH_CHECK(err == CL_SUCCESS, "OpenCL Error : cannot copy bytes from CPU to OpenCL device.");
    C10_OPENCL_CHECK(at::native::syncOpenCLPointer(dst, stream));
}

void OpenCLContext::CopyBytesToCPU(size_t nbytes, const void* src, void* dst) {
    c10::opencl::OpenCLStream stream = c10::opencl::getCurrentOpenCLStream();
    cl_int err = stream.stream()->enqueueReadBuffer(*toBuffer(const_cast<void*>(src)),
        /* blocking */ CL_TRUE,
        /* offset */ 0,
        nbytes,
        dst,
        NULL,
        NULL);
    TORCH_CHECK(err == CL_SUCCESS, "OpenCL Error : cannot copy bytes from OpenCL device to CPU.");
}

void OpenCLContext::CopyBytesAsync(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device) {
    if (nbytes == 0) return;
    cl_int err = CL_SUCCESS;
    switch (src_device.type()) {
        case OPENCL: {
            c10::opencl::OpenCLStream stream = c10::opencl::getStreamFromPool(src_device.index());
            c10::opencl::OpenCLStreamGuard guard{stream};
            if (dst_device.type() == OPENCL) {
                err = stream.stream()->enqueueCopyBuffer(*toBuffer(const_cast<void*>(src)), *toBuffer(dst), 0, 0, nbytes, NULL, NULL);
                C10_OPENCL_CHECK(at::native::syncOpenCLPointer(dst, stream));
            }
            else {
                memcpy(dst, src, nbytes);
                err = stream.stream()->enqueueReadBuffer(*toBuffer(const_cast<void*>(src)), CL_FALSE, 0, nbytes, dst, NULL, NULL);
            }
            break;
        }
        case CPU: {
            if (dst_device.type() == OPENCL) {
                c10::opencl::OpenCLStream stream = c10::opencl::getStreamFromPool(dst_device.index());
                c10::opencl::OpenCLStreamGuard guard{stream};
                memcpy(dst, src, nbytes);
                err = stream.stream()->enqueueWriteBuffer(*toBuffer(dst), CL_FALSE, 0, nbytes, dst, NULL, NULL);
            }
            else {
                err = CL_INVALID_MEM_OBJECT;
            }
            break;
        }
        default: {
            err = CL_INVALID_MEM_OBJECT;
            break;
        }
    }
    TORCH_CHECK(err == CL_SUCCESS, "OpenCL Error : cannot copy bytes from device ",
        DeviceTypeName(src_device.type()),
        " to device ",
        DeviceTypeName(src_device.type()));
}

void OpenCLContext::CopyBytesSync(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device) {
    if (nbytes == 0) return;
    cl_int err = CL_SUCCESS;
    switch (src_device.type()) {
        case OPENCL: {
            c10::opencl::OpenCLStream stream = c10::opencl::getStreamFromPool(src_device.index());
            c10::opencl::OpenCLStreamGuard guard{stream};
            if (dst_device.type() == OPENCL) {
                cl::Event cl_ev;
                err = stream.stream()->enqueueCopyBuffer(*toBuffer(const_cast<void*>(src)), *toBuffer(dst), 0, 0, nbytes, NULL, &cl_ev);
                C10_OPENCL_CHECK(at::native::syncOpenCLPointer(dst, stream));
                cl_ev.wait();
            }
            else {
                memcpy(dst, src, nbytes);
                err = stream.stream()->enqueueReadBuffer(*toBuffer(const_cast<void*>(src)), CL_TRUE, 0, nbytes, dst, NULL, NULL);
            }
            break;
        }
        case CPU: {
            if (dst_device.type() == OPENCL) {
                c10::opencl::OpenCLStream stream = c10::opencl::getStreamFromPool(dst_device.index());
                c10::opencl::OpenCLStreamGuard guard{stream};
                memcpy(dst, src, nbytes);
                err = stream.stream()->enqueueWriteBuffer(*toBuffer(dst), CL_TRUE, 0, nbytes, src, NULL, NULL);
            }
            else {
                err = CL_INVALID_MEM_OBJECT;
            }
            break;
        }
        default: {
            err = CL_INVALID_MEM_OBJECT;
            break;
        }
    }
    TORCH_CHECK(err == CL_SUCCESS, "OpenCL Error : cannot copy bytes from device ",
        DeviceTypeName(src_device.type()),
        " to device ",
        DeviceTypeName(src_device.type()));
}

} // namespace opencl

struct OpenCLPtrContext {
    void* data;
    cl::Buffer* buf;
    size_t nbytes;
};

struct DefaultOpenCLAllocator final : public at::Allocator {
    DefaultOpenCLAllocator() {}
    ~DefaultOpenCLAllocator() override {}
    at::DataPtr allocate(size_t nbytes) const override {
        OpenCLPtrContext* ctx;

        if (nbytes != 0) {
            ctx = new OpenCLPtrContext();
            ctx->data = nullptr;
            ctx->buf = nullptr;
            ctx->nbytes = nbytes;

            cl_int err;
            ctx->data = ALIGNED_MALLOC(nbytes, alignof(cl_long16));
            TORCH_INTERNAL_ASSERT(ctx->data, "Cannot allocate ", nbytes, " byte(s) of memory for OpenCL buffer.");
            ctx->buf = new cl::Buffer{c10::opencl::opencl_context(), CL_MEM_READ_WRITE, nbytes, NULL, &err};
            TORCH_CHECK(err == CL_SUCCESS, "OpenCL Error : Cannot allocate Buffer of ", nbytes, " byte(s). (", ::c10::opencl::clErrorString(err), ")");
            {
                // Lock the mutex
                std::lock_guard<std::mutex> lock(opencl::OpenCLContext::mutex());
                if (buffers.find(ctx->data) != buffers.end()) {
                    TORCH_WARN("Address ", ctx->data, " was reallocated an other time. This shouldn't be happening... "
                    "The reallocation was dropped.");
                    ::free(ctx->data);
                    delete ctx->buf;
                    delete ctx;
                    return {nullptr, nullptr, &Delete, at::Device(OPENCL, c10::opencl::current_device())};
                }
                auto res = buffers.emplace(ctx->data, ctx);
            }
            return {ctx->data, ctx->data, &Delete, at::Device(OPENCL, c10::opencl::current_device())};
        }
        return {nullptr, nullptr, &Delete, at::Device(OPENCL, c10::opencl::current_device())};
    }

    at::DeleterFnPtr raw_deleter() const override {
        return &Delete;
    }

private:
    static std::unordered_map<void*, OpenCLPtrContext*> buffers;
    static void Delete(void* ptr) {
        if (ptr == nullptr) return;

        OpenCLPtrContext* ctx = nullptr;
        {
            // lock the mutex
            std::lock_guard<std::mutex> lock(opencl::OpenCLContext::mutex());
            auto it = std::find_if(buffers.begin(), buffers.end(), [&](std::pair<void*, OpenCLPtrContext*> p) {
                return ptr >= p.second->data && (intptr_t)ptr < ((intptr_t)p.second->data) + (intptr_t)p.second->nbytes;
            });
            if (it != buffers.end()) {
                ctx = it->second;
                buffers.erase(it);
            }
        }
        if (ctx != nullptr) {
            at::native::syncOpenCLPointer(ctx->data);
            c10::opencl::getCurrentOpenCLStream().synchronize();
            if (ctx->buf) delete ctx->buf;
            ctx->buf = nullptr;
            if (ctx->data) ::free(ctx->data);
            ctx->data = nullptr;
            delete ctx;
        }
    }
    friend c10::optional<std::reference_wrapper<cl::Buffer>> caffe2::opencl::getBufferFromPtr(void *ptr, size_t * size);
};

std::unordered_map<void*, OpenCLPtrContext*> DefaultOpenCLAllocator::buffers;
static DefaultOpenCLAllocator g_opencl_alloc;
REGISTER_ALLOCATOR(OPENCL, &g_opencl_alloc);

namespace opencl {

c10::optional<std::reference_wrapper<cl::Buffer>> getBufferFromPtr(void *ptr, size_t * size) {
    std::lock_guard<std::mutex> lock(opencl::OpenCLContext::mutex());
    auto it = std::find_if(DefaultOpenCLAllocator::buffers.begin(), DefaultOpenCLAllocator::buffers.end(), [&](std::pair<void*, OpenCLPtrContext*> p) {
        return ptr >= p.second->data && (intptr_t)ptr < ((intptr_t)p.second->data) + (intptr_t)p.second->nbytes;
    });
    if (it == DefaultOpenCLAllocator::buffers.end()) {
        return c10::nullopt;
    }
    if (size) *size = it->second->nbytes;
    return {*it->second->buf};
}

}

} // namespace caffe2

namespace at {

REGISTER_COPY_BYTES_FUNCTION(
    DeviceType::OPENCL,
    DeviceType::OPENCL,
    caffe2::opencl::OpenCLContext::CopyBytesSync,
    caffe2::opencl::OpenCLContext::CopyBytesAsync);

REGISTER_COPY_BYTES_FUNCTION(
    DeviceType::OPENGL,
    DeviceType::CPU,
    caffe2::opencl::OpenCLContext::CopyBytesSync,
    caffe2::opencl::OpenCLContext::CopyBytesAsync);

REGISTER_COPY_BYTES_FUNCTION(
    DeviceType::CPU,
    DeviceType::OPENCL,
    caffe2::opencl::OpenCLContext::CopyBytesSync,
    caffe2::opencl::OpenCLContext::CopyBytesAsync);

REGISTER_CONTEXT(DeviceType::OPENCL, caffe2::opencl::OpenCLContext);

} // namespace at
